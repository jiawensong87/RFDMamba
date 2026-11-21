"""RFDMamba: Reliability-Aware State-Space Propagation and
Frequency-Disentangled Fusion for Optical-Image-Guided DSM Super-Resolution.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

_SCAN_IMPORT_ERROR = None
try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn as _selective_scan_cuda,
    )
except (ImportError, OSError) as exc:
    _selective_scan_cuda = None
    _SCAN_IMPORT_ERROR = exc


def _selective_scan_reference(
    u, delta, A, B, C, D, z=None, delta_bias=None,
    delta_softplus=True, return_last_state=False,
):
    """Slow grouped selective-scan recurrence for small CPU smoke tests."""
    if z is not None or return_last_state:
        raise NotImplementedError("Unsupported scan options in the CPU reference.")
    b, kd, l = u.shape
    k = B.shape[1]
    d = kd // k
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta)
    u_ = u.reshape(b, k, d, l)
    dt = delta.reshape(b, k, d, l)
    A_ = A.view(k, d, -1)                      # (K, D, N)
    x = u.new_zeros(b, k, d, A_.shape[-1])     # (B, K, D, N)
    ys = []
    for i in range(l):
        dA = torch.exp(dt[..., i].unsqueeze(-1) * A_.unsqueeze(0))
        dBu = (dt[..., i].unsqueeze(-1)
               * B[..., i].unsqueeze(2)
               * u_[..., i].unsqueeze(-1))
        x = x * dA + dBu
        ys.append((x * C[..., i].unsqueeze(2)).sum(-1))
    y = torch.stack(ys, dim=-1).view(b, kd, l)
    y = y + D.view(1, -1, 1) * u
    return y


def get_norm(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm that gracefully handles ``channels < num_groups``."""
    if channels < num_groups:
        num_groups = channels
    return nn.GroupNorm(num_groups, channels)


def make_directional_kernels():
    """4-direction gradient kernels (4, 1, 3, 3): horizontal, vertical,
    and the two diagonals. Used by :class:`StructureExtractor`."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    ky = kx.t().contiguous()
    kd1 = torch.tensor([[0., 1., 2.], [-1., 0., 1.], [-2., -1., 0.]])
    kd2 = torch.tensor([[-2., -1., 0.], [-1., 0., 1.], [0., 1., 2.]])
    return torch.stack([kx, ky, kd1, kd2]).unsqueeze(1)


def make_haar_kernels():
    """Haar decomposition kernels (4, 1, 2, 2) in order LL, LH, HL, HH."""
    ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
    hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
    hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
    return torch.stack([ll, lh, hl, hh]).unsqueeze(1)


class SS2D(nn.Module):
    """Four-direction 2-D selective scan (SS2D), summed in image order."""

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2.,
                 dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random",
                 dt_scale=1.0, dt_init_floor=1e-4, dropout=0.,
                 conv_bias=True, bias=False, device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner,
                                groups=self.d_inner, bias=conv_bias,
                                kernel_size=d_conv, padding=(d_conv - 1) // 2,
                                **factory_kwargs)
        self.act = nn.SiLU()

        x_projs = [nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2,
                             bias=False, **factory_kwargs) for _ in range(4)]
        self.x_proj_weight = nn.Parameter(torch.stack([p.weight for p in x_projs]))

        dt_projs = [self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init,
                                  dt_min, dt_max, dt_init_floor, **factory_kwargs)
                    for _ in range(4)]
        self.dt_projs_weight = nn.Parameter(torch.stack([p.weight for p in dt_projs]))
        self.dt_projs_bias = nn.Parameter(torch.stack([p.bias for p in dt_projs]))

        self.A_logs = self._A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self._D_init(self.d_inner, copies=4, merge=True)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **fkw):
        proj = nn.Linear(dt_rank, d_inner, bias=True, **fkw)
        std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(proj.weight, std)
        else:
            nn.init.uniform_(proj.weight, -std, std)
        dt = torch.exp(
            torch.rand(d_inner, **fkw) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            proj.bias.copy_(inv_dt)
        proj.bias._no_reinit = True
        return proj

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                   "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        p = nn.Parameter(A_log)
        p._no_weight_decay = True
        return p

    @staticmethod
    def _D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n -> r n", r=copies)
            if merge:
                D = D.flatten(0, 1)
        p = nn.Parameter(D)
        p._no_weight_decay = True
        return p

    def _forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L, K = H * W, 4
        x_hw = x.view(B, -1, L)
        x_wh = torch.transpose(x, 2, 3).contiguous().view(B, -1, L)
        x_hwwh = torch.stack([x_hw, x_wh], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)
        scan = _selective_scan_cuda if xs.is_cuda else _selective_scan_reference
        if scan is None:
            raise RuntimeError(
                "CUDA inference requires the compiled mamba-ssm selective-scan kernel."
            ) from _SCAN_IMPORT_ERROR
        out_y = scan(
            xs, dts, As, Bs, Cs, Ds,
            z=None, delta_bias=dt_bias, delta_softplus=True, return_last_state=False,
        ).view(B, K, -1, L)
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), 2, 3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), 2, 3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = self.act(self.conv2d(x.permute(0, 3, 1, 2).contiguous()))
        y1, y2, y3, y4 = self._forward_core(x)
        y = (y1 + y2 + y3 + y4).transpose(1, 2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y) * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class CrossScanSSM(nn.Module):
    """BCHW -> BCHW wrapper around :class:`SS2D`."""

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 3, expand: int = 2):
        super().__init__()
        self.ss2d = SS2D(d_model=dim, d_state=d_state, d_conv=d_conv, expand=float(expand))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ss2d(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, mid), nn.ReLU(True),
            nn.Linear(mid, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False), get_norm(channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False), get_norm(channels),
        )

    def forward(self, x):
        return F.gelu(x + self.conv(x))


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False), get_norm(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), get_norm(out_ch), nn.GELU(),
        )
        self.down = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.skip_proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        x = self.conv(x) + self.skip_proj(x)
        skip = x
        x = self.down(x)
        return x, skip


class DecoderBlock(nn.Module):
    """Decode using the fused skip and effective reliability as inputs."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_ch: int = 1):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False), get_norm(in_ch), nn.GELU(),
        )
        self.reduce = nn.Conv2d(in_ch + skip_ch + cond_ch, out_ch, 1, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), get_norm(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), get_norm(out_ch), nn.GELU(),
        )

    def forward(self, x, skip, cond):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        if cond.shape[2:] != skip.shape[2:]:
            cond = F.interpolate(cond, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = self.reduce(torch.cat([x, skip, cond], dim=1))
        return self.conv(x)


class StructureExtractor(nn.Module):
    """RGB feature -> structural embedding + single-channel structure map.

    A 1x1 projection to a low dimension, per-channel fixed 4-direction
    gradients (non-learnable, keeping the "structure" semantics
    interpretable), then a learnable 1x1 aggregation. ``struct_map`` is
    the per-image-normalised mean of the directional responses, used for
    visualisation and as a condition for the reliability head.
    """

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.mid_ch = mid_ch
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False), get_norm(mid_ch), nn.GELU(),
        )
        self.register_buffer('dir_kernels', make_directional_kernels())  # (4,1,3,3)
        self.agg = nn.Sequential(
            nn.Conv2d(mid_ch * 4, out_ch, 1, bias=False), get_norm(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), get_norm(out_ch), nn.GELU(),
        )

    def forward(self, rgb_feat: torch.Tensor):
        f = self.proj(rgb_feat)                                   # [B, mid, H, W]
        w = self.dir_kernels.repeat(self.mid_ch, 1, 1, 1)         # (4*mid, 1, 3, 3)
        g = F.conv2d(f, w, padding=1, groups=self.mid_ch)         # [B, 4*mid, H, W]
        g = torch.abs(g)
        struct_map = g.mean(dim=1, keepdim=True)
        struct_map = struct_map / (struct_map.amax(dim=(2, 3), keepdim=True) + 1e-6)
        struct_feat = self.agg(g)
        return struct_feat, struct_map


class _GREASSP(nn.Module):
    """Guidance Reliability Estimation (GRE) and Anchored State-Space Propagation (ASSP)."""

    def __init__(self, channels: int, d_state: int = 16):
        super().__init__()
        self.reliability_floor = 0.1

        mid_q = max(channels // 4, 1)
        mid_h = max(channels // 2, 1)

        self.cs_ssm = CrossScanSSM(channels, d_state=d_state)

        # RGB structure candidate extraction.
        self.extractor = StructureExtractor(channels, mid_q, mid_h)

        cond_ch = channels + mid_h + 1  # dsm + struct_feat + struct_map

        # GRE head (supervised by L_rel).
        self.reliability_head = nn.Sequential(
            nn.Conv2d(cond_ch, mid_q, 3, 1, 1, bias=False),
            get_norm(mid_q), nn.GELU(),
            nn.Conv2d(mid_q, 1, 1),
            nn.Sigmoid(),
        )

        # Start training with zero optical injection.
        self.inject_head = nn.Sequential(
            nn.Conv2d(mid_h, channels, 3, 1, 1, bias=False),
            get_norm(channels), nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.zeros_(self.inject_head[-1].weight)
        nn.init.zeros_(self.inject_head[-1].bias)

        # Output refine + DSM-dominant residual.
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            get_norm(channels), nn.GELU(),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, dsm_feat: torch.Tensor, rgb_feat: torch.Tensor):
        struct_feat, struct_map = self.extractor(rgb_feat)
        cond = torch.cat([dsm_feat, struct_feat, struct_map], dim=1)
        # Keep the training graph used in the experiments: the GRE
        # condition is detached, but the injected structure is not.
        reliability = self.reliability_head(cond.detach())
        eff_r = self.reliability_floor + (1.0 - self.reliability_floor) * reliability
        inject = self.inject_head(struct_feat) * eff_r
        ssm_out = self.cs_ssm(dsm_feat + inject)
        output = dsm_feat + self.res_scale * self.refine(ssm_out)

        info = {
            "rgb_structure": struct_map,
            "reliability_map": reliability,
            "effective_reliability": eff_r,
            "inject_feat": inject,
            "ssm_out": ssm_out,
            "res_scale": self.res_scale.detach(),
            "injection_mode": "pre",
        }
        return output, info


class FrequencyDisentangledFusion(nn.Module):
    """Frequency-Disentangled Fusion (FDF), without a reliability gate."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.register_buffer('haar_kernels', make_haar_kernels())  # (4,1,2,2)

        self.hf_proj = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            get_norm(channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            get_norm(channels), nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1, bias=False),
            get_norm(channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            get_norm(channels), nn.GELU(),
        )
        ll_mid = max(channels // 4, 1)
        self.ll_proj = nn.Sequential(
            nn.Conv2d(channels, ll_mid, 1, bias=False),
            get_norm(ll_mid), nn.GELU(),
            nn.Conv2d(ll_mid, channels, 1, bias=False),
        )
        self.ll_scale = nn.Parameter(torch.tensor(0.1))

    def _haar(self, x: torch.Tensor):
        B, C, H, W = x.shape
        xp = F.pad(x, (0, 1, 0, 1), mode='replicate')
        w = self.haar_kernels.repeat(C, 1, 1, 1)                  # (4C,1,2,2)
        y = F.conv2d(xp, w, groups=C).view(B, C, 4, H, W)
        ll, lh, hl, hh = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
        return ll, torch.cat([lh, hl, hh], dim=1)

    def forward(self, rgb_feat: torch.Tensor, dsm_feat: torch.Tensor):
        ll, hf_bands = self._haar(rgb_feat)
        hf = self.hf_proj(hf_bands)
        fused = self.fusion(torch.cat([dsm_feat, hf], dim=1)) + dsm_feat
        output = fused + self.ll_scale * self.ll_proj(ll)
        return output, {
            'rgb_ll': ll,
            'rgb_hf': hf,
            'll_scale': self.ll_scale.detach(),
        }


class _ASSPFDFBlock(nn.Module):
    """Apply GRE/ASSP, then fuse optical detail and context without gating FDF."""

    def __init__(self, channels: int):
        super().__init__()
        self.pre_ssm = _GREASSP(channels)
        self.fdf = FrequencyDisentangledFusion(channels)

    def forward(self, rgb_feat: torch.Tensor, dsm_feat: torch.Tensor):
        ssm_out, ssm_info = self.pre_ssm(dsm_feat, rgb_feat)
        output, fdf_info = self.fdf(rgb_feat, ssm_out)
        return output, {"ssm_info": ssm_info, "fdf_info": fdf_info}


class RFDMamba(nn.Module):
    """Paper configuration for optical-image-guided DSM super-resolution."""

    def __init__(self) -> None:
        super().__init__()
        rgb_in, dsm_in = 3, 1
        self.chs = [32, 64, 128, 160]
        self.num_levels = num_levels = len(self.chs) - 1

        self.rgb_proj = nn.Conv2d(rgb_in, self.chs[0], 3, 1, 1)
        self.dsm_proj = nn.Conv2d(dsm_in, self.chs[0], 3, 1, 1)

        self.rgb_enc = nn.ModuleList()
        self.dsm_enc = nn.ModuleList()
        self.fusions = nn.ModuleList()
        for i in range(num_levels):
            self.rgb_enc.append(EncoderBlock(self.chs[i], self.chs[i + 1]))
            self.dsm_enc.append(EncoderBlock(self.chs[i], self.chs[i + 1]))
            self.fusions.append(_ASSPFDFBlock(self.chs[i + 1]))

        self.bottleneck = nn.Sequential(
            ResBlock(self.chs[-1]), ResBlock(self.chs[-1]), SEBlock(self.chs[-1]),
        )
        self.bottleneck_ssm = _GREASSP(self.chs[-1])

        # Learnable deep RGB contribution.
        self.rgb_deep_scale = nn.Parameter(torch.tensor(0.1))
        self.rgb_deep_contrib = nn.Sequential(
            nn.Conv2d(self.chs[-1], max(self.chs[-1] // 4, 1), 1, bias=False),
            get_norm(max(self.chs[-1] // 4, 1)), nn.GELU(),
            nn.Conv2d(max(self.chs[-1] // 4, 1), self.chs[-1], 1, bias=False),
        )

        self.decoders = nn.ModuleList()
        for i in range(num_levels):
            idx = num_levels - 1 - i
            self.decoders.append(
                DecoderBlock(self.chs[idx + 1], self.chs[idx + 1], self.chs[idx],
                             cond_ch=1)
            )

        self.head = nn.Sequential(
            nn.Conv2d(self.chs[0], self.chs[0] // 2, 3, 1, 1), nn.GELU(),
            nn.Conv2d(self.chs[0] // 2, dsm_in, 3, 1, 1),
        )

    def forward(
        self,
        optical: torch.Tensor,
        dsm_up: torch.Tensor,
        *,
        return_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict]]:
        """Reconstruct a DSM on the optical grid.

        optical: normalized [B, 3, H, W] optical image (RGB or NIR-R-G).
        dsm_up: standardized [B, 1, H, W] bicubic DSM on the same grid.
        The result uses the DSM input's standardization. With return_info,
        also return diagnostics for the three encoder stages and bottleneck.
        """
        if (
            optical.ndim != 4
            or dsm_up.ndim != 4
            or optical.shape[1] != 3
            or dsm_up.shape[1] != 1
            or optical.shape[0] != dsm_up.shape[0]
            or optical.shape[-2:] != dsm_up.shape[-2:]
        ):
            raise ValueError("Expected aligned [B, 3, H, W] and [B, 1, H, W] tensors.")
        stride = 2 ** self.num_levels
        if any(size < stride or size % stride != 0 for size in optical.shape[-2:]):
            raise ValueError(f"Height and width must be positive multiples of {stride}.")

        rgb, dsm = optical, dsm_up
        rgb = torch.nan_to_num(rgb)
        dsm = torch.nan_to_num(dsm)

        f_r = self.rgb_proj(rgb)
        f_d = self.dsm_proj(dsm)

        skips, rel_skips, infos = [], [], []
        for i in range(self.num_levels):
            f_r, s_r = self.rgb_enc[i](f_r)
            f_d, s_d = self.dsm_enc[i](f_d)
            fused_skip, info = self.fusions[i](s_r, s_d)
            skips.append(fused_skip)
            rel_skips.append(info['ssm_info'].get('effective_reliability'))
            infos.append(info)
            # CAE: feed the fused feature back into the DSM encoder.
            f_d = f_d + F.interpolate(fused_skip, scale_factor=0.5, mode="nearest")

        rgb_deep = self.rgb_deep_contrib(f_r)
        x = self.bottleneck(f_d + self.rgb_deep_scale * rgb_deep)

        x, bottleneck_info = self.bottleneck_ssm(x, f_r)
        infos.append({'ssm_info': bottleneck_info})

        for i, decoder in enumerate(self.decoders):
            idx = self.num_levels - 1 - i
            x = decoder(x, skips[idx], rel_skips[idx])

        out = dsm + self.head(x)
        return (out, infos) if return_info else out
