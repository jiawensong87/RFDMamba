"""RFDMamba reconstruction and geometry-consistency supervision."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_sobel_kernels():
    """Return (sobel_x, sobel_y), each of shape (1, 1, 3, 3).

    Used only by the loss and the reliability pseudo-label.
    """
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
    return kx, ky


def sobel_magnitude(img: torch.Tensor,
                    sobel_x: torch.Tensor,
                    sobel_y: torch.Tensor,
                    normalize: bool = True) -> torch.Tensor:
    """Sobel gradient magnitude of a single-channel image, optionally
    normalised per image to [0, 1]. Used by (a) the gradient loss term
    and (b) the reliability pseudo-label."""
    if img.shape[1] > 1:
        img = img.mean(dim=1, keepdim=True)
    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    if normalize:
        mag = mag / (mag.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return mag


@torch.no_grad()
def build_reliability_targets(gt_dsm: torch.Tensor, rgb: torch.Tensor,
                              sobel_x: torch.Tensor, sobel_y: torch.Tensor,
                              tau_rgb: float = 0.15, tau_dsm: float = 0.05,
                              dsm_dilate: int = 2,
                              dsm_normalize: bool = False):
    """Build the positive / negative masks for reliability supervision.

    Supervision is applied only where the RGB has an edge:
      pos = RGB edge AND GT-DSM elevation jump  -> genuine boundary, label 1
      neg = RGB edge AND flat GT-DSM            -> texture/shadow distractor

    Returns (pos, neg), each [B,1,H,W] in {0,1}.

    The DSM gradient is thresholded in absolute (global-normalised) units
    (``dsm_normalize=False``), so ``tau_dsm`` is comparable across patches;
    ``edge_dsm`` is morphologically dilated by ``dsm_dilate`` to tolerate a
    few-pixel registration offset between the orthophoto and the DSM. The
    RGB side keeps per-image normalisation.
    """
    g_rgb = sobel_magnitude(rgb, sobel_x, sobel_y, normalize=True)
    g_dsm = sobel_magnitude(gt_dsm, sobel_x, sobel_y, normalize=dsm_normalize)
    edge_rgb = (g_rgb > tau_rgb).float()
    edge_dsm = (g_dsm > tau_dsm).float()
    if dsm_dilate > 0:
        k = 2 * dsm_dilate + 1
        edge_dsm = F.max_pool2d(edge_dsm, k, stride=1, padding=dsm_dilate)
    pos = edge_rgb * edge_dsm
    neg = edge_rgb * (1.0 - edge_dsm)
    return pos, neg


class HybridStructureLoss(nn.Module):
    """L_total = w_l1 * L1 + w_grad * L_grad + w_rel * L_rel.

    L_rel applies a class-balanced masked BCE to the reliability maps at
    all scales, against the geometry-consistency pseudo-label. Pseudo-label
    masks are downsampled by adaptive max-pool (preserving thin edges); a
    coarse pixel with both positive and negative evidence is treated as
    positive (conservative, favouring structure retention).

    The optional ``rgb`` argument enables L_rel; if not provided (or
    ``use_rel=False``), L_rel is zero.
    """

    def __init__(self, w_l1: float = 1.0, w_grad: float = 0.5,
                 w_rel: float = 0.2,
                 use_grad: bool = True, use_rel: bool = True,
                 tau_rgb: float = 0.15, tau_dsm: float = 0.05,
                 dsm_dilate: int = 2, dsm_normalize: bool = False):
        super().__init__()
        self.w_l1, self.w_grad, self.w_rel = w_l1, w_grad, w_rel
        self.use_grad, self.use_rel = use_grad, use_rel
        self.tau_rgb, self.tau_dsm = tau_rgb, tau_dsm
        self.dsm_dilate, self.dsm_normalize = dsm_dilate, dsm_normalize

        kx, ky = make_sobel_kernels()
        self.register_buffer('sobel_x', kx)
        self.register_buffer('sobel_y', ky)

    def _gradient_loss(self, pred, gt):
        p_x = F.conv2d(pred, self.sobel_x, padding=1)
        p_y = F.conv2d(pred, self.sobel_y, padding=1)
        g_x = F.conv2d(gt, self.sobel_x, padding=1)
        g_y = F.conv2d(gt, self.sobel_y, padding=1)
        return F.l1_loss(p_x, g_x) + F.l1_loss(p_y, g_y)

    def _reliability_loss(self, info_list, target, rgb):
        pos_full, neg_full = build_reliability_targets(
            target, rgb, self.sobel_x, self.sobel_y,
            tau_rgb=self.tau_rgb, tau_dsm=self.tau_dsm,
            dsm_dilate=self.dsm_dilate, dsm_normalize=self.dsm_normalize)

        total, count = target.new_tensor(0.0), 0
        for info in info_list:
            ssm_info = info.get('ssm_info')
            if not ssm_info:
                continue
            r = ssm_info.get('reliability_map')
            if r is None:                      # PlainSSM / w/o reliability -> skip
                continue
            size = r.shape[-2:]
            pos = F.adaptive_max_pool2d(pos_full, size)
            neg = F.adaptive_max_pool2d(neg_full, size)
            mask = (pos + neg).clamp(max=1.0)
            label = pos                        # positive precedence (conservative)
            if mask.sum() < 1:
                continue
            bce = F.binary_cross_entropy(r.clamp(1e-6, 1 - 1e-6), label,
                                         reduction='none')
            # Class balancing: positive and negative classes each take half
            # the effective weight.
            n_eff = mask.sum()
            n_pos = (label * mask).sum().clamp(min=1.0)
            n_neg = (n_eff - n_pos).clamp(min=1.0)
            w = (label * (n_eff / (2.0 * n_pos))
                 + (1.0 - label) * (n_eff / (2.0 * n_neg)))
            total = total + (bce * w * mask).sum() / (n_eff + 1e-6)
            count += 1
        return total / count if count > 0 else total

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                info_list: list, rgb: torch.Tensor = None) -> dict:
        dev = pred.device
        losses = {'l1': F.l1_loss(pred, target)}
        losses['grad'] = (self._gradient_loss(pred, target) if self.use_grad
                          else torch.tensor(0.0, device=dev))
        losses['rel'] = (self._reliability_loss(info_list, target, rgb)
                         if (self.use_rel and rgb is not None and info_list)
                         else torch.tensor(0.0, device=dev))
        losses['total'] = (self.w_l1 * losses['l1']
                           + self.w_grad * losses['grad']
                           + self.w_rel * losses['rel'])
        return losses
