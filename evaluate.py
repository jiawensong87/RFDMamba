"""FP32 evaluation of RFDMamba on an existing held-out split."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from data import DSMPatches
from model import RFDMamba


def patch_metrics(gt, pred, border):
    if border:
        gt = gt[border:-border, border:-border]
        pred = pred[border:-border, border:-border]
    if min(gt.shape) < 7 or not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("Evaluation requires finite DSMs and at least a 7x7 scored area.")
    err = gt.astype(np.float64) - pred.astype(np.float64)
    gy, gx = np.gradient(gt.astype(np.float64))
    mag = np.hypot(gx, gy)
    edge = mag / (mag.max() + 1e-6) > 0.1
    data_range = float(np.ptp(gt.astype(np.float64)))
    return {
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "mae_m": float(np.mean(np.abs(err))),
        "ssim": float(structural_similarity(
            gt.astype(np.float64), pred.astype(np.float64),
            data_range=data_range if data_range >= 1e-6 else 1.0, win_size=7,
        )),
        "edge_rmse_m": float(np.sqrt(np.mean(err[edge]**2))) if edge.sum() >= 10 else None,
    }


@torch.inference_mode()
def evaluate(model, loader, device, border):
    model.eval()
    rows = []
    with torch.autocast(device_type=device.type, enabled=False):
        for optical, dsm_up, gt, names in loader:
            pred = model(optical.to(device), dsm_up.to(device)).float().cpu().numpy()
            gt = gt.numpy()
            mean, std = loader.dataset.mean, loader.dataset.std
            for i, name in enumerate(names):
                rows.append({"tile": name, **patch_metrics(
                    gt[i, 0] * std + mean, pred[i, 0] * std + mean, border,
                )})
    if not rows:
        raise ValueError("No test patches were evaluated.")
    metrics = {}
    for key in rows[0].keys() - {"tile"}:
        values = [r[key] for r in rows if r[key] is not None]
        metrics[key] = float(np.mean(values)) if values else None
    return metrics, rows


def parser_for_eval():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--scale", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dsm-mean", type=float)
    parser.add_argument("--dsm-std", type=float)
    parser.add_argument("--expected-count", type=int)
    return parser


def main():
    parser = parser_for_eval()
    args = parser.parse_args()
    if (args.dsm_mean is None) != (args.dsm_std is None):
        parser.error("Provide both --dsm-mean and --dsm-std, or neither.")
    stats = None if args.dsm_mean is None else (args.dsm_mean, args.dsm_std)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    dataset = DSMPatches(args.data_root, args.split, args.scale, args.patch_size, stats=stats)
    if args.expected_count is not None and len(dataset) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} pairs, found {len(dataset)}.")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    model = RFDMamba()
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True), strict=True)
    metrics, rows = evaluate(model.to(device), loader, device, args.scale)
    result = {
        "scale": args.scale, "split": args.split, "count": len(rows),
        "dsm_normalization": {"mean_m": dataset.mean, "std_m": dataset.std},
        "metrics": metrics,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
