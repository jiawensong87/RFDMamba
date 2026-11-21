"""Train the paper's full RFDMamba model on prepared train/val TIFF pairs."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import DSMPatches
from evaluate import evaluate
from losses import HybridStructureLoss
from model import RFDMamba


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def train_epoch(model, loader, loss, optimizer, scaler, device, amp, warmup):
    model.train()
    count, total = 0, 0.0
    for optical, dsm_up, gt, _ in loader:
        optical, dsm_up, gt = (x.to(device) for x in (optical, dsm_up, gt))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            pred, diagnostics = model(optical, dsm_up, return_info=True)
        # The auxiliary losses, including BCE, are evaluated in FP32.
        diagnostics = [{
            "ssm_info": {"reliability_map": stage["ssm_info"]["reliability_map"].float()}
        } for stage in diagnostics]
        value = loss(pred.float(), gt, diagnostics, optical)["total"]
        if not torch.isfinite(value):
            raise RuntimeError("Nonfinite training loss.")
        scaler.scale(value).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if warmup is not None and scaler.get_scale() >= before:
            warmup.step()
        count += len(optical)
        total += value.item() * len(optical)
    return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.val_every, args.patience) < 1 or args.warmup_epochs < 0:
        parser.error("Epoch counts, batch size and patience must be positive; warmup can be zero.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
        if args.patch_size > 32:
            parser.error("The CPU reference scan is only for small functional checks.")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train = DSMPatches(args.data_root, "train", args.scale, args.patch_size, augment=True)
    val = DSMPatches(args.data_root, "val", args.scale, args.patch_size)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, num_workers=args.workers, shuffle=True,
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val, batch_size=1, num_workers=args.workers)
    model, loss = RFDMamba().to(device), HybridStructureLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=args.lr * 0.01,
    )
    warmup = None
    if args.warmup_epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=args.warmup_epochs * len(train_loader),
        )
    amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    args.out.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config["dsm_statistics"] = {"train": [train.mean, train.std], "val": [val.mean, val.std]}
    (args.out / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    best, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, loss, optimizer, scaler, device, amp,
            warmup if epoch <= args.warmup_epochs else None,
        )
        record = {"epoch": epoch, "train_loss": train_loss}
        if epoch > args.warmup_epochs:
            cosine.step()
        if epoch % args.val_every == 0 or epoch == args.epochs:
            stats, _ = evaluate(model, val_loader, device, args.scale)
            record["validation"] = stats
            if stats["rmse_m"] < best:
                best, stale = stats["rmse_m"], 0
                torch.save(model.state_dict(), args.out / "best_model.pth")
            else:
                stale += 1
        with (args.out / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if stale >= args.patience:
            break


if __name__ == "__main__":
    main()
