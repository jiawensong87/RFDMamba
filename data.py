"""Strict loader for pre-split optical/DSM TIFF patches."""

import csv
import random
import re
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


def tile_key(stem):
    """Match the benchmark DSM/optical naming conventions, or identical stems."""
    match = re.fullmatch(
        r"(?:dsm_09cm_matching_|top_mosaic_09cm_)(area\d+_\d+)", stem
    )
    if match:
        return "vaihingen:" + match[1]
    match = re.fullmatch(r"(?:dom50_patch_|dop50_rgb_patch_)(\d+)", stem)
    if match:
        return "cologne:" + match[1]
    return stem


def load_pairs(folder):
    manifest = folder / "pairs.csv"
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        pairs = []
        for row in rows:
            a, b = row["dsm"], row["optical"]
            if any(not x or Path(x).name != x or x in {".", ".."} for x in (a, b)):
                raise ValueError("pairs.csv must contain filenames only.")
            pairs.append((folder / "dsm" / a, folder / "rgb" / b))
    else:
        def index(directory):
            result = {}
            for p in sorted(directory.iterdir()):
                if p.suffix.lower() in {".tif", ".tiff"}:
                    key = tile_key(p.stem)
                    if key in result:
                        raise ValueError(f"Duplicate TIFF stem: {p.stem}")
                    result[key] = p
            return result

        dsm, optical = index(folder / "dsm"), index(folder / "rgb")
        if dsm.keys() != optical.keys():
            raise ValueError("Unmatched TIFF names. Supply an explicit pairs.csv.")
        pairs = [(dsm[name], optical[name]) for name in sorted(dsm)]
    if (
        not pairs
        or len({a for a, _ in pairs}) != len(pairs)
        or len({b for _, b in pairs}) != len(pairs)
    ):
        raise ValueError(f"Empty or duplicate pair list in {folder}.")
    return pairs


def clipped_dsm(path, patch_size):
    dsm = tifffile.imread(path)
    if dsm.shape != (patch_size, patch_size) or not np.isfinite(dsm).all():
        raise ValueError(f"Invalid single-channel DSM patch: {path}")
    low, high = np.percentile(dsm, (1, 99))
    return np.clip(dsm, low, high).astype(np.float32)


def split_statistics(pairs, patch_size):
    total, squared, count = 0.0, 0.0, 0
    for path, _ in pairs:
        tile = clipped_dsm(path, patch_size).astype(np.float64)
        total += tile.sum()
        squared += np.square(tile).sum()
        count += tile.size
    mean = float(total / count)
    std = float(np.sqrt(max(squared / count - mean**2, 0)))
    return mean, std if std >= 1e-6 else 1.0


class DSMPatches(Dataset):
    def __init__(self, root, split, scale, patch_size=256, *, augment=False, stats=None):
        if patch_size < 8 or patch_size % 8 or patch_size % scale:
            raise ValueError("Patch size must be divisible by eight and by the scale.")
        self.pairs = load_pairs(Path(root) / split)
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment
        self.mean, self.std = (
            split_statistics(self.pairs, patch_size) if stats is None else stats
        )
        if not np.isfinite([self.mean, self.std]).all() or self.std <= 0:
            raise ValueError("DSM statistics must be finite, with standard deviation > 0.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        dsm_path, optical_path = self.pairs[idx]
        h = self.patch_size
        gt = clipped_dsm(dsm_path, h)
        image = tifffile.imread(optical_path)
        if image.shape != (h, h, 3) or not np.isfinite(image).all():
            raise ValueError(f"Expected an aligned HWC optical patch: {optical_path}")
        # Calculate moments before spatial augmentation, in the stored band order.
        mean = image.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        std = (image.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)
        image = ((image.astype(np.float32) - mean) / (std + 1e-8))
        if self.augment:
            for axis in (0, 1):
                if random.random() > 0.5:
                    gt, image = np.flip(gt, axis), np.flip(image, axis)
            if random.random() > 0.5:
                turns = random.randint(1, 3)
                gt, image = np.rot90(gt, turns), np.rot90(image, turns)
        lr = cv2.resize(np.ascontiguousarray(gt), (h // self.scale,) * 2, interpolation=cv2.INTER_AREA)
        up = cv2.resize(lr, (h, h), interpolation=cv2.INTER_CUBIC)
        scale = self.std + 1e-8
        return (
            torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            torch.from_numpy(np.ascontiguousarray(((up - self.mean) / scale)[None])),
            torch.from_numpy(np.ascontiguousarray(((gt - self.mean) / scale)[None])),
            dsm_path.name,
        )
