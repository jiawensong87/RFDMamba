# RFDMamba

**Reliability-Aware State-Space Propagation and Frequency-Disentangled Fusion for Optical-Image-Guided DSM Super-Resolution**

Jiawen Song and Qingsong Wang

This repository provides the PyTorch model implementation, training and
evaluation entry points for the full model, and pretrained weights. Source
imagery, prepared dataset patches, and private experiment logs are not included.

## Architecture

- **CAE:** Cross-Modal Aggregation Encoder
- **GRE:** Guidance Reliability Estimation
- **ASSP:** Anchored State-Space Propagation
- **FDF:** Frequency-Disentangled Fusion

In the paper configuration, GRE weights the structural injection before the four-directional selective scan, and the decoder receives the effective reliability maps. The reliability floor is 0.1. The FDF high-frequency branch is not gated.

## Quick start

Install Python 3.10 or later and a suitable [PyTorch](https://pytorch.org/get-started/locally/) build, then run `pip install -r requirements.txt`. For GPU training and evaluation, also install the compatible compiled selective-scan operator from [Mamba](https://github.com/state-spaces/mamba#installation).

```python
import torch
from model import RFDMamba

model = RFDMamba().eval()
optical = torch.randn(1, 3, 16, 16)
dsm_up = torch.randn(1, 1, 16, 16)

with torch.inference_mode():
    prediction = model(optical, dsm_up)
```

This example uses synthetic inputs and randomly initialized weights. The CPU reference scan is intended for small functional checks. For trained inference, use an aligned, normalized optical image and a standardized DSM bicubically upsampled to the optical grid. Inputs have shapes `[B, 3, H, W]` and `[B, 1, H, W]`, with height and width divisible by eight. The output uses the same standardization as the input DSM; apply the inverse transform to recover elevations in meters. No reference DSM is needed by the forward method.

## Data layout

Use prepared, co-registered 256×256 TIFF patch pairs in
`<root>/{train,val,test}/{dsm,rgb}/`. The DSM is single-channel in meters;
optical inputs are stored as HWC TIFFs in the original band order, including
NIR–R–G for Vaihingen. The loader recognizes the benchmark filenames
(`dsm_09cm_matching_*` / `top_mosaic_09cm_*` and
`dom50_patch_*` / `dop50_rgb_patch_*`). For other names, use matching TIFF
stems or a `pairs.csv` in each split with columns `dsm,optical` and one
pair of filenames per row. No automatic split or failed-read substitution is
performed. Use the paper's original partitions for benchmark comparisons;
an arbitrary split is not equivalent.

The loader clips each HR DSM at its 1st and 99th percentiles, area-averages
to the selected LR grid, then bicubically upsamples the result. By default,
DSM normalization uses dataset-wide statistics from the selected split,
matching the reference-derived benchmark protocol; this is not the
normalization used for real-data deployment. To evaluate with supplied
statistics, pass both `--dsm-mean` and `--dsm-std`. The reference loader
uses float64 accumulation to compute split statistics; minor numeric
differences across library versions and hardware are possible.

## Training and evaluation

```bash
python train.py --data-root /path/to/prepared/Vaihingen --scale 16 --out runs/vaihingen-x16
python evaluate.py --data-root /path/to/prepared/Vaihingen --scale 16 \
  --weights weights/RFDMamba-Vaihingen-x16.pth --out runs/eval-vaihingen-x16 \
  --expected-count 285 --legacy-amp
```

The training entry point uses the three-term loss (pixel L1, Sobel gradient,
and geometry-consistency reliability), AdamW and validation RMSE selection.
It retains the spatial augmentation implementation (joint flips and quarter
turns) and does not automatically evaluate the test set during training.
The evaluator defaults to FP32, crops a border of `scale` pixels, and
reports tile-wise RMSE, MAE, SSIM and EdgeRMSE. For the archived Vaihingen
benchmark, use `--legacy-amp` to reproduce the original CUDA autocast
setting; this can affect the last reported digit even though the weights
are identical. Use FP32 for the final Cologne benchmark. These are
streamlined full-model entry points, not the full paper experiment suite.

## Checkpoints

Download the RFDMamba weights from the
[benchmark release](https://github.com/jiawensong87/RFDMamba/releases/tag/v1.0.0).
[checkpoints.json](checkpoints.json) lists the six dataset/scale combinations,
their SHA-256 checksums and the corresponding archived paper metrics.
The released files are tensor-only state dictionaries. Load them with
`torch.load(path, map_location="cpu", weights_only=True)` and
`model.load_state_dict(state, strict=True)`. The Cologne ×16 file is the
checkpoint used in the final paper, not the earlier benchmark checkpoint.

## Acknowledgments

The CUDA backend uses the selective-scan operator from [Mamba](https://github.com/state-spaces/mamba). The four-directional SS2D formulation follows [VMamba](https://github.com/MzeroMiko/VMamba).
