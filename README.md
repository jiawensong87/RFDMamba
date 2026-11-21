# RFDMamba

**Reliability-Aware State-Space Propagation and Frequency-Disentangled Fusion for Optical-Image-Guided DSM Super-Resolution**

Jiawen Song and Qingsong Wang

This repository provides the PyTorch model implementation of RFDMamba. It does not include training or evaluation scripts, datasets, or pretrained weights.

## Architecture

- **CAE:** Cross-Modal Aggregation Encoder
- **GRE:** Guidance Reliability Estimation
- **ASSP:** Anchored State-Space Propagation
- **FDF:** Frequency-Disentangled Fusion

In the paper configuration, GRE weights the structural injection before the four-directional selective scan, and the decoder receives the effective reliability maps. The reliability floor is 0.1. The FDF high-frequency branch is not gated.

## Quick start

Install Python 3.10 or later, [PyTorch](https://pytorch.org/get-started/locally/), and `einops`. To run on CUDA, also install the compatible compiled selective-scan operator from [Mamba](https://github.com/state-spaces/mamba#installation).

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

## Acknowledgments

The CUDA backend uses the selective-scan operator from [Mamba](https://github.com/state-spaces/mamba). The four-directional SS2D formulation follows [VMamba](https://github.com/MzeroMiko/VMamba).
