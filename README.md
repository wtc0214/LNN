# Liquid Neural Networks (LNN) Modules for small-object datasets

English README for GitHub, covering setup, datasets, modules, and usage.

## Overview
- Implements Liquid Neural Network (LNN) blocks and integrates them with YOLO backbones/heads (C2f, SPPF) to form C2Liquid and LiquidSPPF variants.
- Supports plug-and-play replacement of standard C2f/SPPF without changing bottleneck structure.

## Environment Setup
```bash
# 1) Create env (Python >=3.10)
conda create -n yolov8 python=3.10 -y
conda activate yolov8

# 2) Install PyTorch (choose CUDA version for your GPU, e.g. cu121)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 3) Install project deps
pip install -r requirements.txt
```

## Datasets (links)
- AI-TOD: https://github.com/jwwangchn/AI-TOD
- UAVDT: https://sites.google.com/site/daviddo0323/projects/uavdt
- VisDrone2019-DET: https://github.com/VisDrone/VisDrone-Dataset

Place datasets under `datasets/` or adjust paths in the YAML files under `ultralytics/cfg/datasets/`.

## Quick Start (Training)
```bash
# Standard LNN model
python train.py --model yolov11_lnn.yaml --data ultralytics/cfg/datasets/VisDrone.yaml --epochs 300

# Lite version
python train.py --model yolov11_lnn_lite.yaml --data ultralytics/cfg/datasets/VisDrone.yaml --epochs 300

# Adaptive version
python train.py --model yolov11_lnn_adaptive.yaml --data ultralytics/cfg/datasets/VisDrone.yaml --epochs 300

# Select model scale (n/s/m/l/x)
python train.py --model yolov11_lnn.yaml --scale s --data ultralytics/cfg/datasets/VisDrone.yaml --epochs 300


## Using in Python
```python
from ultralytics.nn.modules import C2Liquid, LiquidSPPF, C2Liquid_Lite
import torch

c2liquid = C2Liquid(
    c1=256, c2=256, n=3, shortcut=False, g=1, e=0.5,
    use_lnn=True, lnn_hidden_dim=None,
    use_lightweight=True, use_adaptive_fusion=True
)

x = torch.randn(1, 256, 64, 64)
out = c2liquid(x)

# Optional temporal modeling for videos
c2liquid.use_temporal = True
```

## Module List
- Core: `LiquidNeuralUnit`, `LiquidNeuralModule`, `LiquidNeuralModuleLite`
- C2Liquid family: `C2Liquid`, `C2Liquid_Lite`, `C2Liquid_Adaptive`
- LiquidSPPF family: `LiquidSPPF`, `LiquidSPPF_Lite`

## Why LNN
- **Dynamic feature enhancement** via ODE-based hidden state updates.
- **Adaptive fusion** between original and enhanced features.
- **Temporal modeling (optional)** for video detection.
- **Lightweight options**: depthwise separable conv + reduced hidden dims.
- **Plug-and-play**: keep C2f bottleneck unchanged; enhance at output.

## Tuning Tips
- For higher mAP: enable adaptive fusion (`use_adaptive_fusion=True`), tune hidden dim (`c2*0.5` as default).
- For lower FLOPs/params: use Lite variants, reduce `lnn_hidden_dim_ratio`, disable adaptive fusion.

## Notes
- Start with standard LNN, then switch to Lite/Adaptive after baseline is stable.
- If using temporal modeling, call `reset_hidden_state()` at the start of a sequence.
- Keep input/output channels aligned for best performance (module can auto-handle minor mismatches).

## Citation
If you use the LNN modules, please cite:
- Hasani, R., et al. "Liquid Time-Constant Networks." AAAI, 2021.

