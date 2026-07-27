# LNN: Liquid Neural Network Enhanced YOLO for UAV Small Object Detection

README for GitHub. This repository provides the official implementation of LNN (Liquid Neural Network Enhanced Detector), including environment setup, dataset preparation, model configurations, training commands, and module usage for UAV small object detection on AI-TOD, UAVDT, and VisDrone datasets.

## 1) Environment
```bash
# Python >= 3.10
conda create -n yolov8 python=3.10 -y
conda activate yolov8

# PyTorch (pick CUDA version for your GPU; example: cu121)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# Project deps
pip install -r requirements.txt
```

## 2) Datasets
- AI-TOD: https://github.com/jwwangchn/AI-TOD
- UAVDT: https://sites.google.com/site/daviddo0323/projects/uavdt
- VisDrone2019-DET: https://github.com/VisDrone/VisDrone-Dataset

Place datasets under `datasets/` or edit the YAMLs in `ultralytics/cfg/datasets/` (e.g., `VisDrone.yaml`, `ai_tod.yaml`, `uavdt.yaml`).

## 3) Model Configurations
The repository provides several LNN configurations:
ultralytics/cfg/models/
│
├── yolov8_lnn.yaml
├── yolov8_lnn_ai_tod.yaml
├── yolov8_lnn_uavdt.yaml
└── yolov8_lnn_visdrone.yaml

Different configurations are optimized for different UAV scenarios.
The proposed framework mainly introduces three components:
#Liquid Module
-Adaptive hidden-state evolution
-Dynamic feature transformation
-Enhances spatial representation for small objects
#LiquidSPPF
-Liquid-enhanced spatial pyramid pooling
-Improves multi-scale feature aggregation
-Strengthens object representation under scale variation
#C2Liquid_Adaptive
-Adaptive liquid feature fusion
-Dynamically adjusts feature interactions
-Improves localization accuracy for tiny objects

## 4) Quick Start (Training)
```bash
# Generic INR model
python train.py --model yolov8_inr_enhanced.yaml --data ultralytics/cfg/datasets/VisDrone.yaml --epochs 300

# VisDrone tuned
python train.py --model yolov8_inr_visdrone.yaml --data ultralytics/cfg/datasets/VisDrone.yaml --epochs 300

# AI-TOD tuned
python train.py --model yolov8_inr_ai_tod.yaml --data ultralytics/cfg/datasets/ai_tod.yaml --epochs 300

# UAVDT tuned
python train.py --model yolov8_inr_uvadt.yaml --data ultralytics/cfg/datasets/uavdt.yaml --epochs 300

# Multi-dataset helper script
python train_inr_datasets.py --dataset visdrone   # or ai_tod / uvadt / all
```

## 5) Using in Python
```python
import torch
from ultralytics.nn.modules.inr_c2f import INREnhancedC2f

module = INREnhancedC2f(
    c1=256, c2=256, n=3,
    shortcut=False, g=1, e=0.5,
    use_inr=True,       # enable INR enhancement
    use_attention=True, # coordinate-aware attention
    coord_encoding_dim=128
)
x = torch.randn(1, 256, 64, 64)
y = module(x)
```

## 6) What is INREnhancedC2f
- Extends C2f with Implicit Neural Representation (INR) to encode fine-grained spatial details for tiny/clustered objects.
- Adds coordinate encoding + attention for better localization under high-altitude UAV views.
- Drop-in replacement for C2f; preserves interfaces so YAML swaps are simple.

## 7) Key Params to Tune
- `use_inr` (bool): turn INR enhancement on/off.
- `use_attention` (bool): keep on for UAV tiny objects.
- `coord_encoding_dim` (int): 96–192 works well on VisDrone; 128–224 on AI-TOD.
- `e` (expand ratio) and `n` (repeats): balance capacity vs. FLOPs for your GPU.

## 8) Tips for Small-Object UAV Training
- Use higher input resolution if memory allows (e.g., 832–1024 for AI-TOD).
- Keep strong augmentations (mosaic/mixup) modest to avoid over-corrupting tiny targets.
- Validate with class-wise recall; tiny-object recall is the main indicator.

## 9) References
- INR and coordinate encodings inspired by implicit neural representations for vision.
- Please also cite YOLOv8 and your chosen datasets (AI-TOD, UAVDT, VisDrone) when publishing results.

