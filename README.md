# Liquid Neural Networks (LNN) Documentation

## Overview

This project implements modules based on MIT CSAIL’s Liquid Neural Network (LNN) framework and integrates them into YOLO architectures by enhancing the C2f and SPPF modules. The resulting modules—such as C2Liquid and LiquidSPPF—provide dynamic neural state modeling and improved feature representation.

## Core Design Concept

Liquid Neural Networks model neuron dynamics using differentiable ordinary differential equations (ODEs):

```
h_t = h_{t-1} + dt * f_θ(x_t, h_{t-1})
```

where:
- `h_t` hidden state at the current time step
- `x_t` input feature at the current step
- `f_θ` learnable ODE function of the liquid neuron
- `dt` a learnable time-step parameter

## Module List

### 1. Core LNN Modules

#### `LiquidNeuralUnit`
The basic liquid neural unit implementing ODE-based state updates.

#### `LiquidNeuralModule`
Full LNN module including adaptive fusion between original and enhanced features.

#### `LiquidNeuralModuleLite`
A lightweight variant designed to further reduce parameters and FLOPs.

### 2. C2Liquid Series

#### `C2Liquid`
- **Purpose**：Fusion of LNN with the original C2f module
- **Architecture**：Attaches LNN enhancement at the end of C2f; the bottleneck structure remains unchanged
- **Characteristics**：
  - Retains the original C2f bottlenecks
  - Applies LNN enhancement only at the output stage
  - Supports adaptive feature fusion
  - Optional temporal modeling for video tasks

#### 训练命令

```bash
# Standard version
python train.py --model yolov11_lnn.yaml --data your_dataset.yaml --epochs 300

# Lightweight version
python train.py --model yolov11_lnn_lite.yaml --data your_dataset.yaml --epochs 300

# Adaptive version
python train.py --model yolov11_lnn_adaptive.yaml --data your_dataset.yaml --epochs 300

# Specify model scale (n/s/m/l/x)
python train.py --model yolov11_lnn.yaml --scale s --data your_dataset.yaml --epochs 300
```


## Notes

1. **first time**：Start with the standard version before switching to Lite or Adaptive variants
2. **temporal modeling（**：When using temporal modeling (use_temporal=True), call reset_hidden_state() at the start of each video sequence
3. **channel alignment**：The module auto-handles channel mismatch, but matching input/output channels is recommended for optimal performance

   
## Experimental Recommendations
1. **Baseline comparison**：Train with standard C2f, then replace with C2Liquid
2. **Ablation studies**：
   - With vs. without LNN
   - Different hidden dimensions
   - Adaptive vs. fixed fusion
3. **Evaluate trade-offs**：Jointly consider mAP, parameters, and FLOPs

## Citation

If you use these modules, please cite the foundational MIT CSAIL LNN paper:
- Hasani, R., et al. "Liquid Time-Constant Networks." AAAI, 2021.

