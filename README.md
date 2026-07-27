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
 1. AI-TOD Dataset
    
🔗https://github.com/jwwangchn/AI-TOD

 2. UAVDT Dataset
    
🔗https://sites.google.com/site/daviddo0323/projects/uavdt

 3. VisDrone2019-DET Dataset
    
🔗https://github.com/VisDrone/VisDrone-Dataset

Place datasets under `datasets/` or edit the YAMLs in `ultralytics/cfg/datasets/` (e.g., `VisDrone.yaml`, `ai_tod.yaml`, `uavdt.yaml`).

## 3) Model Configurations
The repository provides several LNN configurations:
```bash
ultralytics/cfg/models/
│
├── yolov8_lnn.yaml
├── yolov8_lnn_ai_tod.yaml
├── yolov8_lnn_uavdt.yaml
└── yolov8_lnn_visdrone.yaml
```
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
# General LNN Training
python train.py --model yolov8_lnn.yaml --data ultralytics/cfg/datasets/uavdt.yaml --epochs 300

# VisDrone tuned
python train.py --model yolov8_lnn_visdrone.yaml --data ultralytics/cfg/datasets/VisDrone.yaml --epochs 300

# AI-TOD tuned
python train.py --model yolov8_lnn_ai_tod.yaml --data ultralytics/cfg/datasets/ai_tod.yaml --epochs 300

# UAVDT tuned
python train.py --model yolov8_lnn_uavdt.yaml --data ultralytics/cfg/datasets/uavdt.yaml --epochs 300

# Multi-dataset helper script
python train_lnn.py --dataset visdrone   # or ai_tod / uvadt / all
```

## 5) Using in Python
```python
import torch

from ultralytics.nn.modules.liquid import C2Liquid_Adaptive


module = C2Liquid_Adaptive(
    c1=256,
    c2=256,
    n=3,
    shortcut=False,
    expansion=0.5,
    hidden_dim_ratio=0.75,
    tau=1.0
)


x = torch.randn(1,256,64,64)

y = module(x)

print(y.shape)
```

### 3. Install Dependencies
(It is recommended to directly use the YOLOv11 or YOLOv8 environment that has already been set up on this computer, without the need to download again.)
```bash
# Step 1.Create a virtual environment with conda
conda create -n pt121_py38 python=3.8
conda activate pt121_py38

# Step 2: Install pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch


# Step 3: Install the remaining dependencies

pip install -r requirements.txt


# https://pytorch.org/get-started/previous-versions/
## CUDA 10.2
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
## CUDA 11.3
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
## CUDA 11.6
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
## CPU Only
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch

## CUDA 11.8
#conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
## CUDA 12.1
#conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
## CPU Only
#conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cpuonly -c pytorch
```


### 4. Run the Program
```bash
python train.py --data your_dataset_config.yaml
```
#### Explanation of Training Modes

Below are the Python script files for different training modes included in the project, each targeting specific training needs and data types.

4.1. **`train.py`**
   - Basic training script.
   - Used for standard training processes, suitable for general image classification or detection tasks.

2. **`train-rtdetr.py`**
   - Training script for RTDETR (Real-Time Detection Transformer).

3. **`train_Gray.py`**
   - Grayscale image training script.
   - Specifically for processing datasets of grayscale images, suitable for tasks requiring image analysis in grayscale space.


### 5. Testing
Run the test script to verify if the data loading is correct:
```bash
python val.py
```
### 6. inference scripts
```bash
python get_FPS.py
```
