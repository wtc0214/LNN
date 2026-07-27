# LNN: Liquid Neural Network Enhanced YOLO for UAV Small Object Detection

🚀 Overview

LNN is a lightweight and real-time object detection framework, designed for small object detection in unmanned aerial vehicle (UAV) imagery.

Unlike conventional detectors that rely on fixed convolutional feature extraction and struggle with limited object pixels, complex backgrounds, and scale variations, LNN introduces liquid neural modeling to achieve adaptive feature evolution and dynamic representation learning.

The proposed framework aims to achieve an effective balance between detection accuracy, computational efficiency, and real-time inference capability, making it suitable for UAV perception, aerial monitoring, and edge-device deployment.

## Model Architecture

LNN introduces three key modules to improve feature representation and computational efficiency:

- **Liquid Neural Module (Liquid)**
  
  Introduces adaptive hidden-state evolution based on liquid neural dynamics, enabling continuous feature refinement and improving the representation capability for small and low-resolution objects.


- **Liquid Spatial Pyramid Pooling (LiquidSPPF)**
  
  Integrates liquid state evolution with multi-scale spatial aggregation, enhancing contextual information extraction under large-scale variations and complex UAV backgrounds.


- **C2Liquid_Adaptive**
  
  Combines C2-style feature aggregation with adaptive liquid evolution, allowing dynamic feature fusion and improving localization accuracy for densely distributed small targets.


## Datasets
The experiments are conducted on three datasets:

 1. AI-TOD Dataset
    
🔗https://github.com/jwwangchn/AI-TOD

 2. UAVDT Dataset
    
🔗https://zenodo.org/records/14575517

 3. VisDrone2019-DET Dataset
    
🔗https://github.com/VisDrone/VisDrone-Dataset

Place datasets under `datasets/` or edit the YAMLs in `ultralytics/cfg/datasets/` (e.g., `VisDrone.yaml`, `ai_tod.yaml`, `uavdt.yaml`).

## Environment
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


##  Quick Start (Training)
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


#### Explanation of Training Modes

Below are the Python script files for different training modes included in the project, each targeting specific training needs and data types.

4.1. **`train.py`**
   - Basic training script.
   - Used for standard training processes, suitable for general image classification or detection tasks.

4.2. **`train-rtdetr.py`**
   - Training script for RTDETR (Real-Time Detection Transformer).

4.3. **`train_Gray.py`**
   - Grayscale image training script.
   - Specifically for processing datasets of grayscale images, suitable for tasks requiring image analysis in grayscale space.


###  Testing
Run the test script to verify if the data loading is correct:
```bash
python val.py
```
###  inference scripts
```bash
python get_FPS.py
```
