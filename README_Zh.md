# LNN：基于液态神经网络增强的无人机小目标检测框架

## 项目简介
LNN（Liquid Neural Network Enhanced YOLO）是一种面向无人机（UAV）图像小目标检测的轻量化实时目标检测框架。不同于传统检测器依赖固定卷积特征提取方式、难以有效处理无人机视角下目标像素稀少、背景复杂以及尺度变化剧烈等问题，LNN 引入液态神经网络（Liquid Neural Network）动态特征建模机制，实现特征状态的自适应演化与动态表示学习。该框架旨在实现检测精度、计算效率和实时推理速度之间的有效平衡，可广泛应用于无人机视觉感知、航空监测以及边缘设备部署等实时检测场景。

## 模型结构
LNN 引入三个关键模块，以提升无人机小目标检测中的特征表达能力和计算效率：
-Liquid Neural Module（Liquid）
引入基于液态神经动力学的自适应隐藏状态演化机制，使网络能够根据输入特征动态调整状态更新过程。该模块能够持续优化特征表示，有效增强低分辨率、小尺寸目标的结构信息提取能力。
-Liquid Spatial Pyramid Pooling（LiquidSPPF）
将液态状态演化机制与多尺度空间池化结构相结合，实现更加灵活的上下文信息聚合。该模块能够增强不同尺度目标的特征表达能力，提高模型在复杂无人机背景和尺度变化环境中的检测鲁棒性。
-C2Liquid_Adaptive
结合 C2 特征融合结构与自适应液态神经演化机制，实现动态特征融合。该模块能够根据输入内容调整不同尺度特征的贡献，提高密集分布小目标的定位精度和检测稳定性。

数据集

本项目在以下三个公开无人机小目标数据集上进行实验验证：

1. AI-TOD Dataset

🔗 https://github.com/jwwangchn/AI-TOD

2. UAVDT Dataset

🔗 https://sites.google.com/site/daviddo0323/projects/uavdt

3. VisDrone2019-DET Dataset

🔗 https://github.com/VisDrone/VisDrone-Dataset

### 3. Install Dependencies
(环境安装推荐直接使用已配置好的 YOLOv8 或 YOLOv11 环境，无需重复安装）
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


### 4. 运行训练
```bash
python train.py --data your_dataset_config.yaml
```
#### 训练脚本说明

本项目包含多个训练脚本，适用于不同任务：

4.1. **`train.py`**
  - 基础训练脚本，适用于通用目标检测任务


4.2. **`train-rtdetr.py`**
   - 用于 RT-DETR 模型的训练

4.3. **`train_Gray.py`**
   - 灰度图训练脚本，适用于单通道图像任务


### 5.测试与验证

运行以下命令进行模型验证：
```bash
python val.py
```
