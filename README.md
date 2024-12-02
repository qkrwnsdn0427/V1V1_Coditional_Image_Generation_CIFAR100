# 2024-2 시각지능학습 Image Generation Challenge
## V1V1_Coditional_Image_Generation_CIFAR100
## Version

Python 3.10

torchvison 0.19.1+cu118  
torch 2.4.1+cu118  

## Requirements
```bash
pip install torch==2.4.1 torchvision==0.19.1 numpy==1.21.2 matplotlib==3.4.3 scikit-learn==1.0.2
git clone https://github.com/qkrwnsdn0427/V1V1_Coditional_Image_Generation_CIFAR100.git
```
## How to run
After you have cloned the repository, you can train cifar100 and change seed value by running the script below 
```bash
!python3 train.py \
    --outdir=/home/v1v1/edm/training-runs \
    --data=/home/v1v1/edm/cifar100_dataset \
    --cond=True \
    --arch=ddpmpp \
    --precond=vp \
    --batch=64 \
    --lr=1e-4 \
    --duration=27.5\
    --augment=0.15
```
```bash
!python generate.py \
    --network=/content/network-snapshot-002500.pkl \
    --outdir=/content/edm/generated-images \
    --seeds=0-4999 \
    --batch=64 \
    --steps=18
```
## Implementation Details


## CUDA and GPU Information
CUDA Version: 11.8

GPU: NVIDIA RTX 3090

## Cifar-100 Results

| Network         | Dropout |    Per Epoch  |         FID        | Intra-FID | Inception Score |
|-----------------|---------|---------------|--------------------|-------------|-------------|
|  DDPM++         |   0.13  |  28000000     | 4.5871075833975965 | 29.77 ± 4.46 |            |
