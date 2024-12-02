# 2024-2 시각지능학습 Image Generation Challenge
## V1V1_Coditional_Image_Generation_CIFAR100
## Version

Python 3.10

torchvison 0.19.1+cu118  
torch 2.4.1+cu118  

## Requirements

```bash
git clone https://github.com/qkrwnsdn0427/V1V1_Coditional_Image_Generation_CIFAR100.git
```
## How to run
After you have cloned the repository, you can train cifar100 and change seed value by running the script below 

Prepare cifar100 dataset
```bash
!python3 cifar100_download.py
```

```bash
!python3 dataset_tool.py --source /V1V1_Coditional_Image_Generation_CIFAR100/cifar-100-python.tar.gz --dest /V1V1_Coditional_Image_Generation_CIFAR100/cifar100_dataset --resolution=32x32
```

```bash
!python3 train.py \
    --outdir=/V1V1_Coditional_Image_Generation_CIFAR100/training-runs \
    --data=/V1V1_Coditional_Image_Generation_CIFAR100/cifar100_dataset \
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
    --network=V1V1_Coditional_Image_Generation_CIFAR100/training-runs/00002-cifar100_dataset-cond-ddpmpp-vp-gpus1-batch64-fp32/network-snapshot-027526.pkl \
    --outdir=/V1V1_Coditional_Image_Generation_CIFAR100/generated-images \
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
|  DDPM++         |   0.13  |  552     | 4.5871075833975965 | 29.77 ± 4.46 |            |
