from torchvision.datasets import CIFAR100
import os
from PIL import Image

# 저장 경로 설정
dataset_root = "./datasets/cifar100"
output_dir = "./datasets/cifar100_imagefolder"

# CIFAR-100 데이터셋 다운로드
train_dataset = CIFAR100(root=dataset_root, train=True, download=True)
