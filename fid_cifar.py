import os
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

# CIFAR-100 train 이미지 저장 함수
def save_cifar100_train_images(output_dir):
    """
    CIFAR-100 train 데이터를 하나의 디렉토리에 저장합니다.

    Args:
        output_dir (str): 이미지를 저장할 디렉토리 경로.
    """
    os.makedirs(output_dir, exist_ok=True)  # 저장 디렉토리 생성

    # CIFAR-100 train 데이터 로드
    transform = transforms.ToTensor()  # Tensor 형식으로 로드
    dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    print(f"Saving {len(dataset)} images to {output_dir}...")

    for idx, (image, label) in enumerate(tqdm(dataset, desc="Saving images")):
        # PIL 이미지로 변환
        image = transforms.ToPILImage()(image)
        # 파일명: train_<index>.png
        image_path = os.path.join(output_dir, f"train_{idx:05d}.png")
        # 이미지 저장
        image.save(image_path)

    print(f"All images saved to {output_dir}.")

# 저장 경로
output_dir = './cifar100_train_images'

# 실행
save_cifar100_train_images(output_dir)