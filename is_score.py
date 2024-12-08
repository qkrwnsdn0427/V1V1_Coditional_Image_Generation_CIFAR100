import numpy as np
from scipy import linalg # For numpy FID
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P
from torchvision.models.inception import inception_v3
from scipy.linalg import sqrtm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
# returning pool features and logits.
class WrapInception(nn.Module):
  def __init__(self, net):
    super(WrapInception,self).__init__()
    self.net = net
    self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                  requires_grad=False)
    self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                 requires_grad=False)
  def forward(self, x):
    # Normalize x
    x = (x + 1.) / 2.0
    x = (x - self.mean) / self.std
    # Upsample if necessary
    if x.shape[2] != 299 or x.shape[3] != 299:
      x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
    # 299 x 299 x 3
    x = self.net.Conv2d_1a_3x3(x)
    # 149 x 149 x 32
    x = self.net.Conv2d_2a_3x3(x)
    # 147 x 147 x 32
    x = self.net.Conv2d_2b_3x3(x)
    # 147 x 147 x 64
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 73 x 73 x 64
    x = self.net.Conv2d_3b_1x1(x)
    # 73 x 73 x 80
    x = self.net.Conv2d_4a_3x3(x)
    # 71 x 71 x 192
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 35 x 35 x 192
    x = self.net.Mixed_5b(x)
    # 35 x 35 x 256
    x = self.net.Mixed_5c(x)
    # 35 x 35 x 288
    x = self.net.Mixed_5d(x)
    # 35 x 35 x 288
    x = self.net.Mixed_6a(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6b(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6c(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6d(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6e(x)
    # 17 x 17 x 768
    # 17 x 17 x 768
    x = self.net.Mixed_7a(x)
    # 8 x 8 x 1280
    x = self.net.Mixed_7b(x)
    # 8 x 8 x 2048
    x = self.net.Mixed_7c(x)
    # 8 x 8 x 2048
    pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
    # 1 x 1 x 2048
    logits = self.net.fc(F.dropout(pool, training=False).view(pool.size(0), -1))
    # 1000 (num_classes)
    return pool, logits


def calculate_inception_score(pred, num_splits=10):
  scores = []
  for index in range(num_splits):
    pred_chunk = pred[index * (pred.shape[0] // num_splits): (index + 1) * (pred.shape[0] // num_splits), :]
    kl_inception = pred_chunk * (np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
    kl_inception = np.mean(np.sum(kl_inception, 1))
    scores.append(np.exp(kl_inception))
  return np.mean(scores), np.std(scores)


# Loop and run the sampler and the net until it accumulates num_inception_images
# activations. Return the pool, the logits, and the labels (if one wants 
# Inception Accuracy the labels of the generated class will be needed)
def accumulate_inception_activations(dataloader, model, num_inception_images=50000):
    logits_list = []
    with torch.no_grad():
        for images in dataloader:
            images = images.cuda()  # Ensure images are moved to GPU
            _, logits = model(images)  # Get logits
            logits = F.softmax(logits, dim=1).cpu().numpy()  # Apply softmax and move to CPU
            logits_list.append(logits)
    logits = np.concatenate(logits_list, axis=0)  # Concatenate all logits
    return logits

def load_inception_net(parallel=False):
  inception_model = inception_v3(weights="DEFAULT", transform_input=False)   #inception_v3(pretrained=True, transform_input=False)
  inception_model = WrapInception(inception_model.eval()).cuda()
  if parallel:
    print('Parallelizing Inception module...')
    inception_model = nn.DataParallel(inception_model)
  return inception_model


class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = [
            os.path.join(image_folder, f) for f in os.listdir(image_folder)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
if __name__ == "__main__":
    # Replace with your image folder path
    image_folder = "./generated-images"
    
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Adjust to your image resolution
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    # Load dataset and create DataLoader
    dataset = ImageDataset(image_folder=image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load Inception model
    net = load_inception_net()
    
    print('Gathering activations...')
    logits = accumulate_inception_activations(dataloader, net, num_inception_images=50000)
    
    print('Calculating Inception Score...')
    IS_mean, IS_std = calculate_inception_score(logits, num_splits=10)
    print(f"Inception Score: Mean={IS_mean:.4f}, Std={IS_std:.4f}")
