import os
import subprocess
import os
import shutil
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToPILImage

# Super-class to sub-class mapping
superclass_mapping = {
    0: [4, 30, 55, 72, 95],      # aquatic mammals
    1: [1, 32, 67, 73, 91],      # fish
    2: [54, 62, 70, 82, 92],     # flowers
    3: [9, 10, 16, 28, 61],      # food containers
    4: [0, 51, 53, 57, 83],      # fruit and vegetables
    5: [22, 39, 40, 86, 87],     # household electrical devices
    6: [5, 20, 25, 84, 94],      # household furniture
    7: [6, 7, 14, 18, 24],       # insects
    8: [3, 42, 43, 88, 97],      # large carnivores
    9: [12, 17, 37, 68, 76],     # large man-made outdoor things
    10: [23, 33, 49, 60, 71],    # large natural outdoor scenes
    11: [15, 19, 21, 31, 38],    # large omnivores and herbivores
    12: [34, 63, 64, 66, 75],    # medium-sized mammals
    13: [26, 45, 77, 79, 99],    # non-insect invertebrates
    14: [2, 11, 35, 46, 98],     # people
    15: [27, 29, 44, 78, 93],    # reptiles
    16: [36, 50, 65, 74, 80],    # small mammals
    17: [47, 52, 56, 59, 96],    # trees
    18: [8, 13, 48, 58, 90],     # vehicles 1
    19: [41, 69, 81, 85, 89]     # vehicles 2
}

def prepare_real_images(output_dir, num_classes=100, superclass_mapping=None):
    """
    Prepare real images from the CIFAR-100 dataset organized by superclass.

    :param output_dir: Directory to save the organized images.
    :param num_classes: Total number of classes in CIFAR-100 (default: 100).
    :param superclass_mapping: Mapping of super-classes to sub-classes.
    """
    # Load CIFAR-100 training dataset
    cifar100_train = CIFAR100(root="./data", train=True, download=True)

    # Initialize output directory
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over super-class mapping
    for superclass, subclasses in superclass_mapping.items():
        superclass_dir = os.path.join(output_dir, f"super_{superclass}")
        os.makedirs(superclass_dir, exist_ok=True)

        # Collect all images that belong to the current super-class
        for subclass in subclasses:
            for idx, (img, label) in enumerate(zip(cifar100_train.data, cifar100_train.targets)):
                if label == subclass:
                    img = ToPILImage()(img)  # Convert numpy array to PIL image
                    img.save(os.path.join(superclass_dir, f"real_{idx}_class{subclass}.png"))

    print(f"Organized real images into {output_dir}")


# 설정 경로
real_images_output_dir = "./real-images/superclass"

# 실제 이미지 준비 실행
prepare_real_images(real_images_output_dir, superclass_mapping=superclass_mapping)


def generate_images_flat(network_pkl, outdir, seeds_per_class, batch, steps):
    """
    Generate 2500 images for each class (0-19) and organize them under class directories.
    """
    num_images_per_class = 2500
    num_classes = 20
    images_per_seed = 1  # Assume 1 image is generated per seed

    for class_idx in range(num_classes):
        # Create a directory for the class
        class_dir = os.path.join(outdir, f'class_{class_idx}')
        os.makedirs(class_dir, exist_ok=True)

        # Calculate the number of seeds needed
        total_seeds = (num_images_per_class + images_per_seed - 1) // images_per_seed

        print(f"Generating {num_images_per_class} images for class {class_idx}...")

        # Define the command for image generation
        command = [
            "python", "generate.py",
            "--network", network_pkl,
            "--outdir", class_dir,
            "--seeds", f"0-{total_seeds - 1}",
            "--batch", str(batch),
            "--steps", str(steps),
            "--class", str(class_idx)
        ]

        try:
            # Run the command
            subprocess.run(command, check=True)

        except subprocess.CalledProcessError as e:
            print(f"Error while generating images for class {class_idx}: {e}")
        except Exception as e:
            print(f"Unexpected error for class {class_idx}: {e}")

    print("All images have been generated and organized.")

# Configuration for the generation
network_pkl = "/content/V1V1_Coditional_Image_Generation_CIFAR100/training-runs/00000-cifar100_dataset-cond-ddpmpp-vp-gpus1-batch64-fp32/network-snapshot-000000.pkl"
outdir = "./intra_generated-images"
seeds_per_class = 2500  # Number of images per class
batch = 64
steps = 18

# Run the generation
generate_images_flat(network_pkl, outdir, seeds_per_class, batch, steps)
