import os
import subprocess
import numpy as np

def calculate_intra_fid(real_dir, gen_dir, num_superclasses=20, batch_size=16):
    """
    Calculate Intra-FID with mean and standard deviation for each super-class.

    :param real_dir: Path to the real images directory (organized by super-classes).
    :param gen_dir: Path to the generated images directory (organized by super-classes).
    :param num_superclasses: Number of super-classes (default: 20 for CIFAR-100).
    :param batch_size: Batch size for PyTorch FID computation.
    :return: Tuple (mean Intra-FID, std Intra-FID).
    """
    fid_values = []
    for i in range(num_superclasses):
        real_super_dir = os.path.join(real_dir, f"super_{i}")
        gen_super_dir = os.path.join(gen_dir, f"class_{i}")

        # Check if directories exist
        if not os.path.exists(real_super_dir) or not os.path.exists(gen_super_dir):
            print(f"Skipping super-class {i}: Directory does not exist.")
            continue

        try:
            # Run PyTorch FID
            result = subprocess.run(
                ["python", "-m", "pytorch_fid", gen_super_dir, real_super_dir, "--batch-size", str(batch_size)],
                capture_output=True,
                text=True
            )
            print(f"Command Output for super-class {i}: {result.stdout.strip()}")  # For debugging

            # Extract FID value
            fid = float(result.stdout.split("FID:")[-1].strip())
            fid_values.append(fid)
            print(f"Super-class {i}: FID = {fid}")
        except Exception as e:
            print(f"Error calculating FID for super-class {i}: {e}")
            continue

    # Calculate mean and standard deviation
    if fid_values:
        mean_intra_fid = np.mean(fid_values)
        std_intra_fid = np.std(fid_values)
        print(f"Intra-FID: {mean_intra_fid:.2f} ± {std_intra_fid:.2f}")
    else:
        mean_intra_fid, std_intra_fid = None, None
        print("No FID values calculated.")

    return mean_intra_fid, std_intra_fid

# 경로 설정
real_images_dir = "./real-images/superclass"
generated_images_dir = "./intra_generated-images"

# Intra-FID 계산 실행
mean_intra_fid, std_intra_fid = calculate_intra_fid(real_images_dir, generated_images_dir)
