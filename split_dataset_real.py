import os
import shutil
import random
import math

# Define paths
src_dir = "/home/huongntt/Works/anhnx/solve_captcha_pycaptcha/dataset_real"
train_dir = os.path.join(src_dir, "train")
test_dir = os.path.join(src_dir, "test")

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(src_dir) if f.endswith(".png")]
random.shuffle(image_files)  # Shuffle to ensure random distribution

# Calculate split
total_files = len(image_files)
train_count = math.floor(total_files * 0.8)

# Split files
train_files = image_files[:train_count]
test_files = image_files[train_count:]

print(f"Total files: {total_files}")
print(f"Train files: {len(train_files)}")
print(f"Test files: {len(test_files)}")

# Copy files to respective directories
for file in train_files:
    src_path = os.path.join(src_dir, file)
    dst_path = os.path.join(train_dir, file)
    shutil.copy2(src_path, dst_path)
    print(f"Copied {file} to train set")

for file in test_files:
    src_path = os.path.join(src_dir, file)
    dst_path = os.path.join(test_dir, file)
    shutil.copy2(src_path, dst_path)
    print(f"Copied {file} to test set")

print("Dataset split complete!")
