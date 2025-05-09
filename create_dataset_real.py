import os
import shutil

# Define paths
source_dir = '/home/huongntt/Works/anhnx/solve_captcha_pycaptcha/dataset_xa/captcha'
label_file = '/home/huongntt/Works/anhnx/solve_captcha_pycaptcha/dataset_xa/captcha_label.txt'
target_dir = '/home/huongntt/Works/anhnx/solve_captcha_pycaptcha/dataset_real'

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Dictionary to track label counts
label_count = {}

# Read label file
with open(label_file, 'r') as f:
    lines = f.readlines()

# Process each line
for line in lines:
    line = line.strip()
    if not line:
        continue
    
    # Split by tab
    parts = line.split('\t')
    if len(parts) != 2:
        print(f"Warning: Invalid line format: {line}")
        continue
    
    image_name, label = parts
    
    # Update count for this label
    if label not in label_count:
        label_count[label] = 0
    label_count[label] += 1
    
    # New filename: {label}.{count}.png
    new_filename = f"{label}.{label_count[label]}.png"
    
    # Source and target paths
    source_path = os.path.join(source_dir, image_name)
    target_path = os.path.join(target_dir, new_filename)
    
    # Copy file
    if os.path.exists(source_path):
        shutil.copy2(source_path, target_path)
        print(f"Copied: {image_name} → {new_filename}")
    else:
        print(f"Warning: Source file not found: {source_path}")

print(f"Completed. {sum(label_count.values())} images processed.")
