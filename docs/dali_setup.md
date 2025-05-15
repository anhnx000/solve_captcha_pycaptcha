# Using NVIDIA DALI for GPU-Accelerated Data Loading

This project now supports NVIDIA Data Loading Library (DALI) for faster data loading directly to GPU memory during training.

## What is DALI?

NVIDIA DALI is a library for data loading and preprocessing optimized for GPU. It provides significantly improved performance for training deep learning models by:

- Loading data directly to GPU memory
- Performing data augmentation on GPU
- Overlapping data loading with model computation
- Utilizing GPU-optimized image decoding and processing

## Installation

DALI requires CUDA. Make sure you have a compatible CUDA version installed before proceeding.

### Install via pip

```bash
# For CUDA 11.0
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110

# For CUDA 11.1
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda111

# For CUDA 11.2
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda112
```

Choose the command that matches your CUDA version.

## Using DALI in this Project

### Option 1: Use the Main Launcher

The easiest way to use DALI is to add the `--use_dali` flag when running the launcher:

```bash
# Basic usage with DALI enabled
python launcher.py --model_name resnet --use_dali

# With additional options
python launcher.py --model_name resnet --use_dali --use_ctc --save_path ./checkpoint/dali
```

You can also use the example script that includes DALI:

```bash
python examples/train_with_dali_launcher.py
```

### Option 2: Using the DataModule Directly

You can also create a captcha datamodule with DALI enabled directly in your code:

```python
from data import captcha_dm

# Create data module with DALI
dm = captcha_dm(
    batch_size=64,
    num_workers=8,
    use_dali=True  # Enable DALI
)
```

Or use the dedicated example training script:

```bash
python examples/train_with_dali.py
```

## DALI with Albumentations Integration

This project now includes a unique integration of Albumentations with DALI, allowing you to use both libraries together:

### How It Works

1. DALI loads and decodes images on CPU
2. Albumentations performs image transformations on CPU using a Python operator
3. The transformed images are transferred to GPU for model inference
4. This approach combines Albumentations' rich augmentation library with DALI's efficient data loading

### Advantages and Trade-offs

**Advantages:**
- Access to all Albumentations transformations
- Maintain identical augmentation between DALI and non-DALI implementations
- Simpler code maintenance with a single augmentation definition

**Trade-offs:**
- Image augmentation still runs on CPU, not fully leveraging GPU acceleration
- Potential performance decrease compared to pure DALI operations
- Additional CPU-GPU transfer may slightly reduce performance

### Alternative Implementation

The code includes a commented-out alternative implementation that uses DALI's native operations for full GPU acceleration. You can switch to this implementation by modifying the `captcha_dali_pipeline` function.

```python
# Use pure DALI operations for maximum GPU utilization
# This requires adapting your augmentations to DALI's operations
def captcha_dali_pipeline(...):
    # ...
    # Uncomment the pure DALI implementation
    # ...
```

## Expected Performance Improvements

Using DALI can provide substantial performance improvements during training, especially for image-heavy workloads like captcha recognition:

- Reduced CPU bottlenecks
- Lower memory usage
- Faster training iterations
- Better GPU utilization

The code will automatically fall back to standard PyTorch DataLoaders if DALI is not available.

## Troubleshooting

If you encounter issues with DALI, try:

1. Ensuring your CUDA drivers are up to date
2. Installing the correct version of DALI for your CUDA version
3. Setting `use_dali=False` to fall back to standard PyTorch DataLoaders
4. Checking NVIDIA's [DALI documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/) 
