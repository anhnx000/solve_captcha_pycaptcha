#!/usr/bin/env python3
"""
Example script for running the launcher with DALI GPU-accelerated data loading enabled
"""
import os
import sys
import subprocess

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def run_training():
    """
    Run launcher.py with DALI enabled and other parameters
    """
    # Ensure we're in the project root directory
    os.chdir(project_root)
    
    # Prepare the command
    cmd = [
        "python", "launcher.py",
        "--model_name", "resnet",   # Can be one of: resnet, efficientnet, vit, mobilenet, trocr
        "--use_dali",               # Enable DALI GPU-accelerated data loading
        "--save_path", "./checkpoint/dali"
    ]
    
    # Optionally add other arguments
    # Uncomment any of these to use them
    # cmd.extend(["--use_ctc"])    # Enable CTC loss
    # cmd.extend(["--resume_from_checkpoint", "./checkpoint/dali/latest.ckpt"])
    
    # Print the command for verification
    print(f"Running command: {' '.join(cmd)}")
    
    # Execute the command
    subprocess.run(cmd)

if __name__ == "__main__":
    # Check if DALI is available first
    try:
        import nvidia.dali
        print("NVIDIA DALI is available. Running training with GPU-accelerated data loading.")
        run_training()
    except ImportError:
        print("NVIDIA DALI is not installed. Please install it first:")
        print("pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110")
        print("\nOr run without DALI by removing the --use_dali flag.")
        sys.exit(1) 
