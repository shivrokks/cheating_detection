#!/usr/bin/env python3
"""
ROCm-optimized launcher for the cheating detection system.
Sets necessary environment variables and runs the main script.
"""

import os
import sys
import subprocess

# Set ROCm environment variables for optimal performance
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:256'  # Reduced to prevent hanging
os.environ['HIP_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['AMD_SERIALIZE_KERNEL'] = '3'  # Enhanced debugging for HIP kernels
os.environ['HSA_ENABLE_SDMA'] = '0'  # Disable SDMA for stability
os.environ['ROCBLAS_LAYER'] = '0'  # Reduce verbose logging
os.environ['AMD_LOG_LEVEL'] = '1'  # Reduce logging verbosity
os.environ['TORCH_USE_HIP_DSA'] = '1'  # Enable device-side assertions
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force PyTorch to see only GPU 0

# Enable ROCm kernel tuning and optimization for faster performance
os.environ['MIOPEN_FIND_ENFORCE'] = '3'  # Enable comprehensive kernel search
os.environ['MIOPEN_DEBUG_FIND_ONLY_SOLVER'] = '1'  # Find optimal solvers
os.environ['MIOPEN_FIND_MODE'] = '1'  # Enable find mode for kernel optimization
os.environ['MIOPEN_LOG_LEVEL'] = '3'  # Set appropriate logging for tuning
os.environ['MIOPEN_ENABLE_LOGGING'] = '1'  # Enable logging for optimization
os.environ['ROCM_PATH'] = '/opt/rocm'  # Ensure ROCm path is set
os.environ['MIOPEN_SYSTEM_DB_PATH'] = '/opt/rocm/share/miopen/db'  # System database path
os.environ['MIOPEN_USER_DB_PATH'] = os.path.expanduser('~/.config/miopen')  # User database path

# Create MIOpen user database directory if it doesn't exist
miopen_user_path = os.path.expanduser('~/.config/miopen')
os.makedirs(miopen_user_path, exist_ok=True)

# Run the main application with the same arguments
if __name__ == "__main__":
    print("Starting Cheating Detection System with ROCm optimization...")
    print("Enabling MIOpen kernel tuning for optimal performance...")
    print("Note: First run may be slower as ROCm builds optimized kernels.")
    print("Environment variables set:")
    print(f"  HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")
    print(f"  PYTORCH_HIP_ALLOC_CONF: {os.environ.get('PYTORCH_HIP_ALLOC_CONF')}")
    print(f"  HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"  MIOPEN_FIND_ENFORCE: {os.environ.get('MIOPEN_FIND_ENFORCE')}")
    print(f"  MIOPEN_FIND_MODE: {os.environ.get('MIOPEN_FIND_MODE')}")
    print(f"  MIOPEN_USER_DB_PATH: {os.environ.get('MIOPEN_USER_DB_PATH')}")
    print()
    
    # Pass through any command line arguments
    cmd = ['python', 'main.py'] + sys.argv[1:]
    subprocess.run(cmd)
