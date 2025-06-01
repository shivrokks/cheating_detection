#!/usr/bin/env python3
"""
ROCm Model Optimization Script
This script pre-warms the YOLO models to trigger MIOpen kernel optimization
for faster subsequent runs.
"""

import os
import sys
import torch
import numpy as np
import time

def setup_rocm_optimization():
    """Set up environment for ROCm kernel optimization"""
    print("Setting up ROCm optimization environment...")
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
    os.environ['MIOPEN_FIND_ENFORCE'] = '3'  # Enable comprehensive kernel search
    os.environ['MIOPEN_DEBUG_FIND_ONLY_SOLVER'] = '1'  # Find optimal solvers
    os.environ['MIOPEN_FIND_MODE'] = '1'  # Enable find mode for kernel optimization
    os.environ['MIOPEN_LOG_LEVEL'] = '4'  # Verbose logging for optimization
    os.environ['MIOPEN_ENABLE_LOGGING'] = '1'  # Enable logging for optimization
    os.environ['ROCM_PATH'] = '/opt/rocm'
    os.environ['MIOPEN_SYSTEM_DB_PATH'] = '/opt/rocm/share/miopen/db'
    os.environ['MIOPEN_USER_DB_PATH'] = os.path.expanduser('~/.config/miopen')
    os.environ['HIP_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Create MIOpen user database directory
    miopen_user_path = os.path.expanduser('~/.config/miopen')
    os.makedirs(miopen_user_path, exist_ok=True)
    print(f"MIOpen user database path: {miopen_user_path}")

def warm_up_model(model_path, device, warmup_iterations=10):
    """Warm up a YOLO model with multiple inference passes"""
    print(f"Loading model: {model_path}")
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        model.to(device)
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Create dummy input images of different sizes for comprehensive optimization
    input_sizes = [(320, 320), (416, 416), (640, 640)]
    
    print(f"Starting warm-up with {warmup_iterations} iterations per size...")
    
    for size in input_sizes:
        print(f"  Optimizing for input size: {size}")
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        
        # Warm up the model with multiple passes
        with torch.no_grad():
            for i in range(warmup_iterations):
                try:
                    start_time = time.time()
                    results = model(dummy_image, verbose=False, device=device)
                    end_time = time.time()
                    print(f"    Iteration {i+1}/{warmup_iterations}: {(end_time - start_time)*1000:.1f}ms")
                    
                    # Clear cache after each iteration
                    if device.startswith('cuda'):
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"    Error in iteration {i+1}: {e}")
                    continue
    
    print(f"Model warm-up completed for {model_path}")
    return model

def main():
    print("=" * 60)
    print("ROCm Model Optimization Script")
    print("=" * 60)
    
    # Set up ROCm optimization environment
    setup_rocm_optimization()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available!")
        return 1
    
    device = "cuda:0"  # Force discrete GPU
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Model paths
    model_path = "./model/best_yolov12.pt"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return 1
    
    print("\nStarting model optimization process...")
    print("This will take several minutes as ROCm builds optimized kernels.")
    print("Progress will be saved to MIOpen database for future runs.\n")
    
    try:
        # Warm up the YOLO model
        warm_up_model(model_path, device, warmup_iterations=15)
        
        print("\n" + "=" * 60)
        print("Optimization completed successfully!")
        print("The models should now run much faster in subsequent runs.")
        print("Optimized kernels are cached in: ~/.config/miopen")
        print("=" * 60)
        
    except Exception as e:
        print(f"ERROR during optimization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
