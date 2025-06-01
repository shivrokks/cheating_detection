#!/usr/bin/env python3
"""
Performance Test Script for VRAM Optimization
This script demonstrates the performance improvements after removing VRAM limitations
"""

import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO
import os

def test_model_performance():
    """Test model performance with different configurations"""
    
    print("=== YOLO Performance Test with Removed VRAM Limitations ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load model
    model = YOLO("./model/best_yolov12.pt")
    model.to("cuda:0")
    
    # Create test images of different sizes
    test_sizes = [
        (640, 480, "VGA"),
        (1280, 720, "HD 720p"), 
        (1920, 1080, "Full HD 1080p"),
        (2560, 1440, "QHD 1440p"),
        (3840, 2160, "4K UHD")
    ]
    
    inference_sizes = [320, 640, 1280, 1920]  # Model input sizes
    
    print("\n=== PERFORMANCE COMPARISON ===")
    print("Testing inference with different input resolutions...")
    
    for width, height, name in test_sizes:
        print(f"\n--- Testing {name} ({width}x{height}) ---")
        
        # Create test image
        test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        for imgsz in inference_sizes:
            if imgsz > min(width, height):
                continue
                
            print(f"  Model input size: {imgsz}px")
            
            # Warm up
            for _ in range(3):
                with torch.no_grad():
                    _ = model(test_image, verbose=False, device="cuda:0", imgsz=imgsz, half=False)
            
            # Time inference
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(10):
                with torch.no_grad():
                    results = model(test_image, verbose=False, device="cuda:0", imgsz=imgsz, half=False)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            fps = 1.0 / avg_time
            
            # Memory usage
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            
            print(f"    Inference time: {avg_time*1000:.1f}ms")
            print(f"    FPS: {fps:.1f}")
            print(f"    GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
            
            # Count detections
            detection_count = 0
            if results and len(results) > 0 and results[0].boxes is not None:
                detection_count = len(results[0].boxes)
            print(f"    Detections: {detection_count}")

def test_memory_limits():
    """Test maximum memory usage capabilities"""
    
    print("\n=== MEMORY STRESS TEST ===")
    print("Testing maximum VRAM utilization...")
    
    model = YOLO("./model/best_yolov12.pt")
    model.to("cuda:0")
    
    # Test with very large images
    large_sizes = [
        (7680, 4320, "8K UHD"),   # 8K resolution
        (3840, 2160, "4K UHD"),   # 4K resolution
        (2560, 1440, "QHD"),      # QHD resolution
    ]
    
    for width, height, name in large_sizes:
        print(f"\n--- Testing {name} ({width}x{height}) ---")
        
        try:
            # Create large test image
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            print(f"  Image size: {test_image.nbytes / 1024**2:.1f} MB")
            
            # Test with maximum model input size
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                results = model(test_image, verbose=False, device="cuda:0", imgsz=1280, half=False)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            inference_time = end_time - start_time
            
            # Memory usage
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            utilization = (allocated / total) * 100
            
            print(f"  ✅ SUCCESS: {inference_time*1000:.1f}ms")
            print(f"  Memory utilization: {utilization:.1f}% ({allocated:.2f}GB / {total:.1f}GB)")
            
            # Count detections
            detection_count = 0
            if results and len(results) > 0 and results[0].boxes is not None:
                detection_count = len(results[0].boxes)
            print(f"  Detections found: {detection_count}")
            
        except Exception as e:
            print(f"  ❌ FAILED: {str(e)}")
        
        # Clear memory between tests
        torch.cuda.empty_cache()

def show_optimization_summary():
    """Show summary of optimizations applied"""
    
    print("\n" + "="*60)
    print("VRAM LIMITATION REMOVAL - OPTIMIZATION SUMMARY")
    print("="*60)
    
    print("\n✅ LIMITATIONS REMOVED:")
    print("  • Frame resizing constraints (1280px limit)")
    print("  • Small model input size (320px → 1280px)")
    print("  • Frequent GPU cache clearing")
    print("  • Memory-saving batch size limitations")
    print("  • Processing frequency limitations")
    
    print("\n🚀 PERFORMANCE IMPROVEMENTS:")
    print("  • 4x larger model input resolution (320px → 1280px)")
    print("  • Full frame resolution processing")
    print("  • Increased detection processing frequency")
    print("  • Maximum VRAM utilization (up to 12GB)")
    print("  • Better detection accuracy with high-resolution inference")
    
    print("\n⚙️  GPU OPTIMIZATIONS ENABLED:")
    print("  • CuDNN benchmark mode for consistent performance")
    print("  • Non-deterministic operations for speed")
    print("  • TF32 computation for faster inference")
    print("  • ROCm-specific AMD GPU optimizations")
    print("  • Discrete GPU enforcement (RX 6800M)")
    
    print("\n📊 EXPECTED BENEFITS:")
    print("  • Higher detection accuracy due to larger input resolution")
    print("  • Better small object detection capabilities")
    print("  • Faster inference with optimized GPU settings")
    print("  • Full utilization of 12GB VRAM capacity")
    print("  • Improved real-time performance")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available - cannot run performance tests")
        exit(1)
    
    # Configure optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(0)
    
    show_optimization_summary()
    test_model_performance()
    test_memory_limits()
    
    print("\n" + "="*60)
    print("PERFORMANCE TEST COMPLETED")
    print("="*60)
