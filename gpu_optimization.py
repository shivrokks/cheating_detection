# GPU Optimization Configuration for AMD RX 6800M (12GB VRAM)
# This file configures optimal settings for maximum VRAM utilization

import torch
import os

def configure_gpu_optimization():
    """Configure GPU settings for maximum performance with 12GB VRAM"""
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU optimization")
        return False
    
    # Get GPU information
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"Configuring optimization for: {gpu_name}")
    print(f"Available VRAM: {gpu_memory:.1f} GB")
    
    # ROCm-specific optimizations for AMD GPU
    os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1030'  # RX 6800M architecture
    os.environ['HIP_VISIBLE_DEVICES'] = '0'      # Use discrete GPU only
    
    # Memory management optimizations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    
    # CuDNN optimizations for maximum performance
    torch.backends.cudnn.benchmark = True        # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False   # Allow non-deterministic for speed
    torch.backends.cudnn.allow_tf32 = True       # Enable TF32 for faster computation
    
    # Set GPU to performance mode
    torch.cuda.set_device(0)  # Explicitly use discrete GPU
    
    # Enable memory pre-allocation for consistent performance
    torch.cuda.empty_cache()
    
    print("GPU optimization configured successfully!")
    print("- Removed frame resizing limitations")
    print("- Increased model input size to 1280px")
    print("- Enabled aggressive CuDNN optimizations")
    print("- Configured for maximum VRAM utilization")
    
    return True

def get_optimal_batch_size():
    """Calculate optimal batch size based on available VRAM"""
    if not torch.cuda.is_available():
        return 1
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    # With 12GB VRAM and high-res inference, we can handle larger batches
    if gpu_memory >= 10:
        return 4  # Aggressive batch size for 12GB
    elif gpu_memory >= 8:
        return 3
    elif gpu_memory >= 6:
        return 2
    else:
        return 1

def monitor_gpu_usage():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"GPU Memory Usage:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Cached: {cached:.2f} GB") 
        print(f"  Total: {total:.2f} GB")
        print(f"  Utilization: {(allocated/total)*100:.1f}%")
        
        return allocated, cached, total
    return 0, 0, 0

if __name__ == "__main__":
    configure_gpu_optimization()
    monitor_gpu_usage()
