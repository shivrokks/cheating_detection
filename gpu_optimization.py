#!/usr/bin/env python3
"""
GPU Detection and Configuration Utility
This module provides unified GPU detection for both NVIDIA CUDA and AMD ROCm GPUs.
"""

import subprocess
import logging
import sys

logger = logging.getLogger(__name__)

def detect_gpu_type():
    """
    Detect the type of GPU available (NVIDIA CUDA or AMD ROCm)
    
    Returns:
        tuple: (gpu_type, device_name) where gpu_type is 'cuda', 'rocm', or 'cpu'
    """
    
    # First try to detect AMD ROCm GPU
    try:
        result = subprocess.run(['rocm-smi', '--showid'], 
                              capture_output=True, text=True, check=True)
        if 'GPU' in result.stdout and 'AMD' in result.stdout:
            # Extract GPU name from output
            for line in result.stdout.split('\n'):
                if 'Device Name:' in line:
                    gpu_name = line.split('Device Name:')[1].strip()
                    logger.info(f"AMD ROCm GPU detected: {gpu_name}")
                    return 'rocm', gpu_name
            return 'rocm', 'AMD GPU'
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.debug("ROCm not available or no AMD GPU detected")
    
    # Try to detect NVIDIA CUDA GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"NVIDIA CUDA GPU detected: {gpu_name}")
            return 'cuda', gpu_name
    except ImportError:
        logger.debug("PyTorch not available")
    except Exception as e:
        logger.debug(f"CUDA detection failed: {e}")
    
    # Try nvidia-smi as fallback
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        if result.stdout.strip():
            gpu_name = result.stdout.strip().split('\n')[0]
            logger.info(f"NVIDIA GPU detected via nvidia-smi: {gpu_name}")
            return 'cuda', gpu_name
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.debug("nvidia-smi not available")
    
    logger.info("No GPU detected, falling back to CPU")
    return 'cpu', 'CPU'

def get_torch_device():
    """
    Get the appropriate PyTorch device string based on available GPU
    
    Returns:
        str: Device string ('cuda', 'cpu', or custom device for ROCm)
    """
    gpu_type, gpu_name = detect_gpu_type()
    
    if gpu_type == 'cuda':
        # Standard CUDA device
        return 'cuda'
    elif gpu_type == 'rocm':
        # For ROCm, we need to check if PyTorch was compiled with ROCm support
        try:
            import torch
            # Check if PyTorch has ROCm support
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                # PyTorch with ROCm support
                if torch.cuda.is_available():  # In PyTorch with ROCm, this checks for AMD GPUs
                    return 'cuda'  # PyTorch with ROCm still uses 'cuda' as device string
            
            # If we reach here, PyTorch doesn't have ROCm support
            logger.warning(f"AMD GPU detected ({gpu_name}) but PyTorch doesn't have ROCm support")
            logger.warning("Install PyTorch with ROCm support for GPU acceleration")
            return 'cpu'
            
        except ImportError:
            logger.warning("PyTorch not available")
            return 'cpu'
    else:
        return 'cpu'

def install_rocm_pytorch():
    """
    Provide instructions for installing PyTorch with ROCm support
    """
    print("\\n" + "="*60)
    print("AMD GPU DETECTED - ROCm PyTorch Installation Required")
    print("="*60)
    print("Your system has AMD GPUs but PyTorch is compiled for CUDA.")
    print("To use your AMD GPU, install PyTorch with ROCm support:")
    print()
    print("1. Uninstall current PyTorch:")
    print("   pip uninstall torch torchvision torchaudio")
    print()
    print("2. Install PyTorch with ROCm support:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7")
    print()
    print("3. Or for ROCm 6.0+:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0")
    print()
    print("4. Verify installation:")
    print("   python -c \\"import torch; print(f'ROCm available: {torch.cuda.is_available()}'); print('Device count:', torch.cuda.device_count())\\"")
    print()
    print("5. Restart the performance monitor after installation")
    print("="*60)

def get_gpu_info():
    """
    Get detailed GPU information for monitoring
    
    Returns:
        dict: GPU information including type, name, and monitoring capabilities
    """
    gpu_type, gpu_name = detect_gpu_type()
    
    info = {
        'type': gpu_type,
        'name': gpu_name,
        'torch_device': get_torch_device(),
        'monitoring_available': False,
        'compute_available': False
    }
    
    if gpu_type == 'rocm':
        # Check if ROCm monitoring tools are available
        try:
            subprocess.run(['rocm-smi', '--help'], 
                          capture_output=True, text=True, check=True)
            info['monitoring_available'] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Check if PyTorch can use the GPU
        try:
            import torch
            if torch.cuda.is_available():
                info['compute_available'] = True
        except ImportError:
            pass
            
    elif gpu_type == 'cuda':
        # Check NVIDIA monitoring and compute
        try:
            import torch
            if torch.cuda.is_available():
                info['compute_available'] = True
        except ImportError:
            pass
        
        try:
            subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
            info['monitoring_available'] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    
    return info

if __name__ == "__main__":
    # Test the GPU detection
    logging.basicConfig(level=logging.INFO)
    
    print("GPU Detection Results:")
    print("="*40)
    
    gpu_info = get_gpu_info()
    print(f"GPU Type: {gpu_info['type']}")
    print(f"GPU Name: {gpu_info['name']}")
    print(f"PyTorch Device: {gpu_info['torch_device']}")
    print(f"Monitoring Available: {gpu_info['monitoring_available']}")
    print(f"Compute Available: {gpu_info['compute_available']}")
    
    if gpu_info['type'] == 'rocm' and not gpu_info['compute_available']:
        install_rocm_pytorch()
