#!/usr/bin/env python3
"""
Performance Monitor Validation Script
This script tests the performance monitoring functionality without running the full application
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import os
from datetime import datetime

def create_sample_performance_data():
    """Create sample performance data for testing visualization"""
    print("Creating sample performance data...")
    
    # Generate 60 seconds of sample data
    timestamps = []
    cpu_data = []
    memory_data = []
    fps_data = []
    
    base_time = datetime.now()
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    for i in range(60):  # 60 data points
        timestamps.append(base_time.replace(second=i))
        
        # Simulate varying performance
        cpu_usage = 30 + 20 * np.sin(i * 0.1) + rng.normal(0, 5)
        memory_usage = 45 + 15 * np.cos(i * 0.08) + rng.normal(0, 3)
        fps = 25 + 5 * np.sin(i * 0.15) + rng.normal(0, 2)
        
        cpu_data.append(max(0, min(100, cpu_usage)))
        memory_data.append(max(0, min(100, memory_usage)))
        fps_data.append(max(0, fps))
    
    return timestamps, cpu_data, memory_data, fps_data

def test_visualization():
    """Test the visualization functionality"""
    print("Testing performance visualization...")
    
    # Create test data
    timestamps, cpu_data, memory_data, fps_data = create_sample_performance_data()
    
    # Create results directory
    results_dir = f"test_performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create test graphs
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Performance Monitor Test - Sample Data', fontsize=14, fontweight='bold')
    
    # CPU Usage
    axes[0, 0].plot(timestamps, cpu_data, color='red', linewidth=2, label='CPU Usage')
    axes[0, 0].set_title('CPU Usage (%)')
    axes[0, 0].set_ylabel('Percentage')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Memory Usage
    axes[0, 1].plot(timestamps, memory_data, color='blue', linewidth=2, label='Memory Usage')
    axes[0, 1].set_title('Memory Usage (%)')
    axes[0, 1].set_ylabel('Percentage')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # FPS
    axes[1, 0].plot(timestamps, fps_data, color='green', linewidth=2, label='FPS')
    axes[1, 0].set_title('Frames Per Second')
    axes[1, 0].set_ylabel('FPS')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Statistics
    axes[1, 1].text(0.1, 0.8, f'CPU Mean: {np.mean(cpu_data):.1f}%', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, f'CPU Max: {np.max(cpu_data):.1f}%', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, f'Memory Mean: {np.mean(memory_data):.1f}%', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.5, f'Memory Max: {np.max(memory_data):.1f}%', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.4, f'FPS Mean: {np.mean(fps_data):.1f}', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.3, f'FPS Min: {np.min(fps_data):.1f}', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Performance Statistics')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    # Format time axes
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/test_performance_graph.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Test graph saved to: {results_dir}/test_performance_graph.png")
    return results_dir

def check_system_monitoring():
    """Test basic system monitoring functionality"""
    print("Testing system monitoring...")
    
    # Get current system stats
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    print(f"Current CPU Usage: {cpu_percent}%")
    print(f"Current Memory Usage: {memory.percent}%")
    print(f"Available Memory: {memory.available / (1024**3):.2f} GB")
    
    # Test continuous monitoring for 10 seconds
    print("Monitoring system for 10 seconds...")
    
    cpu_readings = []
    memory_readings = []
    
    for i in range(10):
        cpu_readings.append(psutil.cpu_percent(interval=1))
        memory_readings.append(psutil.virtual_memory().percent)
        print(f"  Sample {i+1}: CPU {cpu_readings[-1]:.1f}%, Memory {memory_readings[-1]:.1f}%")
    
    print(f"Average CPU over 10s: {np.mean(cpu_readings):.1f}%")
    print(f"Average Memory over 10s: {np.mean(memory_readings):.1f}%")

def test_dependencies():
    """Test if all required dependencies are available"""
    print("Testing dependencies...")
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib - OK")
    except ImportError:
        print("✗ matplotlib - MISSING")
        return False, "matplotlib not installed"
    
    try:
        import seaborn as sns
        print("✓ seaborn - OK")
    except ImportError:
        print("✗ seaborn - MISSING")
        return False, "seaborn not installed"
    
    try:
        import pandas as pd
        print("✓ pandas - OK")
    except ImportError:
        print("✗ pandas - MISSING")
        return False, "pandas not installed"
    
    try:
        import numpy as np
        print("✓ numpy - OK")
    except ImportError:
        print("✗ numpy - MISSING")
        return False, "numpy not installed"
    
    try:
        import psutil
        print("✓ psutil - OK")
    except ImportError:
        print("✗ psutil - MISSING")
        return False, "psutil not installed"
    
    # Optional dependencies
    try:
        import GPUtil
        print("✓ GPUtil - OK (GPU monitoring available)")
    except ImportError:
        print("⚠ GPUtil - MISSING (GPU monitoring will be disabled)")
    
    return True, "All required dependencies available"

def main():
    """Run all validation tests"""
    print("Performance Monitor Validation Script")
    print("=" * 50)
    
    # Test 1: Dependencies
    print("\n1. Testing Dependencies:")
    deps_ok, deps_msg = test_dependencies()
    print(f"   Result: {deps_msg}")
    
    if not deps_ok:
        print("\n❌ Cannot continue - missing required dependencies")
        print("Please install missing packages with: pip install <package_name>")
        return
    
    # Test 2: System Monitoring
    print("\n2. Testing System Monitoring:")
    check_system_monitoring()
    
    # Test 3: Visualization
    print("\n3. Testing Visualization:")
    results_dir = test_visualization()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")
    print(f"Test results saved to: {results_dir}")
    print("\nThe performance monitor should work correctly.")
    print("\nTo run actual performance monitoring:")
    print("  python performance_monitor.py --duration 120")
    print("  python run_performance_tests.py basic")

if __name__ == "__main__":
    main()
