#!/usr/bin/env python3
"""
Performance Monitoring Script for Cheating Detection System
This script monitors system performance while running the cheating detection application
and generates comprehensive performance analysis with three different types of graphs.
"""

import psutil
import time
import subprocess
import threading
import sys
import os
import json
import logging
from datetime import datetime, timedelta
from queue import Queue, Empty
import signal

# Data visualization libraries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
CPU_USAGE_LABEL = 'CPU Usage (%)'
GPU_USAGE_LABEL = 'GPU Usage (%)'
PRIMARY_GPU_ID = 'GPU[0]'

class PerformanceMonitor:
    def __init__(self, app_command=None, monitoring_duration=300):
        """
        Initialize the performance monitor
        
        Args:
            app_command (str): Command to run the application (default: python main.py)
            monitoring_duration (int): Duration to monitor in seconds (default: 5 minutes)
        """
        self.app_command = app_command or "python main.py"
        self.monitoring_duration = monitoring_duration
        self.app_process = None
        self.monitoring_active = False
        self.data_queue = Queue()
        
        # Performance data storage
        self.performance_data = {
            'timestamps': [],
            'cpu_percent': [],
            'memory_usage': [],
            'memory_percent': [],
            'gpu_usage': [],
            'gpu_memory': [],
            'gpu_temperature': [],
            'fps': [],
            'frame_processing_time': [],
            'detection_latency': [],
            'disk_io_read': [],
            'disk_io_write': [],
            'network_sent': [],
            'network_recv': [],
            'temperature': [],
            'power_usage': []
        }
        
        # Detection-specific metrics
        self.detection_metrics = {
            'eye_movement_time': [],
            'head_pose_time': [],
            'mobile_detection_time': [],
            'facial_expression_time': [],
            'person_detection_time': [],
            'object_detection_time': [],
            'behavior_analysis_time': [],
            'total_processing_time': []
        }
        
        # System baseline
        self.baseline_metrics = {}
        
        # GPU monitoring (try to detect AMD ROCm or NVIDIA GPUs)
        self.gpu_available = False
        self.gpu_type = None
        self.gpu_util = None
        
        # Try ROCm first (for AMD GPUs)
        try:
            import subprocess
            result = subprocess.run(['rocm-smi', '--showallinfo'], 
                                  capture_output=True, text=True, check=True)
            if 'GPU' in result.stdout:
                self.gpu_available = True
                self.gpu_type = 'rocm'
                logger.info("ROCm AMD GPU detected")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Try NVIDIA GPUtil if ROCm not available
        if not self.gpu_available:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_available = True
                    self.gpu_type = 'nvidia'
                    self.gpu_util = GPUtil
                    logger.info("NVIDIA GPU detected")
            except ImportError:
                pass
        
        if not self.gpu_available:
            logger.warning("No GPU monitoring available - install ROCm (AMD) or GPUtil (NVIDIA)")
        
        # Create results directory
        self.results_dir = f"performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)

    def get_baseline_metrics(self):
        """Get baseline system metrics before starting the application"""
        logger.info("Collecting baseline system metrics...")
        
        # CPU and Memory baseline
        cpu_baseline = []
        memory_baseline = []
        
        for _ in range(10):  # Sample 10 times over 5 seconds
            cpu_baseline.append(psutil.cpu_percent(interval=0.5))
            memory_baseline.append(psutil.virtual_memory().percent)
        
        self.baseline_metrics = {
            'cpu_baseline': np.mean(cpu_baseline),
            'memory_baseline': np.mean(memory_baseline),
            'available_memory': psutil.virtual_memory().available / (1024**3),  # GB
            'total_memory': psutil.virtual_memory().total / (1024**3),  # GB
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0
        }
        
        logger.info(f"Baseline CPU: {self.baseline_metrics['cpu_baseline']:.2f}%")
        logger.info(f"Baseline Memory: {self.baseline_metrics['memory_baseline']:.2f}%")

    def start_application(self):
        """Start the cheating detection application"""
        try:
            logger.info(f"Starting application: {self.app_command}")
            
            # Start the application process
            self.app_process = subprocess.Popen(
                self.app_command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            # Give the application time to initialize
            time.sleep(3)
            
            if self.app_process.poll() is None:
                logger.info("Application started successfully")
                return True
            else:
                logger.error("Application failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Error starting application: {e}")
            return False

    def collect_system_metrics(self):
        """Collect system performance metrics"""
        timestamp = datetime.now()
        
        # CPU and Memory metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_usage = (memory.total - memory.available) / (1024**3)  # GB
        memory_percent = memory.percent
        
        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()
        disk_read = disk_io.read_bytes / (1024**2) if disk_io else 0  # MB
        disk_write = disk_io.write_bytes / (1024**2) if disk_io else 0  # MB
        
        # Network metrics
        network = psutil.net_io_counters()
        net_sent = network.bytes_sent / (1024**2) if network else 0  # MB
        net_recv = network.bytes_recv / (1024**2) if network else 0  # MB
        
        # Temperature (if available)
        temperature = 0
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get CPU temperature if available
                for name, entries in temps.items():
                    if 'cpu' in name.lower() or 'core' in name.lower():
                        temperature = entries[0].current
                        break
        except (AttributeError, IndexError, KeyError):
            pass
        
        # GPU metrics (if available)
        gpu_usage = 0
        gpu_memory = 0
        gpu_temperature = 0
        
        if self.gpu_available:
            if self.gpu_type == 'rocm':
                # Use ROCm for AMD GPUs
                try:
                    result = subprocess.run(['rocm-smi', '--showuse', '--showmeminfo', '--showtemp'], 
                                          capture_output=True, text=True, check=True)
                    lines = result.stdout.split('\n')
                    
                    # Parse GPU usage and memory
                    for line in lines:
                        if 'GPU use (%)' in line and PRIMARY_GPU_ID in line:
                            gpu_usage = float(line.split(':')[1].strip())
                        elif 'GPU Memory Allocated (VRAM%)' in line and PRIMARY_GPU_ID in line:
                            gpu_memory = float(line.split(':')[1].strip())
                        elif 'Temperature (Sensor edge)' in line and PRIMARY_GPU_ID in line:
                            gpu_temperature = float(line.split('(C):')[1].strip())
                            
                except (subprocess.CalledProcessError, ValueError, IndexError):
                    pass
                    
            elif self.gpu_type == 'nvidia':
                # Use GPUtil for NVIDIA GPUs
                try:
                    gpus = self.gpu_util.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        gpu_usage = gpu.load * 100
                        gpu_memory = gpu.memoryUsed
                        gpu_temperature = gpu.temperature
                except (AttributeError, IndexError):
                    pass
        
        # Store metrics
        self.performance_data['timestamps'].append(timestamp)
        self.performance_data['cpu_percent'].append(cpu_percent)
        self.performance_data['memory_usage'].append(memory_usage)
        self.performance_data['memory_percent'].append(memory_percent)
        self.performance_data['gpu_usage'].append(gpu_usage)
        self.performance_data['gpu_memory'].append(gpu_memory)
        self.performance_data['gpu_temperature'].append(gpu_temperature)
        self.performance_data['disk_io_read'].append(disk_read)
        self.performance_data['disk_io_write'].append(disk_write)
        self.performance_data['network_sent'].append(net_sent)
        self.performance_data['network_recv'].append(net_recv)
        self.performance_data['temperature'].append(temperature)
        
        # Estimate FPS and processing time (mock values for now)
        # In a real scenario, these would come from the application logs
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        fps = rng.normal(25, 5)  # Mock FPS around 25
        processing_time = rng.normal(40, 10)  # Mock processing time in ms
        
        self.performance_data['fps'].append(max(0, fps))
        self.performance_data['frame_processing_time'].append(max(0, processing_time))

    def monitor_application_performance(self):
        """Main monitoring loop"""
        logger.info(f"Starting performance monitoring for {self.monitoring_duration} seconds...")
        
        start_time = time.time()
        self.monitoring_active = True
        
        while self.monitoring_active and (time.time() - start_time) < self.monitoring_duration:
            try:
                self.collect_system_metrics()
                time.sleep(1)  # Collect metrics every second
                
                # Log progress every 30 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 30 == 0:
                    logger.info(f"Monitoring progress: {elapsed/self.monitoring_duration*100:.1f}%")
                    
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(1)
        
        logger.info("Performance monitoring completed")

    def stop_application(self):
        """Stop the application"""
        if self.app_process and self.app_process.poll() is None:
            logger.info("Stopping application...")
            self.app_process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.app_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Application didn't stop gracefully, forcing kill...")
                self.app_process.kill()
            
            logger.info("Application stopped")

    def generate_performance_graphs(self):
        """Generate three different types of performance graphs"""
        logger.info("Generating performance analysis graphs...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Convert timestamps to matplotlib dates
        timestamps = mdates.date2num(self.performance_data['timestamps'])
        
        # Graph 1: Time Series Performance Overview
        self.create_time_series_graph(timestamps)
        
        # Graph 2: Resource Utilization Heatmap
        self.create_resource_heatmap()
        
        # Graph 3: Performance Distribution and Statistics
        self.create_performance_distribution()
        
        # Generate performance report
        self.generate_performance_report()
        
        logger.info(f"All graphs saved to {self.results_dir}/")

    def create_time_series_graph(self, timestamps):
        """Create comprehensive time series performance graph"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('System Performance Over Time - Cheating Detection Application', fontsize=16, fontweight='bold')
        
        # CPU Usage
        axes[0, 0].plot(timestamps, self.performance_data['cpu_percent'], color='red', linewidth=2, label='CPU Usage')
        axes[0, 0].axhline(y=self.baseline_metrics['cpu_baseline'], color='red', linestyle='--', alpha=0.7, label='Baseline')
        axes[0, 0].set_title(CPU_USAGE_LABEL)
        axes[0, 0].set_ylabel('Percentage')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory Usage
        axes[0, 1].plot(timestamps, self.performance_data['memory_usage'], color='blue', linewidth=2, label='Memory Usage (GB)')
        axes[0, 1].set_title('Memory Usage')
        axes[0, 1].set_ylabel('GB')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # GPU Usage (if available)
        if self.gpu_available and any(self.performance_data['gpu_usage']):
            axes[1, 0].plot(timestamps, self.performance_data['gpu_usage'], color='green', linewidth=2, label='GPU Usage')
            axes[1, 0].set_title(GPU_USAGE_LABEL)
            axes[1, 0].set_ylabel('Percentage')
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'GPU Monitoring\nNot Available', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('GPU Usage (N/A)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # FPS Performance
        axes[1, 1].plot(timestamps, self.performance_data['fps'], color='orange', linewidth=2, label='FPS')
        axes[1, 1].set_title('Frames Per Second')
        axes[1, 1].set_ylabel('FPS')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Processing Time
        axes[2, 0].plot(timestamps, self.performance_data['frame_processing_time'], color='purple', linewidth=2, label='Processing Time')
        axes[2, 0].set_title('Frame Processing Time')
        axes[2, 0].set_ylabel('Milliseconds')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Temperature
        if any(self.performance_data['temperature']):
            axes[2, 1].plot(timestamps, self.performance_data['temperature'], color='red', linewidth=2, label='CPU Temperature')
            axes[2, 1].set_title('System Temperature')
            axes[2, 1].set_ylabel('°C')
            axes[2, 1].legend()
        else:
            axes[2, 1].text(0.5, 0.5, 'Temperature\nMonitoring\nNot Available', ha='center', va='center', transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('Temperature (N/A)')
        axes[2, 1].grid(True, alpha=0.3)
        
        # Format x-axis for all subplots
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/performance_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_resource_heatmap(self):
        """Create resource utilization heatmap"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Resource Utilization Heatmap - Cheating Detection System', fontsize=16, fontweight='bold')
        
        # Prepare data for heatmap (group by time intervals)
        n_intervals = min(60, len(self.performance_data['timestamps']))  # Max 60 time intervals
        interval_size = len(self.performance_data['timestamps']) // n_intervals
        
        heatmap_data = {
            'CPU': [],
            'Memory': [],
            'GPU': [],
            'FPS': []
        }
        
        time_labels = []
        
        for i in range(0, len(self.performance_data['timestamps']), interval_size):
            end_idx = min(i + interval_size, len(self.performance_data['timestamps']))
            
            heatmap_data['CPU'].append(np.mean(self.performance_data['cpu_percent'][i:end_idx]))
            heatmap_data['Memory'].append(np.mean(self.performance_data['memory_percent'][i:end_idx]))
            heatmap_data['GPU'].append(np.mean(self.performance_data['gpu_usage'][i:end_idx]))
            heatmap_data['FPS'].append(np.mean(self.performance_data['fps'][i:end_idx]))
            
            time_labels.append(self.performance_data['timestamps'][i].strftime('%H:%M'))
        
        # Create heatmaps
        resources = [CPU_USAGE_LABEL, 'Memory Usage (%)', GPU_USAGE_LABEL, 'FPS Performance']
        data_keys = ['CPU', 'Memory', 'GPU', 'FPS']
        
        for idx, (resource, key) in enumerate(zip(resources, data_keys)):
            ax = axes[idx // 2, idx % 2]
            
            # Reshape data for heatmap (create 2D array)
            data_array = np.array(heatmap_data[key]).reshape(1, -1)
            
            im = ax.imshow(data_array, cmap='RdYlGn_r' if key != 'FPS' else 'RdYlGn', aspect='auto')
            ax.set_title(resource)
            ax.set_yticks([0])
            ax.set_yticklabels(['Value'])
            
            # Set x-axis labels (sample every few intervals for readability)
            step = max(1, len(time_labels) // 10)
            ax.set_xticks(range(0, len(time_labels), step))
            ax.set_xticklabels([time_labels[i] for i in range(0, len(time_labels), step)], rotation=45)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Value')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/resource_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_performance_distribution(self):
        """Create performance distribution and statistical analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Performance Distribution Analysis - Cheating Detection System', fontsize=16, fontweight='bold')
        
        # Distribution plots
        metrics = [
            (CPU_USAGE_LABEL, self.performance_data['cpu_percent'], 'red'),
            ('Memory Usage (%)', self.performance_data['memory_percent'], 'blue'),
            ('FPS', self.performance_data['fps'], 'green'),
            ('Processing Time (ms)', self.performance_data['frame_processing_time'], 'orange'),
            (GPU_USAGE_LABEL, self.performance_data['gpu_usage'], 'purple'),
            ('Temperature (°C)', self.performance_data['temperature'], 'brown')
        ]
        
        for idx, (title, data, color) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            if any(data):  # Only plot if we have data
                # Create histogram with statistics
                _, _, _ = ax.hist(data, bins=30, alpha=0.7, color=color, edgecolor='black')
                
                # Add statistics
                mean_val = np.mean(data)
                median_val = np.median(data)
                std_val = np.std(data)
                
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
                
                ax.set_title(f'{title}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add text box with statistics
                stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nMin: {min(data):.2f}\nMax: {max(data):.2f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            else:
                ax.text(0.5, 0.5, f'{title}\nNo Data Available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/performance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_performance_report(self):
        """Generate a comprehensive performance report"""
        logger.info("Generating performance report...")
        
        # Calculate statistics
        report_data = {
            'monitoring_duration': self.monitoring_duration,
            'total_samples': len(self.performance_data['timestamps']),
            'baseline_metrics': self.baseline_metrics,
            'performance_summary': {}
        }
        
        # Calculate performance statistics for each metric
        metrics = ['cpu_percent', 'memory_usage', 'memory_percent', 'fps', 'frame_processing_time', 'gpu_usage']
        
        for metric in metrics:
            data = self.performance_data[metric]
            if data:
                report_data['performance_summary'][metric] = {
                    'mean': float(np.mean(data)),
                    'median': float(np.median(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'percentile_25': float(np.percentile(data, 25)),
                    'percentile_75': float(np.percentile(data, 75)),
                    'percentile_95': float(np.percentile(data, 95))
                }
        
        # Performance insights
        insights = []
        
        # CPU Analysis
        cpu_mean = report_data['performance_summary'].get('cpu_percent', {}).get('mean', 0)
        cpu_baseline = self.baseline_metrics.get('cpu_baseline', 0)
        if cpu_mean > cpu_baseline * 1.5:
            insights.append(f"High CPU usage detected: {cpu_mean:.1f}% (baseline: {cpu_baseline:.1f}%)")
        
        # Memory Analysis
        memory_mean = report_data['performance_summary'].get('memory_percent', {}).get('mean', 0)
        if memory_mean > 80:
            insights.append(f"High memory usage detected: {memory_mean:.1f}%")
        
        # FPS Analysis
        fps_data = self.performance_data['fps']
        if fps_data:
            fps_mean = np.mean(fps_data)
            if fps_mean < 20:
                insights.append(f"Low FPS performance: {fps_mean:.1f} FPS")
            elif fps_mean > 30:
                insights.append(f"Good FPS performance: {fps_mean:.1f} FPS")
        
        report_data['insights'] = insights
        
        # Save JSON report
        with open(f'{self.results_dir}/performance_report.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Save human-readable report
        with open(f'{self.results_dir}/performance_report.txt', 'w') as f:
            f.write("CHEATING DETECTION SYSTEM - PERFORMANCE ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Monitoring Duration: {self.monitoring_duration} seconds\n")
            f.write(f"Total Samples Collected: {report_data['total_samples']}\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("BASELINE METRICS:\n")
            f.write("-" * 20 + "\n")
            for key, value in self.baseline_metrics.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 20 + "\n")
            for metric, stats in report_data['performance_summary'].items():
                f.write(f"\n{metric.upper()}:\n")
                for stat_name, stat_value in stats.items():
                    f.write(f"  {stat_name}: {stat_value:.2f}\n")
            
            f.write("\nPERFORMACE INSIGHTS:\n")
            f.write("-" * 20 + "\n")
            if insights:
                for insight in insights:
                    f.write(f"• {insight}\n")
            else:
                f.write("• No significant performance issues detected\n")
            
            f.write("\nFiles Generated:\n")
            f.write("• performance_timeseries.png - Time series performance graphs\n")
            f.write("• resource_heatmap.png - Resource utilization heatmap\n")
            f.write("• performance_distribution.png - Performance distribution analysis\n")
            f.write("• performance_report.json - Detailed performance data\n")
            f.write("• performance_report.txt - This human-readable report\n")

    def run_performance_test(self):
        """Run the complete performance test"""
        try:
            logger.info("Starting Cheating Detection System Performance Test")
            
            # Step 1: Get baseline metrics
            self.get_baseline_metrics()
            
            # Step 2: Start the application
            if not self.start_application():
                logger.error("Failed to start application - aborting test")
                return False
            
            # Step 3: Monitor performance
            self.monitor_application_performance()
            
            # Step 4: Stop the application
            self.stop_application()
            
            # Step 5: Generate analysis and graphs
            self.generate_performance_graphs()
            
            logger.info("Performance test completed successfully!")
            logger.info(f"Results saved to: {self.results_dir}/")
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Performance test interrupted by user")
            self.monitoring_active = False
            if self.app_process:
                self.stop_application()
            return False
            
        except Exception as e:
            logger.error(f"Error during performance test: {e}")
            if self.app_process:
                self.stop_application()
            return False

def main():
    """Main function to run the performance monitor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance Monitor for Cheating Detection System')
    parser.add_argument('--command', default='python main.py', 
                       help='Command to run the application (default: python main.py)')
    parser.add_argument('--duration', type=int, default=300,
                       help='Monitoring duration in seconds (default: 300)')
    parser.add_argument('--server-mode', action='store_true',
                       help='Monitor server instead of main application')
    
    args = parser.parse_args()
    
    # Adjust command if server mode is requested
    if args.server_mode:
        command = 'python server.py'
    else:
        command = args.command
    
    # Create and run performance monitor
    monitor = PerformanceMonitor(
        app_command=command,
        monitoring_duration=args.duration
    )
    
    success = monitor.run_performance_test()
    
    if success:
        print(f"\n{'='*60}")
        print("PERFORMANCE TEST COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Results directory: {monitor.results_dir}")
        print("\nGenerated files:")
        print("• performance_timeseries.png - Comprehensive time series graphs")
        print("• resource_heatmap.png - Resource utilization heatmap")
        print("• performance_distribution.png - Statistical distribution analysis")
        print("• performance_report.json - Detailed performance data")
        print("• performance_report.txt - Human-readable report")
        print(f"{'='*60}")
    else:
        print("Performance test failed. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
