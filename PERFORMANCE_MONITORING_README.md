# Performance Monitoring System for Cheating Detection Application

This performance monitoring system provides comprehensive analysis of your cheating detection application's performance, generating detailed reports and three different types of visualizations.

## Features

### 📊 Performance Metrics Tracked
- **System Resources**: CPU usage, Memory usage, Disk I/O, Network I/O
- **GPU Metrics**: GPU utilization and memory (if available)
- **Application Performance**: FPS, Frame processing time, Detection latency
- **System Health**: Temperature monitoring, Power usage estimation
- **Detection-Specific Metrics**: Processing time for each detection module

### 📈 Three Types of Graphs Generated

1. **Time Series Performance Graph** (`performance_timeseries.png`)
   - Real-time performance metrics over time
   - CPU, Memory, GPU usage trends
   - FPS and processing time analysis
   - Temperature monitoring

2. **Resource Utilization Heatmap** (`resource_heatmap.png`)
   - Visual representation of resource usage patterns
   - Identifies performance bottlenecks
   - Time-based resource consumption analysis

3. **Performance Distribution Analysis** (`performance_distribution.png`)
   - Statistical distribution of performance metrics
   - Histograms with mean, median, and standard deviation
   - Performance outlier identification

## Installation

### Required Dependencies

```bash
# Install additional performance monitoring dependencies
pip install GPUtil flask flask-cors flask-socketio pyngrok

# Verify existing dependencies
pip install -r requirements.txt
```

### Optional Dependencies

```bash
# For GPU monitoring (NVIDIA GPUs only)
pip install GPUtil

# For advanced analysis
pip install jupyter notebook
```

## Usage

### Quick Start

```bash
# Run basic performance test (2 minutes)
python run_performance_tests.py basic

# Run server performance test (3 minutes)
python run_performance_tests.py server

# Run extended analysis (10 minutes)
python run_performance_tests.py extended
```

### Advanced Usage

```bash
# Custom monitoring duration (5 minutes)
python performance_monitor.py --duration 300

# Monitor server component
python performance_monitor.py --server-mode --duration 180

# Custom application command
python performance_monitor.py --command "python main.py --train" --duration 240
```

### Validation and Testing

```bash
# Test the performance monitoring system
python validate_performance_monitor.py

# This will:
# - Check all dependencies
# - Test system monitoring
# - Generate sample graphs
# - Validate functionality
```

## Output Files

Each performance test creates a timestamped directory with the following files:

### 📁 Results Directory Structure
```
performance_results_YYYYMMDD_HHMMSS/
├── performance_timeseries.png      # Time series graphs
├── resource_heatmap.png             # Resource utilization heatmap  
├── performance_distribution.png     # Statistical distribution analysis
├── performance_report.json          # Detailed performance data
└── performance_report.txt           # Human-readable report
```

### 📊 Graph Details

#### 1. Time Series Performance Graph
- **CPU Usage**: Real-time CPU utilization with baseline comparison
- **Memory Usage**: Memory consumption in GB and percentage
- **GPU Usage**: GPU utilization (if available)
- **FPS Performance**: Frames per second over time
- **Processing Time**: Frame processing latency
- **Temperature**: System temperature monitoring

#### 2. Resource Utilization Heatmap
- Visual representation of resource usage patterns
- Time-based analysis of CPU, Memory, GPU, and FPS
- Color-coded intensity maps for easy identification of bottlenecks

#### 3. Performance Distribution Analysis
- Statistical histograms for each metric
- Mean, median, standard deviation indicators
- Min/max values and percentiles
- Performance outlier identification

## Performance Metrics Explained

### System Metrics
- **CPU Usage (%)**: Processor utilization percentage
- **Memory Usage (GB/%)**: RAM consumption
- **Disk I/O (MB/s)**: Read/write speeds
- **Network I/O (MB/s)**: Network traffic
- **Temperature (°C)**: System temperature (if available)

### Application Metrics
- **FPS**: Frames processed per second
- **Processing Time (ms)**: Time to process each frame
- **Detection Latency (ms)**: Time for detection algorithms

### Detection Module Metrics
- **Eye Movement Time**: Time for eye tracking processing
- **Head Pose Time**: Time for head pose estimation
- **Mobile Detection Time**: Time for mobile device detection
- **Facial Expression Time**: Time for expression analysis
- **Person Detection Time**: Time for person counting
- **Object Detection Time**: Time for suspicious object detection
- **Behavior Analysis Time**: Time for behavior classification

## Interpreting Results

### Performance Thresholds

#### 🟢 Good Performance
- CPU Usage: < 70%
- Memory Usage: < 80%
- FPS: > 25
- Processing Time: < 50ms

#### 🟡 Acceptable Performance
- CPU Usage: 70-85%
- Memory Usage: 80-90%
- FPS: 20-25
- Processing Time: 50-80ms

#### 🔴 Poor Performance
- CPU Usage: > 85%
- Memory Usage: > 90%
- FPS: < 20
- Processing Time: > 80ms

### Common Performance Issues

1. **High CPU Usage**
   - **Cause**: Intensive processing algorithms
   - **Solution**: Optimize detection algorithms, reduce frame rate

2. **High Memory Usage**
   - **Cause**: Memory leaks, large model sizes
   - **Solution**: Implement memory management, use smaller models

3. **Low FPS**
   - **Cause**: Slow processing, hardware limitations
   - **Solution**: Hardware upgrade, algorithm optimization

4. **High Processing Time**
   - **Cause**: Complex detection algorithms
   - **Solution**: Parallel processing, GPU acceleration

## Configuration Options

### Command Line Arguments

```bash
python performance_monitor.py [OPTIONS]

Options:
  --command TEXT     Command to run the application (default: python main.py)
  --duration INTEGER Monitoring duration in seconds (default: 300)
  --server-mode      Monitor server instead of main application
  --help             Show help message
```

### Environment Variables

```bash
# Optional: Set custom paths
export PERFORMANCE_RESULTS_DIR="/path/to/results"
export PERFORMANCE_LOG_LEVEL="DEBUG"
```

## Troubleshooting

### Common Issues

1. **GPUtil Import Error**
   ```bash
   # GPU monitoring not available
   pip install GPUtil
   ```

2. **Permission Denied**
   ```bash
   # Make scripts executable
   chmod +x performance_monitor.py
   chmod +x run_performance_tests.py
   ```

3. **Application Won't Start**
   ```bash
   # Check if camera is available
   python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
   ```

4. **Memory Issues**
   ```bash
   # Reduce monitoring duration
   python performance_monitor.py --duration 60
   ```

### Performance Optimization Tips

1. **Hardware Optimization**
   - Use dedicated GPU for processing
   - Ensure adequate RAM (8GB+ recommended)
   - Use SSD for faster I/O

2. **Software Optimization**
   - Close unnecessary applications
   - Use virtual environments
   - Monitor system resources

3. **Detection Optimization**
   - Reduce frame resolution
   - Skip frames for processing
   - Use lighter detection models

## Examples

### Basic Performance Test
```bash
# Run a quick 2-minute test
python run_performance_tests.py basic
```

### Extended Analysis
```bash
# Run comprehensive 10-minute analysis
python run_performance_tests.py extended
```

### Custom Test
```bash
# Monitor specific component for 5 minutes
python performance_monitor.py --command "python main.py --train" --duration 300
```

## Integration with Existing Code

### Adding Performance Hooks

You can integrate performance monitoring directly into your application:

```python
# In your main detection loop
import time
import psutil

def monitor_detection_performance():
    start_time = time.time()
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    
    # Your detection code here
    
    processing_time = time.time() - start_time
    
    # Log performance metrics
    print(f"Processing time: {processing_time*1000:.2f}ms")
    print(f"CPU: {cpu_usage}%, Memory: {memory_usage}%")
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Run the validation script: `python validate_performance_monitor.py`
3. Review the generated log files
4. Check system resources and dependencies

---

**Note**: This performance monitoring system is designed to work with the cheating detection application. Ensure all dependencies are installed and the application runs correctly before using the performance monitor.
