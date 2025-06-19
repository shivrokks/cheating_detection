# Performance Monitoring System - Complete Guide

## Overview
The performance monitoring system provides comprehensive analysis of your cheating detection application's performance while it runs. It measures various metrics and generates three different types of graphs to visualize performance data.

## 🎯 Features

### Performance Metrics Monitored
- **System Resources**
  - CPU Usage (%)
  - Memory Usage (GB and %)
  - Available Memory
  - System Load
  
- **Application Performance**
  - Frame Rate (FPS)
  - Frame Processing Time (ms)
  - Detection Accuracy
  - Response Time
  
- **GPU Metrics** (if available)
  - GPU Usage (%)
  - GPU Memory Usage
  - GPU Temperature

### 📊 Three Types of Performance Graphs Generated

1. **Time Series Graph** (`performance_timeseries.png`)
   - Shows how metrics change over time
   - Multiple subplots for different metric categories
   - Helps identify performance trends and bottlenecks
   - Real-time progression visualization

2. **Distribution Graph** (`performance_distribution.png`)
   - Statistical distribution of performance metrics
   - Histograms showing frequency distributions
   - Box plots for quartile analysis
   - Helps understand typical performance ranges

3. **Resource Heatmap** (`resource_heatmap.png`)
   - Color-coded visualization of resource usage
   - Shows correlation between different metrics
   - Identifies peak usage periods
   - Easy-to-read intensity mapping

## 🚀 Usage Options

### 1. Quick Performance Test Script
```bash
# Quick 2-minute test
python quick_performance_test.py quick

# Quick test with custom duration
python quick_performance_test.py quick --duration 60

# Standard 5-minute comprehensive test
python quick_performance_test.py standard

# Extended 10-minute test with detailed analysis
python quick_performance_test.py extended

# Server performance test
python quick_performance_test.py server

# Basic functionality test
python quick_performance_test.py basic

# List recent test results
python quick_performance_test.py --list-results
```

### 2. Direct Performance Monitor
```bash
# Basic monitoring for 120 seconds
python performance_monitor.py --duration 120

# With custom sampling interval
python performance_monitor.py --duration 300 --interval 2

# Help for all options
python performance_monitor.py --help
```

### 3. Predefined Test Suites
```bash
# Basic test (2 minutes)
python run_performance_tests.py basic

# Server test (5 minutes)
python run_performance_tests.py server

# Extended test (10 minutes)
python run_performance_tests.py extended
```

## 📁 Output Files

Each performance test creates a timestamped directory containing:

- **`performance_timeseries.png`** - Time series visualization
- **`resource_heatmap.png`** - Resource usage heatmap
- **`performance_distribution.png`** - Statistical distributions
- **`performance_report.json`** - Raw performance data
- **`performance_report.txt`** - Human-readable summary report

## 📈 Performance Insights

The system automatically generates insights such as:
- High CPU usage alerts
- Memory usage warnings
- Performance bottlenecks identification
- Baseline comparison analysis
- Statistical performance summaries

## 🔧 System Requirements

### Required Dependencies
- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Seaborn
- psutil
- Pandas

### Optional Dependencies
- GPUtil (for GPU monitoring)
- Jupyter (for notebook analysis)

## 📊 Sample Performance Report

```
PERFORMANCE SUMMARY:
--------------------
CPU_PERCENT:
  mean: 12.96%    median: 13.50%
  std: 4.53%      min: 3.00%      max: 26.40%

MEMORY_USAGE:
  mean: 7.11 GB   median: 7.17 GB
  std: 0.25 GB    min: 5.95 GB    max: 7.38 GB

FPS:
  mean: 26.52     median: 26.52
  std: 0.00       min: 26.52     max: 26.52

PERFORMANCE INSIGHTS:
• High CPU usage detected: 13.0% (baseline: 5.3%)
```

## 🎨 Graph Examples

### Time Series Graph Features
- Multiple metric subplots
- Time-based x-axis
- Color-coded metric lines
- Performance trend visualization
- Grid lines for easy reading

### Distribution Graph Features
- Histogram distributions
- Statistical summaries
- Box plots with quartiles
- Outlier identification
- Performance range analysis

### Resource Heatmap Features
- Color intensity mapping
- Resource correlation matrix
- Time-based heat patterns
- Easy bottleneck identification
- Visual performance peaks

## 🔍 Troubleshooting

### Common Issues
1. **GPU monitoring disabled**: Install GPUtil (`pip install GPUtil`)
2. **Application won't start**: Check if main.py is accessible
3. **Graph generation warnings**: Normal matplotlib warnings, graphs still generated
4. **Permission errors**: Ensure write permissions in the directory

### Performance Tips
- Run tests when system is relatively idle
- Close unnecessary applications during testing
- Use longer test durations for better statistical analysis
- Monitor system temperature during extended tests

## 📝 Validation

Use the validation script to test the system:
```bash
python validate_performance_monitor.py
```

This will verify:
- All dependencies are installed
- System monitoring works correctly
- Graph generation functions properly
- File output is created successfully

## 🎯 Best Practices

1. **Test Duration Selection**
   - Quick tests (30-120s): For rapid feedback
   - Standard tests (5min): For typical performance analysis
   - Extended tests (10min+): For comprehensive analysis

2. **Interpretation Guidelines**
   - Compare against baseline metrics
   - Look for performance trends over time
   - Identify resource usage spikes
   - Analyze statistical distributions

3. **Regular Monitoring**
   - Run weekly performance tests
   - Compare results over time
   - Monitor for performance degradation
   - Optimize based on bottlenecks identified

---

*This performance monitoring system provides comprehensive insights into your cheating detection application's behavior, helping you optimize performance and identify potential issues before they impact users.*
