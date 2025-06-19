#!/usr/bin/env python3
"""
Performance Monitor Usage Examples
This script demonstrates different ways to use the performance monitor
"""

import os
import sys
from performance_monitor import PerformanceMonitor

def run_basic_test():
    """Run a basic 2-minute performance test"""
    print("Running basic performance test (2 minutes)...")
    
    monitor = PerformanceMonitor(
        app_command="python main.py",
        monitoring_duration=120  # 2 minutes
    )
    
    success = monitor.run_performance_test()
    
    if success:
        print(f"Test completed! Results in: {monitor.results_dir}")
    else:
        print("Test failed!")

def run_server_test():
    """Run performance test on the server component"""
    print("Running server performance test (3 minutes)...")
    
    monitor = PerformanceMonitor(
        app_command="python server.py",
        monitoring_duration=180  # 3 minutes
    )
    
    success = monitor.run_performance_test()
    
    if success:
        print(f"Server test completed! Results in: {monitor.results_dir}")
    else:
        print("Server test failed!")

def run_extended_test():
    """Run an extended 10-minute performance test"""
    print("Running extended performance test (10 minutes)...")
    
    monitor = PerformanceMonitor(
        app_command="python main.py",
        monitoring_duration=600  # 10 minutes
    )
    
    success = monitor.run_performance_test()
    
    if success:
        print(f"Extended test completed! Results in: {monitor.results_dir}")
        
        # Print some quick stats
        print("\nQuick Performance Summary:")
        print("=" * 40)
        
        import json
        report_file = os.path.join(monitor.results_dir, "performance_report.json")
        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                data = json.load(f)
                
            summary = data.get('performance_summary', {})
            
            if 'cpu_percent' in summary:
                cpu_stats = summary['cpu_percent']
                print(f"CPU Usage - Mean: {cpu_stats['mean']:.1f}%, Max: {cpu_stats['max']:.1f}%")
            
            if 'memory_percent' in summary:
                mem_stats = summary['memory_percent']
                print(f"Memory Usage - Mean: {mem_stats['mean']:.1f}%, Max: {mem_stats['max']:.1f}%")
                
            if 'fps' in summary:
                fps_stats = summary['fps']
                print(f"FPS - Mean: {fps_stats['mean']:.1f}, Min: {fps_stats['min']:.1f}")
                
            insights = data.get('insights', [])
            if insights:
                print("\nPerformance Insights:")
                for insight in insights:
                    print(f"• {insight}")
            else:
                print("\n• No significant performance issues detected")
    else:
        print("Extended test failed!")

def main():
    if len(sys.argv) < 2:
        print("Performance Monitor Usage Examples")
        print("=" * 40)
        print("Usage: python run_performance_tests.py [test_type]")
        print("\nAvailable test types:")
        print("  basic     - Basic 2-minute test")
        print("  server    - Server performance test (3 minutes)")
        print("  extended  - Extended 10-minute test with detailed analysis")
        print("\nExamples:")
        print("  python run_performance_tests.py basic")
        print("  python run_performance_tests.py server")
        print("  python run_performance_tests.py extended")
        return
    
    test_type = sys.argv[1].lower()
    
    if test_type == "basic":
        run_basic_test()
    elif test_type == "server":
        run_server_test()
    elif test_type == "extended":
        run_extended_test()
    else:
        print(f"Unknown test type: {test_type}")
        print("Available types: basic, server, extended")

if __name__ == "__main__":
    main()
