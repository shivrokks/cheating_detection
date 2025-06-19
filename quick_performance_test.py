#!/usr/bin/env python3
"""
Quick Performance Test Script
============================
A simplified script to run performance tests with different configurations.
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime

def run_performance_test(test_type, duration=None):
    """Run a performance test with specified parameters."""
    
    print(f"🚀 Starting {test_type} performance test...")
    print("=" * 50)
    
    if test_type == "quick":
        # Quick 2-minute test
        duration = duration or 120
        cmd = f"python performance_monitor.py --duration {duration}"
        
    elif test_type == "standard":
        # Standard 5-minute test with comprehensive monitoring
        duration = duration or 300
        cmd = f"python performance_monitor.py --duration {duration} --interval 2"
        
    elif test_type == "extended":
        # Extended test using the run_performance_tests.py script
        cmd = "python run_performance_tests.py extended"
        
    elif test_type == "server":
        # Server performance test
        cmd = "python run_performance_tests.py server"
        
    elif test_type == "basic":
        # Basic test
        cmd = "python run_performance_tests.py basic"
        
    else:
        print(f"❌ Unknown test type: {test_type}")
        print("Available types: quick, standard, extended, server, basic")
        return False
    
    try:
        print(f"Executing: {cmd}")
        subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print("\n✅ Performance test completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Performance test failed with error code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return False

def list_recent_results():
    """List recent performance test results."""
    print("\n📊 Recent Performance Test Results:")
    print("=" * 40)
    
    results_found = False
    for item in sorted(os.listdir('.'), reverse=True):
        if item.startswith('performance_results_') or item.startswith('test_performance_results_'):
            print(f"• {item}")
            results_found = True
    
    if not results_found:
        print("No performance test results found.")

def main():
    parser = argparse.ArgumentParser(
        description="Quick Performance Test Script for Cheating Detection System"
    )
    
    parser.add_argument(
        'test_type',
        choices=['quick', 'standard', 'extended', 'server', 'basic'],
        help='Type of performance test to run'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        help='Test duration in seconds (for quick/standard tests only)'
    )
    
    parser.add_argument(
        '--list-results',
        action='store_true',
        help='List recent test results'
    )
    
    args = parser.parse_args()
    
    if args.list_results:
        list_recent_results()
        return
    
    print("Cheating Detection System - Performance Test")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = run_performance_test(args.test_type, args.duration)
    
    if success:
        print("\n📈 Performance graphs generated:")
        print("• Time Series Graph - Shows metrics over time")
        print("• Distribution Graph - Shows statistical distributions")
        print("• Resource Heatmap - Shows resource usage patterns")
        
        list_recent_results()
        
        print(f"\n🎯 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("\n❌ Test failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
