#!/usr/bin/env python3
"""
Test script for Cheating Detection Server
Tests the server endpoints and functionality
"""

import requests
import base64
import cv2
import json
import time
import sys

def test_health_endpoint(base_url):
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Calibrating: {data.get('calibrating')}")
            print(f"   Audio initialized: {data.get('audioInitialized')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_process_frame_endpoint(base_url):
    """Test the process frame endpoint with a sample image"""
    print("\n🖼️ Testing process frame endpoint...")
    try:
        # Create a simple test image
        test_image = cv2.imread("test_image.jpg")
        if test_image is None:
            # Create a simple colored rectangle if no test image exists
            test_image = cv2.rectangle(
                cv2.zeros((480, 640, 3), dtype=cv2.uint8),
                (100, 100), (540, 380), (0, 255, 0), -1
            )
            cv2.putText(test_image, "TEST IMAGE", (200, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', test_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare request
        payload = {
            "image": f"data:image/jpeg;base64,{img_base64}",
            "return_processed_image": False
        }
        
        # Send request
        response = requests.post(
            f"{base_url}/process_frame", 
            json=payload, 
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Frame processing successful")
                results = data.get('results', {})
                print(f"   Calibrating: {results.get('calibrating', 'Unknown')}")
                print(f"   Behavior: {results.get('behavior', 'Unknown')}")
                print(f"   Confidence: {results.get('behavior_confidence', 0):.2f}")
                print(f"   Overall suspicious: {results.get('overall_suspicious', False)}")
                return True
            else:
                print(f"❌ Frame processing failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Frame processing request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Frame processing error: {e}")
        return False

def test_status_endpoint(base_url):
    """Test the status endpoint"""
    print("\n📊 Testing status endpoint...")
    try:
        response = requests.get(f"{base_url}/get_status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Status check successful")
                status = data.get('status', {})
                if status:
                    print(f"   Last detection timestamp: {status.get('timestamp', 'None')}")
                    print(f"   Calibrating: {status.get('calibrating', 'Unknown')}")
                else:
                    print("   No previous detection results")
                return True
            else:
                print(f"❌ Status check failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Status request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Status check error: {e}")
        return False

def test_websocket_connection(base_url):
    """Test WebSocket connection (basic connectivity)"""
    print("\n🔌 Testing WebSocket connectivity...")
    try:
        # Convert HTTP URL to WebSocket URL
        ws_url = base_url.replace('http', 'ws')
        
        # Try to import socketio for testing
        try:
            import socketio
            
            sio = socketio.Client()
            connected = False
            
            @sio.event
            def connect():
                nonlocal connected
                connected = True
                print("✅ WebSocket connection successful")
            
            @sio.event
            def disconnect():
                print("🔌 WebSocket disconnected")
            
            # Try to connect
            sio.connect(ws_url, wait_timeout=10)
            
            if connected:
                sio.disconnect()
                return True
            else:
                print("❌ WebSocket connection failed")
                return False
                
        except ImportError:
            print("⚠️  socketio not available, skipping WebSocket test")
            print("   Install with: pip install python-socketio")
            return True  # Don't fail the test if socketio is not available
            
    except Exception as e:
        print(f"❌ WebSocket test error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Cheating Detection Server Test Suite")
    print("=" * 50)
    
    # Get server URL
    if len(sys.argv) > 1:
        base_url = sys.argv[1].rstrip('/')
    else:
        base_url = input("Enter server URL (e.g., https://abc123.ngrok.io): ").strip().rstrip('/')
    
    if not base_url:
        print("❌ No server URL provided")
        return
    
    print(f"🎯 Testing server: {base_url}")
    print("=" * 50)
    
    # Run tests
    tests = [
        ("Health Check", lambda: test_health_endpoint(base_url)),
        ("Process Frame", lambda: test_process_frame_endpoint(base_url)),
        ("Status Check", lambda: test_status_endpoint(base_url)),
        ("WebSocket", lambda: test_websocket_connection(base_url))
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
            time.sleep(1)  # Brief pause between tests
        except KeyboardInterrupt:
            print("\n\n⏹️ Tests interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error in {test_name}: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("=" * 50)
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! Server is ready for Android app.")
    elif passed > 0:
        print("⚠️  Some tests failed. Check server configuration.")
    else:
        print("❌ All tests failed. Server may not be running or accessible.")
    
    print("\n💡 Next steps:")
    print("1. If tests passed, use this URL in your Android app")
    print("2. If tests failed, check server logs and ngrok configuration")
    print("3. Ensure all dependencies are installed")

if __name__ == "__main__":
    main()
