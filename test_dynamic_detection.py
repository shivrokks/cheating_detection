#!/usr/bin/env python3
"""
Test script to verify dynamic detection behavior with real webcam
"""

import cv2
import numpy as np
import time
import sys

def test_dynamic_detection():
    """Test dynamic detection with real webcam"""
    print("Testing dynamic detection with webcam...")
    print("This will test for 30 seconds. Move your eyes, head, and change expressions!")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No webcam available")
            return False
        
        # Import detection modules
        from eye_movement import process_eye_movement
        from facial_expression import process_facial_expression
        from head_pose import process_head_pose
        
        # Track results to verify they change
        gaze_results = set()
        expression_results = set()
        head_results = set()
        
        calibrated_angles = None
        frame_count = 0
        start_time = time.time()
        
        print("\n🎯 Instructions:")
        print("- Look left, right, up, down to test gaze detection")
        print("- Smile and raise eyebrows to test expression detection")
        print("- Move your head to test head pose detection")
        print("- Press 'q' to quit early\n")
        
        while time.time() - start_time < 30:  # Test for 30 seconds
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Test eye movement
            try:
                _, gaze_direction = process_eye_movement(frame.copy())
                gaze_results.add(gaze_direction)
                print(f"Frame {frame_count}: Gaze = {gaze_direction}")
            except Exception as e:
                print(f"Eye movement error: {e}")
            
            # Test facial expression
            try:
                _, expression = process_facial_expression(frame.copy())
                expression_results.add(expression)
                print(f"Frame {frame_count}: Expression = {expression}")
            except Exception as e:
                print(f"Facial expression error: {e}")
            
            # Test head pose
            try:
                if calibrated_angles is None:
                    _, result = process_head_pose(frame.copy(), None)
                    if isinstance(result, tuple) and len(result) == 3:
                        calibrated_angles = result
                        print(f"Frame {frame_count}: Calibrated angles = {result}")
                    else:
                        head_results.add(result)
                        print(f"Frame {frame_count}: Head = {result}")
                else:
                    _, head_direction = process_head_pose(frame.copy(), calibrated_angles)
                    head_results.add(head_direction)
                    print(f"Frame {frame_count}: Head = {head_direction}")
            except Exception as e:
                print(f"Head pose error: {e}")
            
            # Display frame
            cv2.imshow('Dynamic Detection Test', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Small delay
            time.sleep(0.1)
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Analyze results
        print(f"\n📊 Results after {frame_count} frames:")
        print(f"Gaze variations: {len(gaze_results)} - {gaze_results}")
        print(f"Expression variations: {len(expression_results)} - {expression_results}")
        print(f"Head pose variations: {len(head_results)} - {head_results}")
        
        # Check if we got dynamic results
        dynamic_gaze = len(gaze_results) > 1
        dynamic_expression = len(expression_results) > 1
        dynamic_head = len(head_results) > 1
        
        print(f"\n✅ Dynamic Detection Results:")
        print(f"Gaze Detection: {'✅ DYNAMIC' if dynamic_gaze else '❌ STATIC'}")
        print(f"Expression Detection: {'✅ DYNAMIC' if dynamic_expression else '❌ STATIC'}")
        print(f"Head Pose Detection: {'✅ DYNAMIC' if dynamic_head else '❌ STATIC'}")
        
        success = dynamic_gaze and dynamic_expression and dynamic_head
        
        if success:
            print("\n🎉 All detections are working dynamically!")
        else:
            print("\n⚠️  Some detections are still static. Check the debug output above.")
        
        return success
        
    except Exception as e:
        print(f"❌ Dynamic detection test failed: {e}")
        return False

def test_server_integration():
    """Test with the actual server processing"""
    print("\nTesting server integration...")
    
    try:
        from server import process_frame_detection
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No webcam available")
            return False
        
        results_gaze = set()
        results_expression = set()
        
        for i in range(10):  # Test 10 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                _, results = process_frame_detection(frame, False, 0.0)
                
                if 'gaze_direction' in results:
                    results_gaze.add(results['gaze_direction'])
                if 'facial_expression' in results:
                    results_expression.add(results['facial_expression'])
                
                print(f"Server Frame {i+1}: Gaze={results.get('gaze_direction', 'N/A')}, Expression={results.get('facial_expression', 'N/A')}")
                
            except Exception as e:
                print(f"Server processing error: {e}")
            
            time.sleep(0.5)  # Half second between frames
        
        cap.release()
        
        print(f"\nServer Results:")
        print(f"Gaze variations: {len(results_gaze)} - {results_gaze}")
        print(f"Expression variations: {len(results_expression)} - {results_expression}")
        
        success = len(results_gaze) > 1 or len(results_expression) > 1
        
        if success:
            print("✅ Server integration working dynamically!")
        else:
            print("❌ Server integration still showing static results")
        
        return success
        
    except Exception as e:
        print(f"❌ Server integration test failed: {e}")
        return False

def main():
    """Run dynamic detection tests"""
    print("="*60)
    print("🔧 Testing Dynamic Detection Behavior")
    print("="*60)
    
    # Test individual modules
    dynamic_test_passed = test_dynamic_detection()
    
    # Test server integration
    server_test_passed = test_server_integration()
    
    print("\n" + "="*60)
    print("📊 Final Results:")
    print(f"Dynamic Detection Test: {'✅ PASSED' if dynamic_test_passed else '❌ FAILED'}")
    print(f"Server Integration Test: {'✅ PASSED' if server_test_passed else '❌ FAILED'}")
    
    if dynamic_test_passed and server_test_passed:
        print("\n🎉 All dynamic detection tests passed!")
        print("Your app should now see varying gaze and expression results.")
    else:
        print("\n⚠️  Some tests failed. Check the debug output above.")
    
    print("="*60)
    
    return dynamic_test_passed and server_test_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
