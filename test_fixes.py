#!/usr/bin/env python3
"""
Test script to verify the fixes for the cheating detection system
"""

import cv2
import numpy as np
import time
import sys
import os

def test_face_detection():
    """Test face detection with the fixed OpenCV cascade loading"""
    print("Testing face detection...")
    
    try:
        from facial_expression import detect_faces_improved
        
        # Create a test frame (black image)
        test_frame = cv2.imread('test_image.jpg') if os.path.exists('test_image.jpg') else None
        
        if test_frame is None:
            # Create a simple test frame
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            test_frame = cv2.rectangle(test_frame, (200, 150), (440, 330), (255, 255, 255), -1)
            print("Using synthetic test frame")
        
        gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces_improved(gray)
        
        print(f"✓ Face detection working - detected {len(faces)} faces")
        return True
        
    except Exception as e:
        print(f"✗ Face detection failed: {e}")
        return False

def test_mobile_detection():
    """Test mobile detection with lowered threshold"""
    print("Testing mobile detection...")
    
    try:
        from mobile_detection import process_mobile_detection
        
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        _, mobile_detected = process_mobile_detection(test_frame)
        
        print(f"✓ Mobile detection working - detected: {mobile_detected}")
        return True
        
    except Exception as e:
        print(f"✗ Mobile detection failed: {e}")
        return False

def test_eye_movement():
    """Test eye movement detection"""
    print("Testing eye movement detection...")
    
    try:
        from eye_movement import process_eye_movement
        
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        _, gaze_direction = process_eye_movement(test_frame)
        
        print(f"✓ Eye movement detection working - direction: {gaze_direction}")
        return True
        
    except Exception as e:
        print(f"✗ Eye movement detection failed: {e}")
        return False

def test_head_pose():
    """Test head pose detection"""
    print("Testing head pose detection...")
    
    try:
        from head_pose import process_head_pose
        
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test calibration mode
        _, calibration_result = process_head_pose(test_frame, None)
        print(f"✓ Head pose calibration working - result: {calibration_result}")
        
        # Test detection mode (if calibration worked)
        if calibration_result is not None and isinstance(calibration_result, tuple):
            _, head_direction = process_head_pose(test_frame, calibration_result)
            print(f"✓ Head pose detection working - direction: {head_direction}")
        
        return True
        
    except Exception as e:
        print(f"✗ Head pose detection failed: {e}")
        return False

def test_lip_movement():
    """Test lip movement detection"""
    print("Testing lip movement detection...")
    
    try:
        from lip_movement import process_lip_movement
        
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        _, is_talking = process_lip_movement(test_frame)
        
        print(f"✓ Lip movement detection working - talking: {is_talking}")
        return True
        
    except Exception as e:
        print(f"✗ Lip movement detection failed: {e}")
        return False

def test_webcam_integration():
    """Test with actual webcam if available"""
    print("Testing with webcam...")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️  No webcam available, skipping webcam test")
            return True
        
        # Read a few frames
        for i in range(5):
            ret, frame = cap.read()
            if not ret:
                break
            
            print(f"Frame {i+1}: {frame.shape}")
            
            # Test one detection on the last frame
            if i == 4:
                from facial_expression import detect_faces_improved
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detect_faces_improved(gray)
                print(f"Real webcam faces detected: {len(faces)}")
        
        cap.release()
        print("✓ Webcam integration working")
        return True
        
    except Exception as e:
        print(f"✗ Webcam integration failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("🔧 Testing Cheating Detection System Fixes")
    print("="*60)
    
    tests = [
        test_face_detection,
        test_mobile_detection,
        test_eye_movement,
        test_head_pose,
        test_lip_movement,
        test_webcam_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            print()
    
    print("="*60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fixes should work.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
