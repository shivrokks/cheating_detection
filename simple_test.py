#!/usr/bin/env python3
"""
Simple test to check detection functions
"""

import cv2
import numpy as np

def test_simple():
    """Test detection functions with a simple image"""
    
    # Create a simple test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (100, 100, 100)  # Gray background
    
    print("Testing detection functions with test image...")
    
    try:
        from eye_movement import process_eye_movement
        result_frame, gaze_direction = process_eye_movement(test_image.copy())
        print(f"✓ Eye movement: {gaze_direction}")
    except Exception as e:
        print(f"✗ Eye movement error: {e}")
    
    try:
        from head_pose import process_head_pose
        result_frame, head_direction = process_head_pose(test_image.copy(), None)
        print(f"✓ Head pose: {head_direction}")
    except Exception as e:
        print(f"✗ Head pose error: {e}")
    
    try:
        from mobile_detection import process_mobile_detection
        result_frame, mobile_detected = process_mobile_detection(test_image.copy())
        print(f"✓ Mobile detection: {mobile_detected}")
    except Exception as e:
        print(f"✗ Mobile detection error: {e}")
    
    try:
        from lip_movement import process_lip_movement
        result_frame, is_talking = process_lip_movement(test_image.copy())
        print(f"✓ Lip movement: {is_talking}")
    except Exception as e:
        print(f"✗ Lip movement error: {e}")
    
    try:
        from facial_expression import process_facial_expression
        result_frame, facial_expression = process_facial_expression(test_image.copy())
        print(f"✓ Facial expression: {facial_expression}")
    except Exception as e:
        print(f"✗ Facial expression error: {e}")

if __name__ == "__main__":
    test_simple()
