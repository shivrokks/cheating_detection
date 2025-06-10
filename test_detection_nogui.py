#!/usr/bin/env python3
"""
Test detection functions without GUI
"""

import cv2
import numpy as np
import time

def test_detection_functions():
    """Test detection functions with webcam but no GUI"""
    print("Testing detection functions (no GUI)...")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    frame_count = 0
    calibrated_angles = None
    calibration_frames = 0
    
    print("Starting detection test for 10 frames...")
    
    while frame_count < 10:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame_count += 1
        print(f"\n--- Testing Frame {frame_count} ---")
        
        try:
            # Test eye movement detection
            from eye_movement import process_eye_movement
            frame_copy = frame.copy()
            _, gaze_direction = process_eye_movement(frame_copy)
            print(f"✓ Gaze: {gaze_direction}")
            
            # Test head pose detection
            from head_pose import process_head_pose
            frame_copy = frame.copy()
            if calibration_frames < 5:  # Calibrate for first 5 frames
                _, head_result = process_head_pose(frame_copy, None)
                if isinstance(head_result, tuple) and len(head_result) == 3:
                    calibrated_angles = head_result
                calibration_frames += 1
                head_direction = "Calibrating..."
            else:
                _, head_direction = process_head_pose(frame_copy, calibrated_angles)
            print(f"✓ Head: {head_direction}")
            
            # Test mobile detection
            from mobile_detection import process_mobile_detection
            frame_copy = frame.copy()
            _, mobile_detected = process_mobile_detection(frame_copy)
            print(f"✓ Mobile: {mobile_detected}")
            
            # Test lip movement detection
            from lip_movement import process_lip_movement
            frame_copy = frame.copy()
            _, is_talking = process_lip_movement(frame_copy)
            print(f"✓ Talking: {is_talking}")
            
            # Test facial expression detection
            from facial_expression import process_facial_expression
            frame_copy = frame.copy()
            _, facial_expression = process_facial_expression(frame_copy)
            print(f"✓ Expression: {facial_expression}")
            
            # Test person detection
            from person_detection import process_person_detection
            frame_copy = frame.copy()
            _, person_count, multiple_people, new_person_entered = process_person_detection(frame_copy)
            print(f"✓ People: {person_count}, Multiple: {multiple_people}, New: {new_person_entered}")
            
        except Exception as e:
            print(f"✗ Error in frame {frame_count}: {e}")
        
        # Small delay between frames
        time.sleep(0.5)
    
    cap.release()
    print("\nDetection test completed!")

if __name__ == "__main__":
    test_detection_functions()
