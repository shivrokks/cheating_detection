#!/usr/bin/env python3
"""
Comprehensive test to verify all detection functions are working properly
"""

import cv2
import numpy as np
import time
import sys
import os

def test_detection_with_webcam():
    """Test all detection functions with webcam feed"""
    print("Starting comprehensive detection test...")
    print("This will test all detection functions with live webcam feed")
    print("Move your head, change gaze direction, talk, and make expressions")
    print("Press 'q' to quit")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    frame_count = 0
    start_time = time.time()
    
    # Initialize calibration for head pose
    calibrated_angles = None
    calibration_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame_count += 1
        current_time = time.time()
        
        # Test all detection functions
        try:
            # Test eye movement detection
            from eye_movement import process_eye_movement
            frame_copy = frame.copy()
            frame_copy, gaze_direction = process_eye_movement(frame_copy)
            
            # Test head pose detection with calibration
            from head_pose import process_head_pose
            frame_copy = frame.copy()
            if calibration_frames < 30:  # Calibrate for first 30 frames
                frame_copy, head_result = process_head_pose(frame_copy, None)
                if isinstance(head_result, tuple) and len(head_result) == 3:
                    calibrated_angles = head_result
                calibration_frames += 1
                head_direction = "Calibrating..."
            else:
                frame_copy, head_direction = process_head_pose(frame_copy, calibrated_angles)
            
            # Test mobile detection
            from mobile_detection import process_mobile_detection
            frame_copy = frame.copy()
            frame_copy, mobile_detected = process_mobile_detection(frame_copy)
            
            # Test lip movement detection
            from lip_movement import process_lip_movement
            frame_copy = frame.copy()
            frame_copy, is_talking = process_lip_movement(frame_copy)
            
            # Test facial expression detection
            from facial_expression import process_facial_expression
            frame_copy = frame.copy()
            frame_copy, facial_expression = process_facial_expression(frame_copy)
            
            # Test person detection
            from person_detection import process_person_detection
            frame_copy = frame.copy()
            frame_copy, person_count, multiple_people, new_person_entered = process_person_detection(frame_copy)
            
            # Display results on frame
            y_offset = 30
            line_height = 25
            
            cv2.putText(frame, f"Frame: {frame_count}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += line_height
            
            cv2.putText(frame, f"Gaze: {gaze_direction}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += line_height
            
            cv2.putText(frame, f"Head: {head_direction}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += line_height
            
            cv2.putText(frame, f"Mobile: {mobile_detected}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += line_height
            
            cv2.putText(frame, f"Talking: {is_talking}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += line_height
            
            cv2.putText(frame, f"Expression: {facial_expression}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += line_height
            
            cv2.putText(frame, f"People: {person_count}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Print results every 30 frames
            if frame_count % 30 == 0:
                elapsed_time = current_time - start_time
                fps = frame_count / elapsed_time
                print(f"\n--- Frame {frame_count} (FPS: {fps:.1f}) ---")
                print(f"Gaze: {gaze_direction}")
                print(f"Head: {head_direction}")
                print(f"Mobile: {mobile_detected}")
                print(f"Talking: {is_talking}")
                print(f"Expression: {facial_expression}")
                print(f"People: {person_count}")
                print("-" * 40)
            
        except Exception as e:
            print(f"Error in frame {frame_count}: {e}")
            cv2.putText(frame, f"Error: {str(e)[:50]}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow('Comprehensive Detection Test', frame)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        # Auto-quit after 5 minutes
        if current_time - start_time > 300:
            print("Test completed after 5 minutes")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final summary
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    print(f"\nTest Summary:")
    print(f"Total frames processed: {frame_count}")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Average FPS: {fps:.1f}")
    print("Test completed successfully!")

if __name__ == "__main__":
    test_detection_with_webcam()
