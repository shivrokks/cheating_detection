#!/usr/bin/env python3

import cv2
import numpy as np
from eye_movement import process_eye_movement
from head_pose import process_head_pose

def test_detection_with_webcam():
    """Test detection functions with webcam feed"""
    print("Testing detection functions with webcam...")
    print("Press 'q' to quit")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame_count += 1
        
        # Test eye movement detection
        try:
            processed_frame, gaze_direction = process_eye_movement(frame.copy())
            print(f"Frame {frame_count}: Gaze direction: {gaze_direction}")
        except Exception as e:
            print(f"Frame {frame_count}: Eye movement error: {e}")
            gaze_direction = "Error"
        
        # Test head pose detection
        try:
            processed_frame, head_direction = process_head_pose(frame.copy(), None)
            print(f"Frame {frame_count}: Head direction: {head_direction}")
        except Exception as e:
            print(f"Frame {frame_count}: Head pose error: {e}")
            head_direction = "Error"
        
        # Display results on frame
        cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Head: {head_direction}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Detection Test', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Test only first 10 frames
        if frame_count >= 10:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")

if __name__ == "__main__":
    test_detection_with_webcam()
