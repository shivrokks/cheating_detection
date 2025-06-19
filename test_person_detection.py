#!/usr/bin/env python3
"""
Test script specifically for person detection
"""

import cv2
import numpy as np
import time

def test_person_detection():
    """Test person detection with webcam"""
    print("Testing person detection...")
    print("Press 'q' to quit")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Import person detection
    try:
        from person_detection import process_person_detection
        print("✓ Person detection module loaded successfully")
    except Exception as e:
        print(f"✗ Error loading person detection: {e}")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame_count += 1
        
        try:
            # Test person detection
            processed_frame, person_count, multiple_people, new_person_entered = process_person_detection(frame.copy())
            
            print(f"Frame {frame_count}: People={person_count}, Multiple={multiple_people}, New={new_person_entered}")
            
            # Display the frame
            cv2.imshow('Person Detection Test', processed_frame)
            
        except Exception as e:
            print(f"Frame {frame_count}: Person detection error: {e}")
            cv2.imshow('Person Detection Test', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Small delay
        time.sleep(0.1)
    
    cap.release()
    cv2.destroyAllWindows()
    print("Person detection test completed!")

if __name__ == "__main__":
    test_person_detection()
