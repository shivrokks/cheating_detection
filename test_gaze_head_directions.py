#!/usr/bin/env python3
"""
Test script to verify all gaze and head directions are working
"""

import cv2
import numpy as np
import time

def test_direction_detection():
    """Test gaze and head direction detection"""
    print("Testing gaze and head direction detection...")
    print("Move your eyes and head in different directions")
    print("Press 'q' to quit")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Import detection modules
    try:
        from eye_movement import process_eye_movement
        from head_pose import process_head_pose
        print("✓ Detection modules loaded successfully")
    except Exception as e:
        print(f"✗ Error loading detection modules: {e}")
        return
    
    frame_count = 0
    calibrated_angles = None
    calibration_frames = 0
    
    # Track detected directions
    gaze_directions = set()
    head_directions = set()
    
    print("\nCalibrating head pose for 5 seconds...")
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame_count += 1
        current_time = time.time()
        
        try:
            # Test eye movement detection
            frame_copy = frame.copy()
            frame_copy, gaze_direction = process_eye_movement(frame_copy)
            gaze_directions.add(gaze_direction)
            
            # Test head pose detection with calibration
            frame_copy = frame.copy()
            if current_time - start_time <= 5:  # Calibrate for first 5 seconds
                frame_copy, head_result = process_head_pose(frame_copy, None)
                if isinstance(head_result, tuple) and len(head_result) == 3:
                    calibrated_angles = head_result
                head_direction = "Calibrating..."
                cv2.putText(frame, "CALIBRATING - Keep head straight", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                frame_copy, head_direction = process_head_pose(frame_copy, calibrated_angles)
                head_directions.add(head_direction)
            
            # Display current detections
            cv2.putText(frame, f"Gaze: {gaze_direction}", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Head: {head_direction}", (20, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Display detected directions so far
            y_offset = 200
            cv2.putText(frame, "Detected Gaze Directions:", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            for i, direction in enumerate(sorted(gaze_directions)):
                cv2.putText(frame, f"  {direction}", (20, y_offset + 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            y_offset = 350
            cv2.putText(frame, "Detected Head Directions:", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            for i, direction in enumerate(sorted(head_directions)):
                cv2.putText(frame, f"  {direction}", (20, y_offset + 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Print every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: Gaze={gaze_direction}, Head={head_direction}")
                print(f"  Gaze directions detected: {sorted(gaze_directions)}")
                print(f"  Head directions detected: {sorted(head_directions)}")
            
            # Display the frame
            cv2.imshow('Direction Detection Test', frame)
            
        except Exception as e:
            print(f"Frame {frame_count}: Detection error: {e}")
            cv2.imshow('Direction Detection Test', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Small delay
        time.sleep(0.033)  # ~30 FPS
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print("="*50)
    print(f"Gaze directions detected: {sorted(gaze_directions)}")
    print(f"Head directions detected: {sorted(head_directions)}")
    
    # Check if all expected directions were detected
    expected_gaze = {"Looking Left", "Looking Right", "Looking Up", "Looking Down", "Looking Center"}
    expected_head = {"Looking Left", "Looking Right", "Looking Up", "Looking Down", "Looking at Screen"}
    
    missing_gaze = expected_gaze - gaze_directions
    missing_head = expected_head - head_directions
    
    if missing_gaze:
        print(f"⚠️  Missing gaze directions: {missing_gaze}")
    else:
        print("✅ All gaze directions detected!")
        
    if missing_head:
        print(f"⚠️  Missing head directions: {missing_head}")
    else:
        print("✅ All head directions detected!")
    
    print("Direction detection test completed!")

if __name__ == "__main__":
    test_direction_detection()
