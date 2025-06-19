#!/usr/bin/env python3
"""
Test script to simulate mobile camera detection and verify all directions work
"""

import cv2
import numpy as np
import time

def test_mobile_detection():
    """Test detection with mobile-like conditions"""
    print("Testing mobile camera detection...")
    print("This test simulates mobile camera conditions")
    print("Move your head and eyes slowly to test all directions")
    print("Press 'q' to quit")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties to simulate mobile camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
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
    start_time = time.time()
    
    # Track detected directions
    gaze_directions = set()
    head_directions = set()
    
    # Statistics
    total_frames = 0
    successful_detections = 0
    
    print("\nCalibrating head pose for 5 seconds...")
    print("Keep your head straight and look at the camera...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame_count += 1
        total_frames += 1
        current_time = time.time()
        
        try:
            # Test eye movement detection
            frame_copy = frame.copy()
            frame_copy, gaze_direction = process_eye_movement(frame_copy)
            
            # Test head pose detection with calibration
            frame_copy = frame.copy()
            if current_time - start_time <= 5:  # Calibrate for first 5 seconds
                frame_copy, head_result = process_head_pose(frame_copy, None)
                if isinstance(head_result, tuple) and len(head_result) == 3:
                    calibrated_angles = head_result
                head_direction = "Calibrating..."
                cv2.putText(frame, "CALIBRATING - Keep head straight", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Time left: {5 - int(current_time - start_time)}s", (50, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                frame_copy, head_direction = process_head_pose(frame_copy, calibrated_angles)
                head_directions.add(head_direction)
                
                # Instructions for testing
                if frame_count % 120 == 0:  # Every 4 seconds
                    directions_to_test = ["left", "right", "up", "down", "center"]
                    current_instruction = directions_to_test[(frame_count // 120) % len(directions_to_test)]
                    print(f"Now look {current_instruction}")
            
            gaze_directions.add(gaze_direction)
            successful_detections += 1
            
            # Display current detections
            cv2.putText(frame, f"Gaze: {gaze_direction}", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Head: {head_direction}", (20, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Display statistics
            success_rate = (successful_detections / total_frames) * 100
            cv2.putText(frame, f"Success: {success_rate:.1f}%", (20, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display detected directions count
            cv2.putText(frame, f"Gaze dirs: {len(gaze_directions)}", (20, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Head dirs: {len(head_directions)}", (20, 270), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Print progress every 60 frames
            if frame_count % 60 == 0 and current_time - start_time > 5:
                print(f"Frame {frame_count}: Gaze={gaze_direction}, Head={head_direction}")
                print(f"  Gaze directions: {sorted(gaze_directions)}")
                print(f"  Head directions: {sorted(head_directions)}")
                print(f"  Success rate: {success_rate:.1f}%")
            
        except Exception as e:
            print(f"Frame {frame_count}: Detection error: {e}")
            cv2.putText(frame, f"ERROR: {str(e)[:50]}", (20, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Mobile Detection Test', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Auto-quit after 2 minutes
        if current_time - start_time > 120:
            print("Auto-quitting after 2 minutes...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("MOBILE DETECTION TEST RESULTS:")
    print("="*60)
    print(f"Total frames processed: {total_frames}")
    print(f"Successful detections: {successful_detections}")
    print(f"Success rate: {(successful_detections/total_frames)*100:.1f}%")
    print(f"Gaze directions detected: {sorted(gaze_directions)}")
    print(f"Head directions detected: {sorted(head_directions)}")
    
    # Check coverage
    expected_gaze = {"Looking Left", "Looking Right", "Looking Up", "Looking Down", "Looking Center"}
    expected_head = {"Looking Left", "Looking Right", "Looking Up", "Looking Down", "Looking at Screen"}
    
    gaze_coverage = len(gaze_directions & expected_gaze) / len(expected_gaze) * 100
    head_coverage = len(head_directions & expected_head) / len(expected_head) * 100
    
    print(f"Gaze direction coverage: {gaze_coverage:.1f}%")
    print(f"Head direction coverage: {head_coverage:.1f}%")
    
    if gaze_coverage >= 80 and head_coverage >= 80:
        print("✅ EXCELLENT: Good direction coverage!")
    elif gaze_coverage >= 60 and head_coverage >= 60:
        print("⚠️  GOOD: Decent direction coverage")
    else:
        print("❌ POOR: Low direction coverage - needs improvement")
    
    print("Mobile detection test completed!")

if __name__ == "__main__":
    test_mobile_detection()
