#!/usr/bin/env python3
"""
Extended test to verify detection functions with more frames
"""

import cv2
import numpy as np
import time

def test_detection_functions_extended():
    """Test detection functions with more frames to see variation"""
    print("Testing detection functions (extended test - 20 frames)...")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    frame_count = 0
    calibrated_angles = None
    calibration_frames = 0
    
    # Track detection variations
    gaze_variations = set()
    head_variations = set()
    expression_variations = set()
    talking_detections = 0
    mobile_detections = 0
    
    print("Starting extended detection test for 20 frames...")
    print("Try moving your head, changing gaze, talking, and making expressions!")
    
    while frame_count < 20:
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
            gaze_variations.add(gaze_direction)
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
            head_variations.add(head_direction)
            print(f"✓ Head: {head_direction}")
            
            # Test mobile detection
            from mobile_detection import process_mobile_detection
            frame_copy = frame.copy()
            _, mobile_detected = process_mobile_detection(frame_copy)
            if mobile_detected:
                mobile_detections += 1
            print(f"✓ Mobile: {mobile_detected}")
            
            # Test lip movement detection
            from lip_movement import process_lip_movement
            frame_copy = frame.copy()
            _, is_talking = process_lip_movement(frame_copy)
            if is_talking:
                talking_detections += 1
            print(f"✓ Talking: {is_talking}")
            
            # Test facial expression detection
            from facial_expression import process_facial_expression
            frame_copy = frame.copy()
            _, facial_expression = process_facial_expression(frame_copy)
            expression_variations.add(facial_expression)
            print(f"✓ Expression: {facial_expression}")
            
            # Test person detection
            from person_detection import process_person_detection
            frame_copy = frame.copy()
            _, person_count, multiple_people, new_person_entered = process_person_detection(frame_copy)
            print(f"✓ People: {person_count}, Multiple: {multiple_people}, New: {new_person_entered}")
            
        except Exception as e:
            print(f"✗ Error in frame {frame_count}: {e}")
        
        # Small delay between frames
        time.sleep(0.3)
    
    cap.release()
    
    # Print summary of variations detected
    print("\n" + "="*60)
    print("🎯 DETECTION SUMMARY")
    print("="*60)
    
    print(f"📊 Total frames processed: {frame_count}")
    print(f"👁️  Gaze variations detected: {len(gaze_variations)}")
    print(f"    Variations: {', '.join(sorted(gaze_variations))}")
    
    print(f"🗣️  Head pose variations detected: {len(head_variations)}")
    print(f"    Variations: {', '.join(sorted(head_variations))}")
    
    print(f"😊 Expression variations detected: {len(expression_variations)}")
    print(f"    Variations: {', '.join(sorted(expression_variations))}")
    
    print(f"🎤 Talking detected in: {talking_detections}/{frame_count} frames ({talking_detections/frame_count*100:.1f}%)")
    print(f"📱 Mobile detected in: {mobile_detections}/{frame_count} frames ({mobile_detections/frame_count*100:.1f}%)")
    
    # Assessment
    print("\n" + "="*60)
    print("📈 ASSESSMENT")
    print("="*60)
    
    total_variations = len(gaze_variations) + len(head_variations) + len(expression_variations)
    
    if total_variations >= 8:
        print("🟢 EXCELLENT: High variation in detection results - system is very responsive!")
    elif total_variations >= 5:
        print("🟡 GOOD: Moderate variation in detection results - system is working well!")
    elif total_variations >= 3:
        print("🟠 FAIR: Some variation detected - system is functional but could be more sensitive!")
    else:
        print("🔴 POOR: Low variation - detection system may need further tuning!")
    
    print(f"   Total unique detection states: {total_variations}")
    print(f"   Expected minimum for good performance: 5-8")
    
    print("\n✅ Extended detection test completed!")

if __name__ == "__main__":
    test_detection_functions_extended()
