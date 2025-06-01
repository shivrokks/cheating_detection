import cv2
import time
import os
import argparse
import cv2
import torch
from eye_movement import process_eye_movement
from head_pose import process_head_pose
from mobile_detection import process_mobile_detection
from lip_movement import process_lip_movement
from facial_expression import process_facial_expression
from person_detection import process_person_detection
from object_detection import process_object_detection
from behavior_analysis import process_behavior_analysis, load_training_data, add_training_sample, save_training_data
from audio_detection import initialize_audio_detection, process_audio_detection, draw_audio_info, cleanup_audio_detection

def configure_gpu_optimization():
    """Configure GPU settings for maximum performance with 12GB VRAM"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU optimization")
        return False
    
    # Get GPU information
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"Configuring optimization for: {gpu_name}")
    print(f"Available VRAM: {gpu_memory:.1f} GB")
    
    # ROCm-specific optimizations for AMD GPU
    os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1030'  # RX 6800M architecture
    os.environ['HIP_VISIBLE_DEVICES'] = '0'      # Use discrete GPU only
    
    # CuDNN optimizations for maximum performance
    torch.backends.cudnn.benchmark = True        # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False   # Allow non-deterministic for speed
    torch.backends.cudnn.allow_tf32 = True       # Enable TF32 for faster computation
    
    # Set GPU to performance mode
    torch.cuda.set_device(0)  # Explicitly use discrete GPU
    torch.cuda.empty_cache()  # Initial cache clear
    
    print("GPU optimization configured successfully!")
    print("- Removed frame resizing limitations")
    print("- Increased model input size to 1280px")
    print("- Enabled aggressive CuDNN optimizations")
    print("- Configured for maximum VRAM utilization")
    
    return True

def monitor_gpu_usage():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"GPU Memory Usage:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Cached: {cached:.2f} GB") 
        print(f"  Total: {total:.2f} GB")
        print(f"  Utilization: {(allocated/total)*100:.1f}%")
        
        return allocated, cached, total
    return 0, 0, 0

def main():
    # Configure GPU optimization for maximum VRAM utilization
    print("Configuring GPU optimization for maximum performance...")
    configure_gpu_optimization()
    
    # Check GPU status and ensure we're using the discrete GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"CUDA available with {gpu_count} GPU(s)")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Using discrete GPU 0: {torch.cuda.get_device_name(0)}")
        torch.cuda.set_device(0)  # Explicitly set to use GPU 0
        
        # Monitor initial GPU usage
        monitor_gpu_usage()
    else:
        print("CUDA not available, using CPU")
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    print("Webcam initialized successfully")

    # Create a log directory for screenshots
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)

    # Initialize audio detection
    print("Initializing audio detection...")
    # audio_initialized = initialize_audio_detection() # Temporarily commented out
    audio_initialized = False # Ensure audio is disabled to avoid related errors
    if audio_initialized:
        print("Audio detection initialized successfully")
    else:
        print("Warning: Audio detection failed to initialize or is disabled.")

    # Calibration for head pose
    calibrated_angles = None
    start_time = time.time()

    # Timers for each functionality
    head_misalignment_start_time = None
    eye_misalignment_start_time = None
    mobile_detection_start_time = None
    talking_detection_start_time = None
    expression_detection_start_time = None
    person_detection_start_time = None
    object_detection_start_time = None
    audio_detection_start_time = None  # Add audio detection timer

    # Previous states
    previous_head_state = "Looking at Screen"
    previous_eye_state = "Looking at Screen"
    previous_mobile_state = False
    previous_talking_state = False
    previous_expression_state = "Neutral"
    previous_person_count = 1
    previous_object_state = False

    # Initialize default values
    head_direction = "Looking at Screen"
    is_talking = False
    facial_expression = "Neutral"
    person_count = 1
    multiple_people = False
    new_person_entered = False
    suspicious_objects = False

    # Initialize behavior analysis
    behavior_analysis_start_time = None
    previous_behavior = "Normal"

    # Try to load pre-trained behavior model
    print("Loading behavior training data...")
    load_training_data()
    print("Behavior training data loaded successfully")

    # Parse command line arguments
    print("Parsing command line arguments...")
    parser = argparse.ArgumentParser(description='Cheating Detection System with Behavior Analysis')
    parser.add_argument('--train', action='store_true', help='Run in training mode to collect behavior data')
    args = parser.parse_args()
    print("Command line arguments parsed successfully")

    print("Starting main detection loop...")
    frame_count = 0
    process_every_n_frames = 1  # Process every frame for maximum accuracy with 12GB VRAM
    gpu_monitor_interval = 300  # Monitor GPU every 300 frames (~10 seconds at 30fps)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break

            frame_count += 1
            
            # Skip frames to reduce processing load
            if frame_count % process_every_n_frames != 0:
                # Just handle quit key without showing duplicate window
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
                
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"Processing frame {frame_count}...")
                
            # Monitor GPU usage periodically with 12GB VRAM
            if frame_count % gpu_monitor_interval == 0 and torch.cuda.is_available():
                print(f"\n--- GPU Status at frame {frame_count} ---")
                monitor_gpu_usage()
                print("--- End GPU Status ---\n")

            # Process eye movement
            frame, gaze_direction = process_eye_movement(frame)
            cv2.putText(frame, f"Gaze Direction: {gaze_direction}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Process head pose
            if time.time() - start_time <= 5:  # Calibration time
                cv2.putText(frame, "Calibrating... Keep your head straight", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if calibrated_angles is None:
                    _, calibrated_angles = process_head_pose(frame, None)
            else:
                frame, head_direction = process_head_pose(frame, calibrated_angles)
                cv2.putText(frame, f"Head Direction: {head_direction}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Process mobile detection with full resolution for better accuracy
            if frame_count % 3 == 0:  # Process mobile detection every 3rd frame (increased frequency)
                frame, mobile_detected = process_mobile_detection(frame)
            else:
                mobile_detected = False
            cv2.putText(frame, f"Mobile Detected: {mobile_detected}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Process lip movement detection
            frame, is_talking = process_lip_movement(frame)
            cv2.putText(frame, f"Talking: {is_talking}", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Process facial expression detection
            frame, facial_expression = process_facial_expression(frame)
            cv2.putText(frame, f"Expression: {facial_expression}", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Process person detection
            frame, person_count, multiple_people, new_person_entered = process_person_detection(frame)

            # Process object detection with full resolution for maximum accuracy
            if frame_count % 4 == 0:  # Process object detection every 4th frame (increased frequency)
                frame, suspicious_objects, detected_objects = process_object_detection(frame)
            else:
                suspicious_objects = False
                detected_objects = {}

            # Process audio detection
            audio_result = process_audio_detection() if audio_initialized else {
                'is_suspicious': False,
                'suspicion_score': 0,
                'detected_text': '',
                'suspicious_words': [],
                'recent_detections': 0
            }
            
            # Draw audio information on frame
            frame = draw_audio_info(frame, audio_result)

            # Prepare data for behavior analysis
            head_data = {
                'direction': head_direction,
                'rapid_movement': head_direction == "Rapid Movement",
                'pitch': getattr(process_head_pose, 'last_pitch', 0),
                'yaw': getattr(process_head_pose, 'last_yaw', 0),
                'roll': getattr(process_head_pose, 'last_roll', 0)
            }

            eye_data = {
                'direction': gaze_direction
            }

            lip_data = {
                'is_talking': is_talking
            }

            expression_data = {
                'expression': facial_expression
            }

            person_data = {
                'count': person_count,
                'multiple_people': multiple_people,
                'new_person': new_person_entered
            }

            object_data = {
                'suspicious_objects': suspicious_objects,
                'detected_objects': detected_objects
            }

            # Process behavior analysis
            frame, behavior_result = process_behavior_analysis(
                head_data, eye_data, lip_data, expression_data, person_data, object_data, frame, audio_result
            )

            # Extract behavior information
            behavior = behavior_result['behavior']
            confidence = behavior_result['confidence']

            # Check for head misalignment
            if head_direction != "Looking at Screen":
                if head_misalignment_start_time is None:
                    head_misalignment_start_time = time.time()
                elif time.time() - head_misalignment_start_time >= 3:
                    filename = os.path.join(log_dir, f"head_{head_direction}_{int(time.time())}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
                    head_misalignment_start_time = None  # Reset timer
            else:
                head_misalignment_start_time = None  # Reset timer

            # Check for eye misalignment
            if gaze_direction != "Looking at Screen":
                if eye_misalignment_start_time is None:
                    eye_misalignment_start_time = time.time()
                elif time.time() - eye_misalignment_start_time >= 3:
                    filename = os.path.join(log_dir, f"eye_{gaze_direction}_{int(time.time())}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
                    eye_misalignment_start_time = None  # Reset timer
            else:
                eye_misalignment_start_time = None  # Reset timer

            # Check for mobile detection
            if mobile_detected:
                if mobile_detection_start_time is None:
                    mobile_detection_start_time = time.time()
                elif time.time() - mobile_detection_start_time >= 3:
                    filename = os.path.join(log_dir, f"mobile_detected_{int(time.time())}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
                    mobile_detection_start_time = None  # Reset timer
            else:
                mobile_detection_start_time = None  # Reset timer

            # Check for talking detection
            if is_talking:
                if talking_detection_start_time is None:
                    talking_detection_start_time = time.time()
                elif time.time() - talking_detection_start_time >= 3:
                    filename = os.path.join(log_dir, f"talking_{int(time.time())}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
                    talking_detection_start_time = None  # Reset timer
            else:
                talking_detection_start_time = None  # Reset timer

            # Check for suspicious facial expressions
            if facial_expression != "Neutral":
                if expression_detection_start_time is None:
                    expression_detection_start_time = time.time()
                elif time.time() - expression_detection_start_time >= 3:
                    filename = os.path.join(log_dir, f"expression_{facial_expression}_{int(time.time())}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
                    expression_detection_start_time = None  # Reset timer
            else:
                expression_detection_start_time = None  # Reset timer

            # Check for multiple people or new person
            if multiple_people or new_person_entered:
                if person_detection_start_time is None:
                    person_detection_start_time = time.time()
                elif time.time() - person_detection_start_time >= 2:
                    filename = os.path.join(log_dir, f"person_count_{person_count}_{int(time.time())}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
                    person_detection_start_time = None  # Reset timer
            else:
                person_detection_start_time = None  # Reset timer

            # Check for suspicious objects
            if suspicious_objects:
                if object_detection_start_time is None:
                    object_detection_start_time = time.time()
                elif time.time() - object_detection_start_time >= 3:
                    filename = os.path.join(log_dir, f"suspicious_objects_{int(time.time())}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
                    object_detection_start_time = None  # Reset timer
            else:
                object_detection_start_time = None  # Reset timer

            # Check for suspicious audio
            if audio_result['is_suspicious']:
                if audio_detection_start_time is None:
                    audio_detection_start_time = time.time()
                elif time.time() - audio_detection_start_time >= 2:
                    filename = os.path.join(log_dir, f"suspicious_audio_{int(time.time())}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
                    print(f"Detected suspicious speech: '{audio_result['detected_text']}'")
                    print(f"Suspicious words: {audio_result['suspicious_words']}")
                    audio_detection_start_time = None  # Reset timer
            else:
                audio_detection_start_time = None  # Reset timer

            # Check for suspicious behavior
            if behavior == "Suspicious" and confidence > 0.6:
                if behavior_analysis_start_time is None:
                    behavior_analysis_start_time = time.time()
                elif time.time() - behavior_analysis_start_time >= 2:
                    filename = os.path.join(log_dir, f"suspicious_behavior_{int(time.time())}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
                    behavior_analysis_start_time = None  # Reset timer
            else:
                behavior_analysis_start_time = None  # Reset timer

            # Display the combined output (disable for headless mode)
            try:
                cv2.imshow("Combined Detection", frame)
            except cv2.error as e:
                print(f"Display error (running in headless mode): {e}")
                # Continue without display

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n') and args.train:
                add_training_sample(behavior_result['features'], False)
                print("Added normal behavior sample")
            elif key == ord('s') and args.train:
                add_training_sample(behavior_result['features'], True)
                print("Added suspicious behavior sample")
    finally:
        # Save any collected training data
        if args.train:
            save_training_data()
            print("Training data saved")

        # Clean up audio detection resources
        if audio_initialized: # This condition will be false
            # cleanup_audio_detection() # Temporarily commented out
            pass

        cap.release()
        cv2.destroyAllWindows()

        # If this was run in training mode, print instructions
        if args.train:
            print("\\nTraining mode completed.")
            print("You can now run the system normally to use the trained behavior model.")

if __name__ == "__main__":
    main()