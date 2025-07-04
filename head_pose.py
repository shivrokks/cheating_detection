import cv2
import dlib
import numpy as np
import math
from collections import deque
import time

# Load face detector & landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

# Load OpenCV face detector as fallback with error handling
try:
    opencv_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if opencv_face_cascade.empty():
        print("Warning: OpenCV face cascade failed to load, using dlib only")
        opencv_face_cascade = None
except Exception as e:
    print(f"Error loading OpenCV cascade: {e}")
    opencv_face_cascade = None

def detect_faces_improved(gray_frame):
    """Improved face detection using multiple methods"""
    faces = []

    # Try dlib detector first with multiple scales and preprocessing
    # Method 1: Standard dlib detection
    dlib_faces = detector(gray_frame, 1)
    if len(dlib_faces) > 0:
        return dlib_faces

    # Method 2: More sensitive dlib detection
    dlib_faces = detector(gray_frame, 0)
    if len(dlib_faces) > 0:
        return dlib_faces

    # Method 3: Try with different image preprocessing
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_frame, (3, 3), 0)
    dlib_faces = detector(blurred, 0)
    if len(dlib_faces) > 0:
        return dlib_faces

    # Method 4: Try with contrast enhancement
    enhanced = cv2.convertScaleAbs(gray_frame, alpha=1.2, beta=10)
    dlib_faces = detector(enhanced, 0)
    if len(dlib_faces) > 0:
        return dlib_faces

    # Method 5: OpenCV detector with multiple parameter sets (only if available)
    if opencv_face_cascade is not None:
        try:
            # Try with relaxed parameters first
            opencv_faces = opencv_face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(opencv_faces) == 0:
                # Try with even more relaxed parameters
                opencv_faces = opencv_face_cascade.detectMultiScale(
                    gray_frame,
                    scaleFactor=1.1,
                    minNeighbors=2,
                    minSize=(15, 15),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
        except Exception as e:
            print(f"OpenCV cascade detection error: {e}")
            opencv_faces = []
    else:
        opencv_faces = []

    # Convert OpenCV rectangles to dlib rectangles
    for (x, y, w, h) in opencv_faces:
        faces.append(dlib.rectangle(x, y, x + w, y + h))

    return faces

# 3D Model Points (Mapped to Facial Landmarks)
model_points = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -50.0, -10.0),    # Chin
    (-30.0, 40.0, -10.0),   # Left eye
    (30.0, 40.0, -10.0),    # Right eye
    (-25.0, -30.0, -10.0),  # Left mouth corner
    (25.0, -30.0, -10.0)    # Right mouth corner
], dtype=np.float64)

# Camera Calibration (Assuming 640x480)
focal_length = 640
center = (320, 240)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float64)

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

# Define thresholds for "Looking at Screen"
CALIBRATION_TIME = 5  # Time to set neutral position

# Smoothing filter for stable head pose estimation
ANGLE_HISTORY_SIZE = 10
yaw_history = deque(maxlen=ANGLE_HISTORY_SIZE)
pitch_history = deque(maxlen=ANGLE_HISTORY_SIZE)
roll_history = deque(maxlen=ANGLE_HISTORY_SIZE)

# For rapid movement detection
MOVEMENT_HISTORY_SIZE = 5
yaw_movement_history = deque(maxlen=MOVEMENT_HISTORY_SIZE)
pitch_movement_history = deque(maxlen=MOVEMENT_HISTORY_SIZE)
roll_movement_history = deque(maxlen=MOVEMENT_HISTORY_SIZE)
last_angles = None
rapid_movement_threshold = 20.0  # Degrees per frame threshold for rapid movement (much higher to reduce false positives)

# Global variables for state management
previous_state = "Looking at Screen"
calibrated_angles = None
rapid_movement_detected = False
last_rapid_movement_time = 0
frame_count = 0  # Track frames to avoid early rapid movement detection
DEBUG_MODE = False  # Set to True for debugging

def get_head_pose_angles(image_points):
    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0

    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

def smooth_angle(angle_history, new_angle):
    angle_history.append(new_angle)
    return np.mean(angle_history)

def detect_rapid_movement(current_angles, last_angles):
    """Detect if there is rapid head movement between frames"""
    if last_angles is None:
        return False

    pitch_diff = abs(current_angles[0] - last_angles[0])
    yaw_diff = abs(current_angles[1] - last_angles[1])
    roll_diff = abs(current_angles[2] - last_angles[2])

    # Store movement rates for analysis
    yaw_movement_history.append(yaw_diff)
    pitch_movement_history.append(pitch_diff)
    roll_movement_history.append(roll_diff)

    # Check if any angle changed rapidly
    return (pitch_diff > rapid_movement_threshold or
            yaw_diff > rapid_movement_threshold or
            roll_diff > rapid_movement_threshold)

def process_head_pose(frame, calibrated_angles=None):
    global previous_state, last_angles, rapid_movement_detected, last_rapid_movement_time, frame_count

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Enhance image for better face detection
    enhanced_gray = cv2.equalizeHist(gray)

    # Use improved face detection
    faces = detect_faces_improved(enhanced_gray)
    head_direction = "Looking at Screen"

    # print(f"DEBUG: Head pose - detected {len(faces)} faces")  # Commented out for production

    # Reset rapid movement flag if it's been a while
    if rapid_movement_detected and time.time() - last_rapid_movement_time > 1.0:
        rapid_movement_detected = False

    for face in faces:
        landmarks = predictor(enhanced_gray, face)
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),    # Chin
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye
            (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
        ], dtype=np.float64)

        angles = get_head_pose_angles(image_points)
        if angles is None:
            continue

        pitch = smooth_angle(pitch_history, angles[0])
        yaw = smooth_angle(yaw_history, angles[1])
        roll = smooth_angle(roll_history, angles[2])

        current_angles = (pitch, yaw, roll)

        # Store angles for external access
        process_head_pose.last_pitch = pitch
        process_head_pose.last_yaw = yaw
        process_head_pose.last_roll = roll

        # Detect rapid head movement (only after enough frames to avoid false positives)
        if last_angles is not None and frame_count > 10:  # Wait for 10 frames before detecting rapid movement
            if detect_rapid_movement(current_angles, last_angles):
                rapid_movement_detected = True
                last_rapid_movement_time = time.time()

                # Calculate average movement rates
                avg_yaw_movement = np.mean(yaw_movement_history)
                avg_pitch_movement = np.mean(pitch_movement_history)
                avg_roll_movement = np.mean(roll_movement_history)

                # Store movement rates for external access
                process_head_pose.last_yaw_rate = avg_yaw_movement
                process_head_pose.last_pitch_rate = avg_pitch_movement
                process_head_pose.last_roll_rate = avg_roll_movement

                # Draw movement rates on frame
                cv2.putText(frame, f"Yaw Rate: {avg_yaw_movement:.2f}",
                           (face.left(), face.bottom() + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, f"Pitch Rate: {avg_pitch_movement:.2f}",
                           (face.left(), face.bottom() + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, f"Roll Rate: {avg_roll_movement:.2f}",
                           (face.left(), face.bottom() + 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Update last angles for next frame
        last_angles = current_angles

        # If calibrating, return the current angles as calibrated_angles
        if calibrated_angles is None:
            return frame, (pitch, yaw, roll)

        # Use calibrated angles for head pose detection
        if len(calibrated_angles) == 3:
            pitch_offset, yaw_offset, roll_offset = calibrated_angles
        else:
            raise ValueError("calibrated_angles must contain exactly three elements.")
        # Much more sensitive thresholds
        PITCH_THRESHOLD = 5  # More sensitive
        YAW_THRESHOLD = 8   # More sensitive
        ROLL_THRESHOLD = 4  # More sensitive

        # Determine head direction with more sensitive detection
        if rapid_movement_detected:
            current_state = "Rapid Movement"
        elif yaw < yaw_offset - 8:  # More sensitive left detection
            current_state = "Looking Left"
        elif yaw > yaw_offset + 8:  # More sensitive right detection
            current_state = "Looking Right"
        elif pitch > pitch_offset + 6:  # More sensitive up detection
            current_state = "Looking Up"
        elif pitch < pitch_offset - 6:  # More sensitive down detection
            current_state = "Looking Down"
        elif abs(roll - roll_offset) > 5:  # More sensitive tilt detection
            current_state = "Tilted"
        elif abs(yaw - yaw_offset) <= YAW_THRESHOLD and abs(pitch - pitch_offset) <= PITCH_THRESHOLD and abs(roll - roll_offset) <= ROLL_THRESHOLD:
            current_state = "Looking at Screen"
        else:
            current_state = previous_state

        previous_state = current_state
        head_direction = current_state

        # Draw head pose angles on frame
        cv2.putText(frame, f"Pitch: {pitch:.2f}", (face.left(), face.top() - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, f"Yaw: {yaw:.2f}", (face.left(), face.top() - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, f"Roll: {roll:.2f}", (face.left(), face.top() - 0),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Debug print for testing (enabled for troubleshooting)
        if DEBUG_MODE and calibrated_angles is not None:
            print(f"DEBUG: Head pose - pitch: {pitch:.2f} (offset: {pitch_offset:.2f}), yaw: {yaw:.2f} (offset: {yaw_offset:.2f}), roll: {roll:.2f} (offset: {roll_offset:.2f}), direction: {head_direction}")

    return frame, head_direction
