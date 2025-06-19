import cv2
import dlib
import numpy as np
from collections import deque

# Load dlib’s face detector and 68 landmarks model
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

# Smoothing for gaze detection
gaze_history = deque(maxlen=5)  # Keep last 5 gaze directions
frame_count = 0
DEBUG_MODE = False  # Set to True for debugging

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
            # Try with very relaxed parameters first
            opencv_faces = opencv_face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.03,
                minNeighbors=2,
                minSize=(15, 15),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(opencv_faces) == 0:
                # Try with extremely relaxed parameters
                opencv_faces = opencv_face_cascade.detectMultiScale(
                    gray_frame,
                    scaleFactor=1.05,
                    minNeighbors=1,
                    minSize=(10, 10),
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

def detect_pupil(eye_region):
    if eye_region.size == 0:
        return None, None

    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)

    # Use adaptive threshold for better pupil detection
    threshold_eye = cv2.adaptiveThreshold(blurred_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Filter contours by area to avoid noise
        valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
        if valid_contours:
            pupil_contour = max(valid_contours, key=cv2.contourArea)
            px, py, pw, ph = cv2.boundingRect(pupil_contour)
            return (px + pw // 2, py + ph // 2), (px, py, pw, ph)
    return None, None

def process_eye_movement(frame):
    global gaze_history, frame_count

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Enhance image for better face detection
    enhanced_gray = cv2.equalizeHist(gray)

    # Use improved face detection
    faces = detect_faces_improved(enhanced_gray)
    gaze_direction = "Looking Center"

    if DEBUG_MODE:
        print(f"DEBUG: Eye movement - detected {len(faces)} faces")

    for face in faces:
        landmarks = predictor(enhanced_gray, face)
        
        # Extract left and right eye landmarks
        left_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        right_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
        
        # Get bounding rectangles for the eyes
        left_eye_rect = cv2.boundingRect(left_eye_points)
        right_eye_rect = cv2.boundingRect(right_eye_points)
        
        # Extract eye regions with bounds checking
        h, w = frame.shape[:2]
        left_y1 = max(0, left_eye_rect[1])
        left_y2 = min(h, left_eye_rect[1] + left_eye_rect[3])
        left_x1 = max(0, left_eye_rect[0])
        left_x2 = min(w, left_eye_rect[0] + left_eye_rect[2])

        right_y1 = max(0, right_eye_rect[1])
        right_y2 = min(h, right_eye_rect[1] + right_eye_rect[3])
        right_x1 = max(0, right_eye_rect[0])
        right_x2 = min(w, right_eye_rect[0] + right_eye_rect[2])

        left_eye = frame[left_y1:left_y2, left_x1:left_x2]
        right_eye = frame[right_y1:right_y2, right_x1:right_x2]
        
        # Detect pupils
        left_pupil, left_bbox = detect_pupil(left_eye)
        right_pupil, right_bbox = detect_pupil(right_eye)
        
        # Draw bounding boxes and pupils
        cv2.rectangle(frame, (left_eye_rect[0], left_eye_rect[1]), 
                      (left_eye_rect[0] + left_eye_rect[2], left_eye_rect[1] + left_eye_rect[3]), (0, 255, 0), 2)
        cv2.rectangle(frame, (right_eye_rect[0], right_eye_rect[1]), 
                      (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]), (0, 255, 0), 2)
        
        if left_pupil and left_bbox:
            cv2.circle(frame, (left_eye_rect[0] + left_pupil[0], left_eye_rect[1] + left_pupil[1]), 5, (0, 0, 255), -1)
        if right_pupil and right_bbox:
            cv2.circle(frame, (right_eye_rect[0] + right_pupil[0], right_eye_rect[1] + right_pupil[1]), 5, (0, 0, 255), -1)
        
        # Alternative gaze detection using eye landmarks directly (more reliable)
        # Calculate eye aspect ratios and positions
        left_eye_center = np.mean(left_eye_points, axis=0)
        right_eye_center = np.mean(right_eye_points, axis=0)

        # Get face center for reference
        face_center_x = (face.left() + face.right()) / 2
        face_center_y = (face.top() + face.bottom()) / 2

        # Calculate relative eye positions
        eyes_center_x = (left_eye_center[0] + right_eye_center[0]) / 2
        eyes_center_y = (left_eye_center[1] + right_eye_center[1]) / 2

        # Calculate offset from face center
        face_width = face.right() - face.left()
        face_height = face.bottom() - face.top()

        # Normalize offsets
        x_offset = (eyes_center_x - face_center_x) / face_width
        y_offset = (eyes_center_y - face_center_y) / face_height

        # More balanced and stable thresholds for gaze detection
        # Use different thresholds for different directions to account for natural eye position
        horizontal_threshold = 0.12
        vertical_threshold = 0.08

        # Prioritize horizontal movement over vertical (more reliable)
        if abs(x_offset) > abs(y_offset):
            if x_offset < -horizontal_threshold:
                gaze_direction = "Looking Left"
            elif x_offset > horizontal_threshold:
                gaze_direction = "Looking Right"
            else:
                gaze_direction = "Looking Center"
        else:
            if y_offset < -vertical_threshold:
                gaze_direction = "Looking Up"
            elif y_offset > vertical_threshold:
                gaze_direction = "Looking Down"
            else:
                gaze_direction = "Looking Center"

        # Fallback: Try pupil detection if available
        if left_pupil and right_pupil:
            lx, ly = left_pupil
            rx, ry = right_pupil

            eye_width = left_eye_rect[2]
            eye_height = left_eye_rect[3]

            # Normalize pupil positions
            norm_lx = lx / eye_width if eye_width > 0 else 0.5
            norm_rx = rx / eye_width if eye_width > 0 else 0.5
            norm_ly = ly / eye_height if eye_height > 0 else 0.5
            norm_ry = ry / eye_height if eye_height > 0 else 0.5

            # Average the normalized positions
            avg_x = (norm_lx + norm_rx) / 2
            avg_y = (norm_ly + norm_ry) / 2

            # Use pupil detection with very sensitive thresholds
            if avg_x < 0.35:  # Very sensitive for left
                gaze_direction = "Looking Left"
            elif avg_x > 0.65:  # Very sensitive for right
                gaze_direction = "Looking Right"
            elif avg_y < 0.35:  # Very sensitive for up
                gaze_direction = "Looking Up"
            elif avg_y > 0.65:  # Very sensitive for down
                gaze_direction = "Looking Down"

            # Add debug information
            cv2.putText(frame, f"Pupil: ({avg_x:.2f}, {avg_y:.2f})",
                       (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Add debug information for landmark-based detection
        cv2.putText(frame, f"Eyes: ({x_offset:.2f}, {y_offset:.2f})",
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Debug print for testing (enabled for troubleshooting)
        if DEBUG_MODE:
            print(f"DEBUG: Eye movement - x_offset: {x_offset:.3f}, y_offset: {y_offset:.3f}, direction: {gaze_direction}")

    # Add smoothing to reduce jitter
    gaze_history.append(gaze_direction)

    # Use majority vote from recent history for more stable detection
    if len(gaze_history) >= 3:
        # Count occurrences of each direction
        direction_counts = {}
        for direction in gaze_history:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1

        # Use the most common direction
        smoothed_direction = max(direction_counts, key=direction_counts.get)

        # Only change if the new direction appears at least twice in recent history
        if direction_counts.get(smoothed_direction, 0) >= 2:
            gaze_direction = smoothed_direction

    return frame, gaze_direction
