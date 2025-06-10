import cv2
import dlib
import numpy as np

# Load dlib’s face detector and 68 landmarks model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

# Load OpenCV face detector as fallback
opencv_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

    # Method 5: OpenCV detector with multiple parameter sets
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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Enhance image for better face detection
    enhanced_gray = cv2.equalizeHist(gray)

    # Use improved face detection
    faces = detect_faces_improved(enhanced_gray)
    gaze_direction = "Looking Center"

    # print(f"DEBUG: Eye movement - detected {len(faces)} faces")  # Commented out for production

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
        
        # Gaze Detection with improved thresholds
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

            # Improved gaze direction detection with better thresholds
            if avg_x < 0.35:
                gaze_direction = "Looking Left"
            elif avg_x > 0.65:
                gaze_direction = "Looking Right"
            elif avg_y < 0.35:
                gaze_direction = "Looking Up"
            elif avg_y > 0.65:
                gaze_direction = "Looking Down"
            else:
                gaze_direction = "Looking Center"

            # Add debug information
            cv2.putText(frame, f"Gaze: ({avg_x:.2f}, {avg_y:.2f})",
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Debug print for testing
            # print(f"DEBUG: Eye movement - avg_x: {avg_x:.3f}, avg_y: {avg_y:.3f}, direction: {gaze_direction}")  # Commented out for production
    
    return frame, gaze_direction
