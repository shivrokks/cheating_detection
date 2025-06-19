import cv2
import numpy as np
import time

# Initialize HOG detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# YOLO model for person detection
yolo_model = None
USE_YOLO_PERSON = False

# Try to load YOLO model for person detection
try:
    import torch
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model = YOLO("yolov8n.pt")  # Use standard YOLOv8 nano model
    yolo_model.to(device)
    USE_YOLO_PERSON = True
    print("✓ YOLO person detection model loaded successfully")
except Exception as e:
    print(f"YOLO person detection not available: {e}")
    USE_YOLO_PERSON = False

# Constants for person detection
HOG_CONFIDENCE_THRESHOLD = 0.4  # Increased to reduce false positives
YOLO_CONFIDENCE_THRESHOLD = 0.3  # YOLO confidence threshold
FACE_MIN_SIZE = (40, 40)  # Minimum face size
HISTORY_SIZE = 5  # Reduced for faster response
PERSON_ENTRY_THRESHOLD = 2  # Frames needed to confirm new person
MIN_DETECTION_FRAMES = 2  # Minimum frames for stable detection
NMS_THRESHOLD = 0.4  # Non-maximum suppression threshold

# Initialize variables
person_count_history = []
last_person_entry_time = 0
new_person_frames = 0
previous_person_count = 0

def apply_nms_to_detections(detections, overlap_threshold=0.4):
    """Apply Non-Maximum Suppression to remove overlapping detections"""
    if len(detections) == 0:
        return []

    # Convert to format needed for NMS
    boxes = []
    scores = []

    for detection in detections:
        x, y, w, h = detection['bbox']
        boxes.append([x, y, x + w, y + h])
        scores.append(detection['confidence'])

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.1, overlap_threshold)

    # Return filtered detections
    filtered_detections = []
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            filtered_detections.append(detections[i])

    return filtered_detections

def detect_people_yolo(frame):
    """Detect people using YOLO model"""
    if not USE_YOLO_PERSON or yolo_model is None:
        return []

    try:
        results = yolo_model(frame, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
        detections = []

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())

                    # Class 0 is 'person' in COCO dataset
                    if cls == 0 and conf >= YOLO_CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append({
                            'bbox': (x1, y1, x2-x1, y2-y1),
                            'confidence': conf,
                            'method': 'YOLO'
                        })

        return detections
    except Exception as e:
        print(f"YOLO person detection error: {e}")
        return []

def detect_people_hog(frame):
    """Detect people using HOG descriptor"""
    detections = []

    try:
        # Resize frame for HOG detection
        frame_resized = cv2.resize(frame, (640, 480))
        height_ratio = frame.shape[0] / 480
        width_ratio = frame.shape[1] / 640

        # Standard HOG detection
        try:
            boxes, weights = hog.detectMultiScale(
                frame_resized,
                winStride=(8, 8),
                padding=(16, 16),
                scale=1.05
            )

            for i, box in enumerate(boxes):
                if weights[i] > HOG_CONFIDENCE_THRESHOLD:
                    x, y, w, h = box
                    # Scale back to original frame size
                    x = int(x * width_ratio)
                    y = int(y * height_ratio)
                    w = int(w * width_ratio)
                    h = int(h * height_ratio)

                    detections.append({
                        'bbox': (x, y, w, h),
                        'confidence': weights[i],
                        'method': 'HOG'
                    })
        except Exception as e:
            print(f"HOG detection error: {e}")

    except Exception as e:
        print(f"Error in HOG person detection: {e}")

    return detections

def detect_faces(frame):
    """Detect faces as a fallback for person detection"""
    detections = []

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if not face_cascade.empty():
            # Standard face detection
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=FACE_MIN_SIZE)

            # If no faces found, try more aggressive parameters
            if len(faces) == 0:
                faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))

            for (x, y, w, h) in faces:
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': 0.8,  # Assume good confidence for face detection
                    'method': 'Face'
                })

    except Exception as e:
        print(f"Error in face detection: {e}")

    return detections

def process_person_detection(frame):
    """Process the frame to detect and count people with improved detection"""
    global person_count_history, last_person_entry_time, new_person_frames, previous_person_count

    all_detections = []
    detection_method = "none"

    # Method 1: YOLO person detection (primary method)
    if USE_YOLO_PERSON:
        yolo_detections = detect_people_yolo(frame)
        all_detections.extend(yolo_detections)
        if len(yolo_detections) > 0:
            detection_method = "YOLO"

    # Method 2: HOG detection (fallback if YOLO finds nothing or not available)
    if len(all_detections) == 0 or not USE_YOLO_PERSON:
        hog_detections = detect_people_hog(frame)
        all_detections.extend(hog_detections)
        if len(hog_detections) > 0:
            detection_method = "HOG" if detection_method == "none" else detection_method + "+HOG"

    # Method 3: Face detection (final fallback)
    if len(all_detections) == 0:
        face_detections = detect_faces(frame)
        all_detections.extend(face_detections)
        if len(face_detections) > 0:
            detection_method = "Face" if detection_method == "none" else detection_method + "+Face"

    # Apply Non-Maximum Suppression to remove overlapping detections
    filtered_detections = apply_nms_to_detections(all_detections, NMS_THRESHOLD)

    # Count people and draw detections
    person_count = len(filtered_detections)

    # Draw all filtered detections
    for i, detection in enumerate(filtered_detections):
        x, y, w, h = detection['bbox']
        conf = detection['confidence']
        method = detection['method']

        # Use different colors for different methods
        if method == 'YOLO':
            color = (0, 255, 0)  # Green
        elif method == 'HOG':
            color = (255, 0, 0)  # Blue
        else:  # Face
            color = (0, 0, 255)  # Red

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{method} {i + 1} ({conf:.2f})", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Ensure at least 1 person if we detect any faces (common case for video calls)
    if person_count == 0 and any(d['method'] == 'Face' for d in all_detections):
        person_count = 1
        detection_method = "Face-Fallback"

    # Update person count history
    person_count_history.append(person_count)
    if len(person_count_history) > HISTORY_SIZE:
        person_count_history.pop(0)

    # Calculate stable person count using improved smoothing
    if len(person_count_history) >= 3:
        # Use mode (most frequent value) for stability, fallback to median
        from collections import Counter
        count_freq = Counter(person_count_history)
        most_common = count_freq.most_common(1)[0]

        # If the most common value appears at least 40% of the time, use it
        if most_common[1] >= max(2, len(person_count_history) * 0.4):
            avg_person_count = most_common[0]
        else:
            # Otherwise use median for stability
            avg_person_count = int(np.median(person_count_history))
    else:
        avg_person_count = person_count

    # Clamp person count to reasonable range (0-4 people max for typical exam scenarios)
    avg_person_count = max(0, min(avg_person_count, 4))

    # Ensure at least 1 person if we consistently detect someone
    if avg_person_count == 0 and previous_person_count >= 1:
        # If we've seen a person in the last few frames, assume they're still there
        recent_detections = sum(1 for count in person_count_history[-3:] if count > 0)
        if recent_detections >= 2:  # Need at least 2 recent detections
            avg_person_count = 1

    # Detect if a new person entered the frame
    new_person_entered = False
    if avg_person_count > previous_person_count:
        new_person_frames += 1
        if new_person_frames >= PERSON_ENTRY_THRESHOLD:
            new_person_entered = True
            last_person_entry_time = time.time()
    else:
        new_person_frames = 0

    # Update previous person count
    previous_person_count = avg_person_count

    # Display person count and detection method on frame
    cv2.putText(frame, f"People: {avg_person_count} ({detection_method})", (20, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display warning if multiple people detected
    multiple_people = avg_person_count > 1
    if multiple_people:
        cv2.putText(frame, "WARNING: Multiple people detected!", (20, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display warning if new person entered
    if new_person_entered or (time.time() - last_person_entry_time < 3.0 and avg_person_count > 1):
        cv2.putText(frame, "WARNING: New person entered the frame!", (20, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Debug print for server logs
    if person_count != avg_person_count:
        print(f"Person detection: raw={person_count}, smoothed={avg_person_count}, method={detection_method}")

    return frame, avg_person_count, multiple_people, new_person_entered

