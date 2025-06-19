import cv2
import numpy as np
import os

# Global variables for detection
model = None
MOBILE_CLASS_ID = 67  # COCO dataset cell phone class
USE_YOLO = False

# Try to load YOLO model (optional)
try:
    import torch
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try custom model first
    custom_model_path = "./model/best_yolov12.pt"
    if os.path.exists(custom_model_path):
        print(f"Loading custom mobile detection model: {custom_model_path}")
        model = YOLO(custom_model_path)
        model.to(device)
        MOBILE_CLASS_ID = 0  # Assuming mobile is class 0 in custom model
        USE_YOLO = True
        print("✓ Custom mobile detection model loaded successfully")
    else:
        print(f"Custom model not found, trying standard YOLOv8...")
        try:
            model = YOLO("yolov8n.pt")  # Use standard YOLOv8 nano model
            model.to(device)
            MOBILE_CLASS_ID = 67  # COCO dataset cell phone class
            USE_YOLO = True
            print("✓ Standard YOLOv8 model loaded for mobile detection")
        except Exception as e2:
            print(f"Standard YOLO model failed: {e2}")
            USE_YOLO = False

except Exception as e:
    print(f"YOLO not available: {e}")
    USE_YOLO = False

if not USE_YOLO:
    print("Using alternative mobile detection methods (no YOLO)")

def detect_rectangular_objects(frame):
    """Detect rectangular objects that could be mobile phones"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mobile_candidates = []

    for contour in contours:
        # Filter by area
        area = cv2.contourArea(contour)
        if area < 1000 or area > 50000:  # Reasonable size range for mobile phones
            continue

        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if it's roughly rectangular (4 corners)
        if len(approx) >= 4:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Check aspect ratio (mobile phones are typically tall rectangles)
            aspect_ratio = h / w if w > 0 else 0

            # Mobile phones typically have aspect ratio between 1.5 and 2.5
            if 1.3 <= aspect_ratio <= 3.0:
                # Check if the contour fills the bounding rectangle reasonably well
                rect_area = w * h
                fill_ratio = area / rect_area if rect_area > 0 else 0

                if fill_ratio > 0.6:  # At least 60% filled
                    mobile_candidates.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'fill_ratio': fill_ratio,
                        'confidence': min(fill_ratio * aspect_ratio / 2.0, 1.0)
                    })

    return mobile_candidates

def detect_bright_screens(frame):
    """Detect bright rectangular areas that could be phone screens"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply threshold to find bright areas
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    screen_candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500 or area > 30000:  # Screen size range
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w if w > 0 else 0

        # Screens can be in portrait or landscape
        if 0.5 <= aspect_ratio <= 3.0:
            screen_candidates.append({
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'confidence': min(area / 10000, 1.0)
            })

    return screen_candidates

def process_mobile_detection(frame):
    """
    Detect mobile phones in the frame using multiple methods
    Returns: (processed_frame, mobile_detected_boolean)
    """
    mobile_detected = False
    detection_method = "none"
    all_detections = []

    # Method 1: YOLO detection (if available)
    if USE_YOLO and model is not None:
        try:
            results = model(frame, verbose=False, conf=0.2)

            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())

                        # Get class name if available
                        class_name = "unknown"
                        if hasattr(result, 'names') and cls in result.names:
                            class_name = result.names[cls]

                        # Check if this is a mobile phone detection
                        is_mobile = False

                        if cls == MOBILE_CLASS_ID:
                            is_mobile = True
                        elif class_name.lower() in ['cell phone', 'mobile', 'phone', 'smartphone']:
                            is_mobile = True

                        if is_mobile and conf >= 0.25:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            all_detections.append({
                                'bbox': (x1, y1, x2-x1, y2-y1),
                                'confidence': conf,
                                'method': 'YOLO',
                                'label': f"YOLO: {class_name}"
                            })
                            mobile_detected = True
                            detection_method = "YOLO"
                            print(f"✓ YOLO mobile detection: {class_name}, conf={conf:.3f}")
        except Exception as e:
            print(f"YOLO detection error: {e}")

    # Method 2: Rectangular object detection
    try:
        rect_candidates = detect_rectangular_objects(frame)
        for candidate in rect_candidates:
            if candidate['confidence'] > 0.4:  # Threshold for rectangular detection
                all_detections.append({
                    'bbox': candidate['bbox'],
                    'confidence': candidate['confidence'],
                    'method': 'Rectangle',
                    'label': f"Rect: AR={candidate['aspect_ratio']:.1f}"
                })
                mobile_detected = True
                if detection_method == "none":
                    detection_method = "Rectangle"
                print(f"✓ Rectangular mobile detection: conf={candidate['confidence']:.3f}")
    except Exception as e:
        print(f"Rectangular detection error: {e}")

    # Method 3: Bright screen detection
    try:
        screen_candidates = detect_bright_screens(frame)
        for candidate in screen_candidates:
            if candidate['confidence'] > 0.3:  # Threshold for screen detection
                all_detections.append({
                    'bbox': candidate['bbox'],
                    'confidence': candidate['confidence'],
                    'method': 'Screen',
                    'label': f"Screen: {candidate['area']}"
                })
                mobile_detected = True
                if detection_method == "none":
                    detection_method = "Screen"
                print(f"✓ Screen mobile detection: conf={candidate['confidence']:.3f}")
    except Exception as e:
        print(f"Screen detection error: {e}")

    # Draw all detections
    for i, detection in enumerate(all_detections):
        x, y, w, h = detection['bbox']
        conf = detection['confidence']
        method = detection['method']
        label = detection['label']

        # Use different colors for different methods
        if method == 'YOLO':
            color = (0, 0, 255)  # Red
        elif method == 'Rectangle':
            color = (255, 0, 0)  # Blue
        else:  # Screen
            color = (0, 255, 255)  # Yellow

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Add warning if mobile detected
    if mobile_detected:
        cv2.putText(frame, f"WARNING: Mobile Detected ({detection_method})!", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Detections: {len(all_detections)}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame, mobile_detected

def test_mobile_detection():
    """Test mobile detection with webcam"""
    print("="*60)
    print("MOBILE DETECTION TEST")
    print("="*60)
    print("Available detection methods:")
    print(f"  - YOLO: {'✓' if USE_YOLO else '✗'}")
    print(f"  - Rectangular detection: ✓")
    print(f"  - Screen detection: ✓")
    print()
    print("Instructions:")
    print("1. Hold up a mobile phone to test detection")
    print("2. Try different angles and distances")
    print("3. Test with screen on/off")
    print("4. Press 'q' to quit")
    print("="*60)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    frame_count = 0
    detections = 0
    method_counts = {'YOLO': 0, 'Rectangle': 0, 'Screen': 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        processed_frame, mobile_detected = process_mobile_detection(frame)

        if mobile_detected:
            detections += 1

        # Show statistics
        detection_rate = (detections / frame_count) * 100
        cv2.putText(processed_frame, f"Detection rate: {detection_rate:.1f}%", (20, 400),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(processed_frame, f"Frame: {frame_count}, Detections: {detections}", (20, 430),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show method availability
        y_offset = 460
        cv2.putText(processed_frame, f"YOLO: {'ON' if USE_YOLO else 'OFF'}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if USE_YOLO else (0, 0, 255), 2)

        cv2.imshow('Mobile Detection Test', processed_frame)

        # Print detection info every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: Detection rate = {detection_rate:.1f}%")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*60)
    print("TEST RESULTS:")
    print("="*60)
    print(f"Total frames: {frame_count}")
    print(f"Mobile detections: {detections}")
    print(f"Detection rate: {detection_rate:.1f}%")
    print(f"Methods used: {', '.join([m for m in ['YOLO', 'Rectangle', 'Screen'] if (m == 'YOLO' and USE_YOLO) or m != 'YOLO'])}")

    if detection_rate > 50:
        print("✅ EXCELLENT: High detection rate!")
    elif detection_rate > 20:
        print("⚠️  GOOD: Decent detection rate")
    else:
        print("❌ POOR: Low detection rate - check lighting and phone visibility")

    print("="*60)

if __name__ == "__main__":
    test_mobile_detection()
