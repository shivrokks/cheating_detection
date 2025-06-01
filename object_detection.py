import cv2
import torch
from ultralytics import YOLO
import time
import os
import cv2

# Load pre-trained YOLO model for object detection
model_path = "./model/best_yolov12.pt" # Changed to best_yolov12.pt
model = YOLO(model_path)
print(f"Object detection model loaded from {model_path}")

# Determine device - explicitly use discrete GPU (GPU 0)
if torch.cuda.is_available():
    device = "cuda:0"  # Force use of GPU 0 (discrete AMD RX 6800M)
    print(f"Object detection: Using discrete GPU - {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("Object detection: CUDA not available, using CPU")

model.to(device)
print(f"Object detection model moved to {device}")
if device.startswith("cuda"):
    # Enable aggressive optimizations for maximum performance with 12GB VRAM
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # Allow more aggressive memory usage patterns
    torch.backends.cudnn.allow_tf32 = True
    # Initial cache clear only
    torch.cuda.empty_cache()
# if device == "cuda":
#     model.model.half()  # Enable FP16 if on CUDA
#     print("Object detection model set to FP16 (half precision)")

# Define suspicious objects for exams
SUSPICIOUS_OBJECTS = [
    "book", "cell phone", "laptop", "mouse", "remote", "keyboard", "tv", "person"
]

# Constants for object detection
CONFIDENCE_THRESHOLD = 0.3  # Lowered for better detection with high-res input
DETECTION_PERSISTENCE = 3.0  # How long to show a detection after it disappears (seconds)

# Initialize variables
last_detection_time = {}  # Track when each object was last detected

def process_object_detection(frame):
    """Process the frame to detect suspicious objects"""
    global last_detection_time
    
    try:
        # Use torch.no_grad() to prevent memory accumulation
        with torch.no_grad():
            # No frame resizing - use full resolution for optimal detection accuracy
            frame_resized = frame
            height, width = frame.shape[:2]
            
            # Run inference with high resolution - utilize full VRAM capacity
            # Increased imgsz from 320 to 1280 for better accuracy with 12GB VRAM
            results = model(frame_resized, verbose=False, device=device, imgsz=1280, half=False)
            
            # Track detected objects
            detected_objects = {}
            suspicious_objects_detected = False
            
            current_time = time.time()
        
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        cls_name = model.names[cls]
                        
                        # Skip if confidence is too low
                        if conf < CONFIDENCE_THRESHOLD:
                            continue
                        
                        # Check if this is a suspicious object
                        is_suspicious = cls_name.lower() in SUSPICIOUS_OBJECTS
                        
                        # Update detection time
                        if is_suspicious:
                            last_detection_time[cls_name] = current_time
                            suspicious_objects_detected = True
                        
                        # Add to detected objects
                        if cls_name in detected_objects:
                            detected_objects[cls_name] += 1
                        else:
                            detected_objects[cls_name] = 1
                        
                        # No coordinate scaling needed - using full resolution
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        color = (0, 0, 255) if is_suspicious else (0, 255, 0)
                        label = f"{cls_name} ({conf:.2f})"
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Check for recently detected objects that are no longer visible
            for obj, last_time in list(last_detection_time.items()):
                if current_time - last_time < DETECTION_PERSISTENCE:
                    if obj not in detected_objects:
                        # Object was recently detected but is no longer visible
                        cv2.putText(frame, f"Recently detected: {obj}", (20, 210), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        suspicious_objects_detected = True
                else:
                    # Remove old detections
                    last_detection_time.pop(obj)
            
            # Display warning if suspicious objects detected
            if suspicious_objects_detected:
                cv2.putText(frame, "WARNING: Suspicious objects detected!", (20, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # No need for frequent cache clearing with 12GB VRAM available
            
            return frame, suspicious_objects_detected, detected_objects
    
    except Exception as e:
        print(f"Error in object detection: {e}")
        # Only clear GPU memory on actual errors
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return frame, False, {}
