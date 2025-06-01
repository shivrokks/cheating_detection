import cv2
import torch
from ultralytics import YOLO
import cv2

# Load trained YOLO model
model = YOLO("./model/best_yolov12.pt")

# Determine device - explicitly use discrete GPU (GPU 0)
if torch.cuda.is_available():
    device = "cuda:0"  # Force use of GPU 0 (discrete AMD RX 6800M)
    print(f"Mobile Detection: Using discrete GPU - {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("Mobile Detection: CUDA not available, using CPU")

print(f"Mobile Detection: Using device: {device}")
if device.startswith("cuda"):
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Enable aggressive optimizations for maximum performance with 12GB VRAM
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # Allow more aggressive memory usage patterns
    torch.backends.cudnn.allow_tf32 = True
    # Initial cache clear only
    torch.cuda.empty_cache()
model.to(device)

def process_mobile_detection(frame):
    try:
        # Use torch.no_grad() to prevent memory accumulation
        with torch.no_grad():
            # No frame resizing - use full resolution for optimal detection accuracy
            frame_resized = frame
            height, width = frame.shape[:2]
            
            # Run inference with high resolution - utilize full VRAM capacity
            # Increased imgsz from 320 to 1280 for better accuracy with 12GB VRAM
            results = model(frame_resized, verbose=False, device=device, imgsz=1280, half=False)
            mobile_detected = False

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())

                        if conf < 0.6 or cls != 0:  # Lowered threshold for better detection with high-res input
                            continue

                        # No coordinate scaling needed - using full resolution
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        label = f"Mobile ({conf:.2f})"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        mobile_detected = True
            
            # No need for frequent cache clearing with 12GB VRAM available
                
        return frame, mobile_detected
    except Exception as e:
        print(f"Error in mobile detection: {e}")
        # Only clear GPU memory on actual errors
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return frame, False
