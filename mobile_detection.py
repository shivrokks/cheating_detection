import cv2
import torch
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("./model/best_yolov12.pt")
device = "cuda" if torch.cuda.is_available() else "cpu" # Reverted to allow GPU
model.to(device)

def process_mobile_detection(frame):
    try:
        results = model(frame, verbose=False)
        mobile_detected = False

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())

                    # Debug: print all detections with higher confidence (commented out for production)
                    # if conf > 0.3:  # Show detections above 30% confidence for debugging
                    #     print(f"Mobile detection: class={cls}, conf={conf:.3f}")

                    # Lowered threshold for better detection - mobile class index is 0
                    if conf < 0.5 or cls != 0:  # Reduced from 0.8 to 0.5 for better sensitivity
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"Mobile ({conf:.2f})"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    mobile_detected = True
                    # print(f"Mobile detected with confidence: {conf:.3f}")  # Commented out for production

        return frame, mobile_detected

    except Exception as e:
        print(f"Error in mobile detection: {e}")
        return frame, False
