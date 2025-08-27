from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load YOLOv8 model (pretrained on COCO dataset)
model = YOLO("yolov8n.pt")

def detect_objects(pil_img):
    results = model.predict(pil_img)
    detections = []
    img = np.array(pil_img)
    annotated_img = img.copy()

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id]

        detections.append({"label": label, "confidence": conf, "bbox": [x1, y1, x2, y2]})

        # Draw bounding boxes
        cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(annotated_img, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    return detections, Image.fromarray(annotated_img)
