from ultralytics import YOLO
import cv2
import math
import os

print(os.getcwd())

# Load model
model = YOLO('yolov8n.pt')
class_name = model.names

# Load image
# img = cv2.imread(r'C:\Users\benlo\telebort\Al2\YOLO\outing.jpg')
img = cv2.imread('outing.jpg')
img = cv2.resize(img, (1000, 720))

# Detect objects
results = model(img)

# Draw detections
for r in results:
    for box in r.boxes:
        # Bounding box
        x1, y1, x2, y2 = box.xyxy[0]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
        
        # Confidence and class name
        conf = math.ceil((box.conf[0]*100))/100
        cls = int(box.cls[0])
        
        # Label
        label = f'{class_name[cls]}{conf}'
        cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display image
cv2.imshow('Image', img)
cv2.waitKey(0)
