# Import YOLO
from ultralytics import YOLO
import os

# Fix: correct the model path to match the actual folder structure
model_path = "content/runs/detect/train/weights/best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please check the path and filename.")
model = YOLO(model_path)

# Import OpenCV and Math
import cv2
import math

# Load the model's labels and store them as class_names
class_names = model.names

# Load image from 'image1.jpg' and store it as img
img = cv2.imread('image1.jpg')

# Resize the image to dimension 1280x720
img = cv2.resize(img, (1280, 720))

# Detect objects in the image and store them as results
results = model(img)

# Draw the bounding boxes and labels on the image
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
        conf = math.ceil((box.conf[0]*100))/100
        cls = int(box.cls[0])
        label = f'{class_names[cls]} {conf}'
        cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Show the image
cv2.imshow('Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
