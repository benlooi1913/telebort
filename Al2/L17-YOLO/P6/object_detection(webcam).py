# Import YOLO from ultralytics
from ultralytics import YOLO

# Import OpenCV and Math
import cv2
import math
import os

# Load rock-paper-scissors model
model_path = "content/runs/detect/train/weights/best.pt"  # Update if your model file is elsewhere
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please check the path and filename.")
model = YOLO(model_path)

# Load the model's labels and store them as class_names
class_names = model.names

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detector
    results = model(frame)

    # Draw bounding boxes
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            label = f'{class_names[cls]} {conf}'
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Webcam Detection', frame)

    # stop when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
