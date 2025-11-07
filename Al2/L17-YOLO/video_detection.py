from ultralytics import YOLO
import cv2
import math

# Load model
model = YOLO('yolov8n.pt')
class_name = model.names

# Load video
video = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    # Detect objects
    results = model(frame)
    
    # Draw detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
            
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            label = f'{class_name[cls]}{conf}'
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
