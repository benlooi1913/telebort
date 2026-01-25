# Phase 1: Import OpenCV
import cv2

# Phase 2: Load trained data
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Phase 3: Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Webcam Face Detection', frame)

    # Press 'q' to quit and capture image
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Save the current frame as an image
        cv2.imwrite('captured_face.jpg', frame)
        break

cap.release()
cv2.destroyAllWindows()
