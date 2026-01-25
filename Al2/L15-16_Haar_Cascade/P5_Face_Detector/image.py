# Phase 1: Import OpenCV
import cv2

# Phase 2: Load trained data
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Phase 3: Image Processing for Detection
# Read image
img = cv2.imread("outing.jpg")
if img is None:
    raise FileNotFoundError("outing.jpg not found.")

# resize image
img = cv2.resize(img, (800, 600))

# Change image to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Phase 4: Face Detection
# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# To draw rectangle for each face
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Phase 5: Show image
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()