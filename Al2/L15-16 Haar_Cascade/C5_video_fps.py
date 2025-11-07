# Phase 1: Import OpenCV
import cv2 as cv
import time
import os

face_cascade_path = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
local_fallback = r'c:\Users\benlo\telebort\Al2\L15-16 Haar_Cascade\haarcascade_frontalface_default.xml'
if not os.path.exists(face_cascade_path) and os.path.exists(local_fallback):
    face_cascade_path = local_fallback

trained_face_data = cv.CascadeClassifier(face_cascade_path)
if trained_face_data.empty():
    raise FileNotFoundError(f"Can't open cascade file for faces: {face_cascade_path}")

trained_smile_data = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_smile.xml')

# Start the webcam (0 means default camera)
video = cv.VideoCapture(0)

# Add before the loop
fps_start_time = time.time()
fps_counter = 0
fps = 0  # To display FPS

while True:
    # Capture a single frame from the video
    success, frame = video.read()
    
    # Check if frame was captured successfully
    if not success:
        break

    # Convert frame to grayscale for face detection
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces in the current frame
    face_coordinates = trained_face_data.detectMultiScale(gray_img)

    # Display number of faces
    num_faces = len(face_coordinates)
    cv.putText(frame, f'Faces: {num_faces}', (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # FPS calculation
    fps_counter += 1
    if time.time() - fps_start_time > 1:
        fps = fps_counter
        fps_counter = 0
        fps_start_time = time.time()
    cv.putText(frame, f'FPS: {fps}', (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Display the frame with face boxes
    cv.imshow('Live Face Detector', frame)

    # Check if 'Q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up when done
video.release()
cv.destroyAllWindows()
