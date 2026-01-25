import cv2 as cv

# Use OpenCV's built-in haarcascade path
trained_face_data = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# cv.data.haarcascades gives the path to the haarcascades directory in OpenCV package
# Start the webcam (0 means default camera)
video = cv.VideoCapture(0)

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

    # Draw rectangles around detected faces
    for (x, y, w, h) in face_coordinates:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with face boxes
    cv.imshow('Live Face Detector', frame)

    # Check if 'Q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up when done
video.release()
cv.destroyAllWindows()
