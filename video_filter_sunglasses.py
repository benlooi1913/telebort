import cv2 as cv

# Load cascades
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Load sunglasses image (must have alpha channel)
sunglasses = cv.imread('sunglasses-eyewear.png', cv.IMREAD_UNCHANGED)
if sunglasses is None or sunglasses.shape[2] < 4:
    raise Exception("Sunglasses PNG with alpha channel not found.")

video = cv.VideoCapture(0)

while True:
    success, frame = video.read()
    if not success:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda ex: ex[0])
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[1]
            # Calculate eye centers relative to full frame
            eye_center_x = x + int((ex1 + ex2 + ew1 // 2 + ew2 // 2) / 2)
            eye_center_y = y + int((ey1 + ey2 + eh1 // 2 + eh2 // 2) / 2)
            # Sunglasses size and position
            sg_width = int(abs((ex2 + ew2 // 2) - (ex1 + ew1 // 2)) * 2)
            sg_height = int(sg_width * sunglasses.shape[0] / sunglasses.shape[1])
            x1 = eye_center_x - sg_width // 2
            y1 = eye_center_y - sg_height // 2
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x1 + sg_width, frame.shape[1])
            y2 = min(y1 + sg_height, frame.shape[0])
            sg_resized = cv.resize(sunglasses, (x2 - x1, y2 - y1))
            sg_rgb = sg_resized[:, :, :3]
            sg_alpha = sg_resized[:, :, 3] / 255.0
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                    sg_rgb[:, :, c] * sg_alpha +
                    frame[y1:y2, x1:x2, c] * (1 - sg_alpha)
                ).astype('uint8')

    cv.imshow('Live Sunglasses Filter', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()