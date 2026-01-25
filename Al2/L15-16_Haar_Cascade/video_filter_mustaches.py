import cv2 as cv

# Load cascades
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Load mustache image (must have alpha channel). Fallback to drawing if missing.
mustache = cv.imread('mustache.png', cv.IMREAD_UNCHANGED)
use_image = True
if mustache is None or mustache.shape[2] < 4:
    use_image = False

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

            # eye centers (relative to full frame)
            eye1_cx = x + ex1 + ew1 // 2
            eye1_cy = y + ey1 + eh1 // 2
            eye2_cx = x + ex2 + ew2 // 2
            eye2_cy = y + ey2 + eh2 // 2
            eye_center_x = (eye1_cx + eye2_cx) // 2
            eye_center_y = (eye1_cy + eye2_cy) // 2

            # Estimate mustache size and position
            eye_dist = max(1, abs(eye2_cx - eye1_cx))
            ms_width = int(eye_dist * 2.0)            # width ~ 2x distance between eyes
            ms_height = int(ms_width * 0.35)         # aspect ratio for mustache
            ms_x1 = max(0, eye_center_x - ms_width // 2)
            ms_y1 = min(frame.shape[0]-1, eye_center_y + int(h * 0.12))  # slightly below eyes
            ms_x2 = min(frame.shape[1], ms_x1 + ms_width)
            ms_y2 = min(frame.shape[0], ms_y1 + ms_height)

            if use_image:
                ms_resized = cv.resize(mustache, (ms_x2 - ms_x1, ms_y2 - ms_y1), interpolation=cv.INTER_AREA)
                ms_rgb = ms_resized[:, :, :3]
                ms_alpha = ms_resized[:, :, 3] / 255.0
                for c in range(3):
                    frame[ms_y1:ms_y2, ms_x1:ms_x2, c] = (
                        ms_rgb[:, :, c] * ms_alpha +
                        frame[ms_y1:ms_y2, ms_x1:ms_x2, c] * (1 - ms_alpha)
                    ).astype('uint8')
            else:
                # simple fallback: draw a filled dark mustache (ellipse + rectangle)
                center = (eye_center_x, ms_y1 + ms_height // 2)
                axes = (ms_width // 2, ms_height // 2)
                cv.ellipse(frame, center, axes, 0, 0, 360, (40, 30, 30), -1)  # dark ellipse
                # slight rectangle under ellipse for a thicker look
                rect_y1 = center[1] - ms_height // 8
                rect_y2 = center[1] + ms_height // 4
                cv.rectangle(frame, (ms_x1 + ms_width//8, rect_y1), (ms_x2 - ms_width//8, rect_y2), (40,30,30), -1)

    cv.imshow('Live Mustache Filter', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()