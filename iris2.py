import cv2
import numpy as np

# Load the image
image_path = 'nobiprofile.jpg'  # Change this to your image path
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    # Select the region of interest which is the face area
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]

    # Detect eyes in the face region
    eyes = eye_cascade.detectMultiScale(roi_gray)

    for (ex, ey, ew, eh) in eyes:
        # Select the region of interest which is the eye area
        eye_roi_color = roi_color[ey:ey + eh, ex:ex + ew]

        # Detect iris using HoughCircles
        eye_gray = cv2.cvtColor(eye_roi_color, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(eye_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0,
                                   maxRadius=0)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Draw the circle around the iris
                cv2.circle(eye_roi_color, (x, y), r, (0, 255, 0), 2)

# Display the result
cv2.imshow('Iris Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
