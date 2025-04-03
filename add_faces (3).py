import cv2
import pickle
import numpy as np
import os

# Create 'data' directory if it doesn't exist
if not os.path.exists('data/'):
    os.makedirs('data/')

# Initialize video capture
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces_data = []
i = 0
name = input("Enter your Aadhar number: ")
framesTotal = 51
captureAfterFrame = 6

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Could not access webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50))

        # Ensure we don't capture more than 'framesTotal' images
        if len(faces_data) < framesTotal and i % captureAfterFrame == 0:
            faces_data.append(resized_img)

        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) >= framesTotal:
        break

video.release()
cv2.destroyAllWindows()

# Convert captured face data to numpy array
faces_data = np.asarray(faces_data)

# Ensure reshaping works dynamically
faces_data = faces_data.reshape((len(faces_data), -1))
print(faces_data)

# ---- HANDLE 'names.pkl' FILE ----
names_file = 'data/names.pkl'

if not os.path.exists(names_file) or os.stat(names_file).st_size == 0:
    names = [name] * len(faces_data)
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(names_file, 'rb') as f:
        try:
            names = pickle.load(f)
        except EOFError:
            names = []

    names += [name] * len(faces_data)

    with open(names_file, 'wb') as f:
        pickle.dump(names, f)

# ---- HANDLE 'faces_data.pkl' FILE ----
faces_file = 'data/faces_data.pkl'

if not os.path.exists(faces_file) or os.stat(faces_file).st_size == 0:
    with open(faces_file, 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open(faces_file, 'rb') as f:
        try:
            faces = pickle.load(f)
        except EOFError:
            faces = np.empty((0, faces_data.shape[1]))

    faces = np.append(faces, faces_data, axis=0)

    with open(faces_file, 'wb') as f:
        pickle.dump(faces, f)
