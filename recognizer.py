import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import os
from datetime import datetime

# Load model dan label
model = load_model('trained_model/face_model.h5')
with open('trained_model/label_names.json') as f:
    label_names = json.load(f)

# Load classifier wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Siapkan webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Gagal membuka kamera.")
    exit()

# Siapkan folder unknown
if not os.path.exists('unknown_faces'):
    os.makedirs('unknown_faces')

# Loop pengenalan wajah real-time
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame gagal diambil dari kamera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img_resized = cv2.resize(face_img, (100, 100))
        input_img = face_img_resized.reshape(1, 100, 100, 1) / 255.0

        pred = model.predict(input_img)[0]
        confidence = np.max(pred) * 100
        label_index = np.argmax(pred)

        if confidence > 80:
            label = label_names[label_index]
            text = f"{label} ({confidence:.2f}%)"
        else:
            label = "Unknown"
            text = label
            # Simpan wajah tak dikenal
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            cv2.imwrite(f'unknown_faces/face_{timestamp}.jpg', face_img_resized)

        # Gambar kotak dan label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
