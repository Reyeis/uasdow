from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from datetime import datetime
import os

from waifu_reasons import waifu_reasons  # ✅ Import alasan waifu

app = Flask(__name__)

# Load model umum & label
model = load_model('face_recognition_model.h5')
labels = np.load('label_encoder_classes.npy')

# Load model waifu matcher & label
waifu_model = load_model('waifu_match_model.h5')
waifu_labels = np.load('waifu_label_encoder_classes.npy')

IMG_SIZE = 100
CONFIDENCE_THRESHOLD = 0.80
camera_active = False

# Buat folder penyimpanan wajah tak dikenal
os.makedirs('static/unknown_faces', exist_ok=True)

camera = cv2.VideoCapture(0)

def preprocess_face(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_img = face_img.astype("float32") / 255.0
    face_img = img_to_array(face_img)
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

def generate_frames():
    global camera_active
    while True:
        if not camera_active:
            continue

        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        ).detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            try:
                processed = preprocess_face(face_img)
                prediction = model.predict(processed, verbose=0)[0]
                max_index = np.argmax(prediction)
                confidence = prediction[max_index]

                if confidence >= CONFIDENCE_THRESHOLD:
                    name = labels[max_index]
                    label = f"{name} ({confidence*100:.2f}%)"
                else:
                    name = "Unknown"
                    label = f"Unknown ({confidence*100:.2f}%)"
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f'static/unknown_faces/face_{timestamp}.jpg'
                    cv2.imwrite(filename, cv2.resize(face_img, (100, 100)))

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            except Exception as e:
                print(f"[ERROR] Proses wajah gagal: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active
    camera_active = True
    return ('', 204)


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active
    camera_active = False
    return ('', 204)


@app.route('/match_waifu', methods=['POST'])
def match_waifu():
    success, frame = camera.read()
    if not success:
        return jsonify({"status": "error", "message": "Gagal ambil gambar"})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    ).detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return jsonify({"status": "error", "message": "Tidak ada wajah terdeteksi"})

    (x, y, w, h) = faces[0]
    face_img = frame[y:y+h, x:x+w]
    processed = preprocess_face(face_img)

    prediction = waifu_model.predict(processed, verbose=0)[0]
    max_index = np.argmax(prediction)
    confidence = prediction[max_index]
    best_match = waifu_labels[max_index]

    reason = waifu_reasons.get(best_match, "Waifu ini cocok karena punya vibes yang mirip sama kamu 😄.")

    return jsonify({
        "status": "success",
        "match": best_match,
        "confidence": f"{confidence * 100:.2f}%",
        "reason": reason
    })


if __name__ == '__main__':
    app.run(debug=True)
