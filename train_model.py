import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Path ke dataset MIT Faces
DATASET_DIR = "dataset/"  # sesuaikan jika perlu

# Ukuran gambar
IMG_SIZE = 100

# List untuk data dan label
X = []
y = []

# Load gambar dan label
print("📂 Memuat gambar dari folder:", DATASET_DIR)
for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_path):
        continue
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(person_name)
        except Exception as e:
            print(f"[WARNING] Gagal memuat {img_path}: {e}")

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0

# Encode label
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Bagi data
X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Buat model CNN sederhana
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_categorical.shape[1], activation='softmax')  # jumlah output = jumlah kelas
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
print("🚀 Melatih model...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=8)

# Simpan model dan encoder
print("💾 Menyimpan model ke 'face_recognition_model.h5'")
model.save("face_recognition_model.h5")

print("💾 Menyimpan label encoder ke 'label_encoder.npy'")
np.save("label_encoder_classes.npy", le.classes_)

print("✅ Pelatihan selesai.")
