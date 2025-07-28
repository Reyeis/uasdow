import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Path ke dataset waifu
DATASET_DIR = "dataset/waifu_dataset"  # Ubah ini sesuai folder dataset waifumu

IMG_SIZE = 100

X = []
y = []

print("📂 Memuat gambar dari folder:", DATASET_DIR)
for waifu_name in os.listdir(DATASET_DIR):
    waifu_path = os.path.join(DATASET_DIR, waifu_name)
    if not os.path.isdir(waifu_path):
        continue
    for img_name in os.listdir(waifu_path):
        img_path = os.path.join(waifu_path, img_name)

        # Cek hanya file gambar tertentu
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Tambahan: cek apakah gambar berhasil dibaca
            if img is None:
                print(f"[SKIP] File tidak bisa dibaca (mungkin corrupt): {img_path}")
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(waifu_name)

        except Exception as e:
            print(f"[WARNING] Gagal memuat {img_path}: {e}")

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0

# Encode label
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Bagi data
X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Buat model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("🚀 Melatih model waifu matcher...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=8)

# Simpan model dan label
print("💾 Menyimpan model ke 'waifu_match_model.h5'")
model.save("waifu_match_model.h5")

print("💾 Menyimpan label ke 'waifu_label_encoder_classes.npy'")
np.save("waifu_label_encoder_classes.npy", le.classes_)

print("✅ Pelatihan selesai. Model waifu siap digunakan!")
