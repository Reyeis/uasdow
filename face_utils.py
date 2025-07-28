import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
import os

def load_dataset(dataset_path, image_size=(100, 100)):
    images = []
    labels = []
    label_names = []

    label_map = {}

    for idx, folder in enumerate(sorted(os.listdir(dataset_path))):
        label_names.append(folder)
        folder_path = os.path.join(dataset_path, folder)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(idx)

    images = np.array(images).reshape(-1, image_size[0], image_size[1], 1) / 255.0
    labels = to_categorical(labels)

    return images, labels, label_names
