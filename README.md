##  Waifu Matcher 

Aplikasi **real-time face recognition** yang mencocokkan wajahmu dengan karakter waifu dari berbagai anime menggunakan model deep learning (CNN).
Dilengkapi dengan skor confidence dan alasan personal mengapa karakter tersebut cocok denganmu secara visual dan kepribadian.


## 📌 Fitur Utama

* Deteksi wajah real-time dari webcam
* Pencocokan wajah ke karakter waifu dari lebih dari **100+ karakter anime**
* Skor kuantitatif (confidence) sebagai dasar ilmiah kecocokan
* Penjelasan alasan cocok dengan waifu secara personal
* Tampilan UI elegan, tema dark/light, dan responsif
* Disertai *source code lengkap* dan mudah dijalankan

---

## 🧠 Penjelasan Ilmiah

### 🔍 Dasar Kecocokan Wajah

Sistem menggunakan **Convolutional Neural Network (CNN)** untuk mengekstrak **embedding wajah**. Embedding ini merepresentasikan struktur wajah seperti:

* bentuk mata, hidung, dan bibir
* kontur wajah
* rasio dimensi wajah

> Model kemudian menghitung kemiripan embedding user dengan dataset waifu, lalu mengembalikan karakter dengan skor tertinggi (confidence score).

### 📊 Dasar Kuantitatif

Nilai **confidence (%)** menunjukkan seberapa mirip wajah pengguna dengan karakter tertentu secara numerik berdasarkan hasil softmax prediksi model.

Contoh output:

```python
{"match": "asuna_(sao)", "confidence": "92.38%", "reason": "Kamu punya aura pemimpin yang lembut dan setia, cocok banget sama Asuna."}
```

---

## 🧪 Teknologi yang Digunakan

| Teknologi           | Deskripsi                               |
| ------------------- | --------------------------------------- |
| Python & Flask      | Web framework backend                   |
| OpenCV              | Deteksi wajah real-time                 |
| TensorFlow / Keras  | Model deep learning untuk face matching |
| HTML + CSS + JS     | Frontend responsif                      |
| Font Awesome        | Ikon dan antarmuka modern               |
| SpeechSynthesis API | Text-to-speech langsung di browser      |

---

## 🗂️ Struktur Proyek

| Path                                       | Deskripsi                                                          |
| ------------------------------------------ | ------------------------------------------------------------------ |
| waifu-matcher-ai/                          | Direktori utama proyek                                             |
| ├── app.py                                 | File utama Flask untuk menjalankan server dan logika deteksi       |
| ├── templates/                             | Folder untuk file HTML (template Flask)                            |
| │   └── index.html                         | Tampilan utama aplikasi web                                        |
| ├── static/                                | Folder berisi file statis seperti gambar/foto yang tidak berubah   |
| │   └── unknown\_faces/                    | Folder untuk menyimpan wajah pengguna yang tidak dikenal/dideteksi |
| ├── model/                                 | Folder model machine learning                                      |
| │   ├── face\_recognition\_model.h5        | Model CNN untuk mengenali wajah                                    |
| │   ├── waifu\_match\_model.h5             | Model klasifikasi untuk prediksi waifu                             |
| │   ├── label\_encoder\_classes.npy        | File LabelEncoder untuk kelas wajah pengguna                       |
| │   └── waifu\_label\_encoder\_classes.npy | LabelEncoder untuk label waifu                                     |
| ├── waifu\_reasons.py                      | Berisi mapping alasan kenapa wajah cocok dengan waifu tertentu     |
| └── README.md                              | Dokumentasi proyek dan petunjuk penggunaan                         |



## 🚀 Cara Menjalankan Aplikasi

### 1. Clone Repository

```bash
git clone https://github.com/Reyeis/uasdow.git
cd waifu-matcher-ai
```

### 2. Install Dependensi

bash
pip install -r requirements.txt


### 3. Jalankan Aplikasi

bash
python app.py


Lalu buka browser dan akses `http://localhost:5000`

---

## 🧠 Contoh Output (Hasil Prediksi)

```
Waifu kamu dari tampangmu adalah:
💘 Albedo
(94.57%)

Kenapa cocok? → Tatapanmu tajam dan tenang, cocok dengan Albedo yang setia dan elegan.
```

---

## 🔗 Link Tugas (GitHub Repo)

👉 [https://github.com/namakamu/waifu-matcher-ai](https://github.com/Reyeis/uasdow)

---

## 👨‍🏫 Pengembang

Rais Rasyad Shidiq – [@Reyeis](https://github.com/Reyeis)
