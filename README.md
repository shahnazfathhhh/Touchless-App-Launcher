# Touchless App Launcher 👋
Sistem pengenalan gesture tangan real-time untuk membuka aplikasi tanpa sentuhan fisik.

## Arsitektur Hybrid
- **MediaPipe** → Deteksi 21 landmark tangan + hitung jari (UTAMA)
- **CNN Model** → Validasi hasil MediaPipe (VALIDATOR)
- **Web App** → Antarmuka web untuk menampilkan aplikasi secara interaktif
  
## Gesture
| Jari | Aksi |
|------|------|
| 1 | Tampilkan salam + tanggal & waktu |
| 2 | Buka ChatGPT |
| 3 | Buka YouTube |
| 4 | Buka Instagram |
| Lainnya | Tampilkan hitungan jari |

## Struktur Folder
```
project/
├── Gestures.py              # File utama untuk deteksi gesture
├── HandModule.py            # Wrapper MediaPipe
├── train_model.py           # Script training CNN
├── finger_count_model.h5    # Model CNN hasil training
├── requirements.txt
├── app.py                   # Backend web app
├── templates/
│   └── index.html           # Halaman utama web app
├── static/
│   ├── style.css            # Styling antarmuka web
│   └── script.js            # Logika interaksi pada web app
└── dataset/
    ├── train/
    │   ├── 0/
    │   ├── 1/
    │   ├── 2/
    │   ├── 3/
    │   ├── 4/
    │   └── 5/
    └── test/
        ├── 0/ ... 5/
```

## Cara Install

Project ini berhasil berjalan dengan versi Python 3.10+

```bash
pip install -r requirements.txt
```

## Cara Pakai
### 1. Training model 
Lakukan training model CNN terlebih dahulu jika model belum tersedia atau ingin diperbarui.
```bash
python train_model.py
```

### 2. Jalanin aplikasi
Jalankan aplikasi utama untuk mendeteksi jumlah jari menggunakan webcam.
```bash
python Gestures.py
```
Tekan `q` untuk keluar.

### 3. Menjalankan web app
Jalankan backend untuk mengakses antarmuka melalui browser.
```bash
python app.py
```

Setelah server berjalan, buka browser dan akses alamat lokal yang muncul pada terminal
```bash
http://127.0.0.1:5000/
```
