# Touchless App Launcher 👋
Sistem pengenalan gesture tangan real-time untuk membuka aplikasi tanpa sentuhan fisik.

## Arsitektur Hybrid
- **MediaPipe** → Deteksi 21 landmark tangan + hitung jari (UTAMA)
- **CNN Model** → Validasi hasil MediaPipe (VALIDATOR)

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
├── Gestures.py              ← File utama (jalanin ini)
├── HandModule.py            ← Wrapper MediaPipe
├── train_model.py           ← Script training CNN
├── finger_count_model.h5    ← Model CNN hasil training
├── requirements.txt
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
```bash
pip install -r requirements.txt
```

## Cara Pakai
### 1. Training model 
```bash
python train_model.py
```

### 2. Jalanin aplikasi
```bash
python Gestures.py
```
Tekan `q` untuk keluar.

