"""
Gestures.py — Touchless App Launcher
Arsitektur Hybrid:
  - MediaPipe (via HandModule) → deteksi landmark + hitung jari (UTAMA)
  - CNN model                  → validasi hasil MediaPipe (VALIDATOR)

Gesture:
  1 jari  → Tampilkan salam + tanggal & waktu
  2 jari  → Buka ChatGPT
  3 jari  → Buka YouTube
  4 jari  → Buka Instagram
  Lainnya → Tampilkan hitungan jari
"""

import cv2
import HandModule as md
import math
import numpy as np
from datetime import date, datetime
import webbrowser
from tensorflow.keras.models import load_model

# ── Load CNN Model (validator) ────────────────────────────
# Memastikan file finger_count_model.h5 ada di folder yang sama
CNN_MODEL_PATH = 'finger_count_model.h5'
cnn_model      = load_model(CNN_MODEL_PATH)

# ── HandModule setup ──────────────────────────────────────
obj1 = md.doublehandDetector(detectionCon=0.7, trackCon=0.7)
cap  = cv2.VideoCapture(0)

# ── Cooldown ─────────────────
last_action      = ""
last_action_time = 0
COOLDOWN         = 3  # detik


# ═══════════════════════════════════════════════════════════
# MEDIAPIPE: Hitung jari pakai geometric landmark (UTAMA)
# ═══════════════════════════════════════════════════════════
def fingers_mediapipe(l):
    """Menghitung jari pakai rule-based geometric dari landmark MediaPipe."""
    if len(l) == 0:
        return 0
    total         = 0
    a             = [4, 8, 12, 16, 20]
    temp1         = []
    thumbdist     = math.hypot(l[0][1]-l[4][1],  l[0][2]-l[4][2])
    indexfindist  = math.hypot(l[4][1]-l[8][1],  l[4][2]-l[8][2])
    ringfindist   = math.hypot(l[4][1]-l[15][1], l[4][2]-l[15][2])
    middlefindist = math.hypot(l[4][1]-l[11][1], l[4][2]-l[11][2])

    for i in a:
        temp1.append(math.hypot(l[0][1]-l[i][1], l[0][2]-l[i][2]))
    maxval = max(temp1)

    for i in temp1:
        if i / maxval > 0.62:
            total += 1

    if indexfindist / thumbdist < 0.3 and maxval == thumbdist:
        total = 0
    if maxval == thumbdist and ringfindist / thumbdist < 0.3:
        total = 0
    if maxval == thumbdist and middlefindist / thumbdist < 0.3:
        total = 0

    return min(total, 5)


# ═══════════════════════════════════════════════════════════
# CNN: Validasi hasil MediaPipe (VALIDATOR)
# ═══════════════════════════════════════════════════════════
def binaryMask(img):
    """Konversi ROI ke binary mask — sama seperti preprocessing dataset."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7, 7), 3)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    _, img = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img

def get_hand_roi(frame, lmlist, padding=30):
    """Crop area tangan dari frame berdasarkan landmark list."""
    if len(lmlist) == 0:
        return None
    h, w, _ = frame.shape
    x_list  = [pt[1] for pt in lmlist]
    y_list  = [pt[2] for pt in lmlist]
    x1 = max(0, min(x_list) - padding)
    y1 = max(0, min(y_list) - padding)
    x2 = min(w, max(x_list) + padding)
    y2 = min(h, max(y_list) + padding)
    roi = frame[y1:y2, x1:x2]
    return roi if roi.size > 0 else None

def fingers_cnn(roi_bgr):
    """Prediksi jumlah jari pakai CNN dari ROI tangan."""
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    try:
        mask    = binaryMask(roi_bgr)
        resized = cv2.resize(mask, (64, 64))
        inp     = np.float32(resized) / 255.
        inp     = np.expand_dims(inp, axis=0)   # (1, 64, 64)
        inp     = np.expand_dims(inp, axis=-1)  # (1, 64, 64, 1)
        pred    = np.argmax(cnn_model.predict(inp, verbose=0)[0])
        return int(pred)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════
# HYBRID: Gabungkan MediaPipe + CNN
# ═══════════════════════════════════════════════════════════
def count_fingers_hybrid(frame, lmlist):
    """
    MediaPipe sebagai hasil utama, CNN sebagai validator.
    Returns: (jumlah_jari, is_confident)
      - is_confident=True  → keduanya setuju, aman eksekusi aksi
      - is_confident=False → tidak setuju, hanya tampilkan info
    """
    mp_count  = fingers_mediapipe(lmlist)
    roi       = get_hand_roi(frame, lmlist)
    cnn_count = fingers_cnn(roi)

    if cnn_count is None:
        return mp_count, False          # CNN gagal → tidak confident

    if mp_count == cnn_count:
        return mp_count, True           # Keduanya setuju → confident
    else:
        return mp_count, False          # Tidak setuju → tidak confident


# ═══════════════════════════════════════════════════════════
# AKSI berdasarkan jumlah jari
# ═══════════════════════════════════════════════════════════
def handleAction(count, is_confident, img):
    global last_action, last_action_time
    now = datetime.now().timestamp()

    if count == 1:
        # Tampilkan salam + tanggal & waktu
        st  = "Halo! Selamat Datang!"
        dt  = date.today().strftime("%d-%m-%Y")
        tym = datetime.now().strftime('%H:%M:%S')
        cv2.putText(img, st,               (10, 50),  cv2.FONT_HERSHEY_COMPLEX, 1,   (0, 200, 0), 2)
        cv2.putText(img, "Tanggal: " + dt, (10, 90),  cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 200, 0), 2)
        cv2.putText(img, "Waktu: "   + tym,(10, 125), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 200, 0), 2)

    elif count == 2:
        cv2.putText(img, "Membuka ChatGPT...", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 100, 0), 2)
        if is_confident and (last_action != "chatgpt" or (now - last_action_time) > COOLDOWN):
            webbrowser.open('https://chatgpt.com/')
            last_action      = "chatgpt"
            last_action_time = now

    elif count == 3:
        cv2.putText(img, "Membuka YouTube...", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        if is_confident and (last_action != "youtube" or (now - last_action_time) > COOLDOWN):
            webbrowser.open('https://www.youtube.com/')
            last_action      = "youtube"
            last_action_time = now

    elif count == 4:
        cv2.putText(img, "Membuka Instagram...", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (200, 0, 200), 2)
        if is_confident and (last_action != "instagram" or (now - last_action_time) > COOLDOWN):
            webbrowser.open('https://www.instagram.com/')
            last_action      = "instagram"
            last_action_time = now

    else:
        cv2.putText(img, "Jari: " + str(count), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 0), 2)

    # Status CNN di pojok bawah
    status_text  = "CNN: Setuju [OK]"   if is_confident else "CNN: Tidak Setuju [?]"
    status_color = (0, 255, 0)          if is_confident else (0, 100, 255)
    cv2.putText(img, status_text, (10, 460), cv2.FONT_HERSHEY_COMPLEX, 0.6, status_color, 1)


# ═══════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════
while True:
    _, img = cap.read()
    img    = cv2.flip(img, 1)  # mirror

    img       = obj1.findHands(img)
    leftlist  = obj1.findPositionleft(img)
    rightlist = obj1.findPositionright(img)

    if len(leftlist) != 0 or len(rightlist) != 0:
        if len(leftlist) != 0:
            left_count, left_conf = count_fingers_hybrid(img, leftlist)
        else:
            left_count, left_conf = 0, True

        if len(rightlist) != 0:
            right_count, right_conf = count_fingers_hybrid(img, rightlist)
        else:
            right_count, right_conf = 0, True

        total        = min(left_count + right_count, 10)
        is_confident = left_conf and right_conf

        # Info per tangan kalau 2 tangan terdeteksi
        if len(leftlist) != 0 and len(rightlist) != 0:
            cv2.putText(img, "Kiri: %d | Kanan: %d" % (left_count, right_count),
                        (10, 430), cv2.FONT_HERSHEY_COMPLEX, 0.7, (200, 200, 200), 1)

        handleAction(total, is_confident, img)

    cv2.imshow('Touchless App Launcher', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
