from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
import io
import cv2
import mediapipe as mp

from tensorflow.keras.layers import InputLayer
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import mixed_precision

from collections import deque

# ==============================
# FIX MODEL COMPATIBILITY
# ==============================
original_init = InputLayer.__init__

def custom_inputlayer_init(self, *args, **kwargs):
    if 'batch_shape' in kwargs:
        kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
    original_init(self, *args, **kwargs)

InputLayer.__init__ = custom_inputlayer_init

get_custom_objects().update({
    "DTypePolicy": mixed_precision.Policy
})

# ==============================
# INIT APP
# ==============================
app = Flask(__name__)

model = tf.keras.models.load_model("finger_count_model.h5", compile=False)
classes = ['0','1','2','3','4','5']

# ==============================
# MEDIAPIPE (REALTIME)
# ==============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==============================
# SMOOTHING (ANTI LONCAT)
# ==============================
history = deque(maxlen=5)

# ==============================
# PREPROCESS
# ==============================
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64,64))
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

# ==============================
# HITUNG JARI MEDIAPIPE
# ==============================
def count_fingers(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0]-1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # 4 jari lain
    for i in range(1, 5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i]-2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

# ==============================
# ROUTES
# ==============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("image")

        if not data:
            return jsonify({"prediction": "-"})

        # Decode image
        image_data = base64.b64decode(data.split(",")[1])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        frame = np.array(image)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_mp = hands.process(rgb_frame)

        if result_mp.multi_hand_landmarks:

            hand_landmarks = result_mp.multi_hand_landmarks[0]

            # ==========================
            # CROP HAND
            # ==========================
            h, w, _ = frame.shape
            x_list, y_list = [], []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)

            margin = 40
            xmin = max(0, xmin - margin)
            ymin = max(0, ymin - margin)
            xmax = min(w, xmax + margin)
            ymax = min(h, ymax + margin)

            hand_img = frame[ymin:ymax, xmin:xmax]

            if hand_img.size == 0:
                return jsonify({"prediction": "-"})

            # ==========================
            # MEDIAPIPE RESULT (UTAMA)
            # ==========================
            mp_result = count_fingers(hand_landmarks)

            # ==========================
            # CNN VALIDATION
            # ==========================
            img = preprocess_image(hand_img)
            cnn_pred = model.predict(img, verbose=0)
            confidence = np.max(cnn_pred)
            cnn_result = int(classes[np.argmax(cnn_pred)])

            # ==========================
            # HYBRID DECISION
            # ==========================
            if confidence < 0.7:
                final_result = mp_result
            else:
                if abs(cnn_result - mp_result) <= 1:
                    final_result = mp_result
                else:
                    final_result = cnn_result

            # ==========================
            # SMOOTHING
            # ==========================
            history.append(final_result)
            final_prediction = max(set(history), key=history.count)

            return jsonify({"prediction": str(final_prediction)})

        else:
            return jsonify({"prediction": "-"})

    except Exception as e:
        return jsonify({"error": str(e)})

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
