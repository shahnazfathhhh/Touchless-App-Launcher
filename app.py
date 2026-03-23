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

original_init = InputLayer.__init__

def custom_inputlayer_init(self, *args, **kwargs):
    if 'batch_shape' in kwargs:
        kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
    original_init(self, *args, **kwargs)

InputLayer.__init__ = custom_inputlayer_init

get_custom_objects().update({
    "DTypePolicy": mixed_precision.Policy
})

app = Flask(__name__)

model = tf.keras.models.load_model("finger_count_model.h5", compile=False)
classes = ['0','1','2','3','4','5']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    image = cv2.resize(image, (64,64))               
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)           
    image = np.expand_dims(image, axis=0)            
    return image

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["image"]

        image_data = base64.b64decode(data.split(",")[1])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        frame = np.array(image)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_mp = hands.process(rgb_frame)

        if result_mp.multi_hand_landmarks:

            for hand_landmarks in result_mp.multi_hand_landmarks:

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
                    return jsonify({"prediction": "Tidak terdeteksi"})

                img = preprocess_image(hand_img)

                pred = model.predict(img, verbose=0)
                prediction = classes[np.argmax(pred)]

                return jsonify({"prediction": prediction})

        else:
            return jsonify({"prediction": "Tidak ada tangan"})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
