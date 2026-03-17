from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
import io

app = Flask(__name__)

model = tf.keras.models.load_model("model.h5")

classes = ['1','2','3','4','5']

def preprocess_image(image):
    image = image.resize((128,128))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["image"]

    image_data = base64.b64decode(data.split(",")[1])
    image = Image.open(io.BytesIO(image_data))

    img = preprocess_image(image)

    pred = model.predict(img)

    result = classes[np.argmax(pred)]

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)