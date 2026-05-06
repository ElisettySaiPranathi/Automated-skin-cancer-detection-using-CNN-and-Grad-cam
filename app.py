import os
import cv2
import json
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                                     BatchNormalization, GlobalAveragePooling2D,
                                     Concatenate, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
import webbrowser
import tensorflow as tf

# ================= FLASK SETUP =================
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= MODEL ARCHITECTURE (EXACT TRAIN.PY MATCH) =================
IMAGE_SIZE = 128
input_layer = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# ----- Custom CNN branch -----
x1 = Conv2D(32, (3,3), activation='relu')(input_layer)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D()(x1)

x1 = Conv2D(64, (3,3), activation='relu')(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D()(x1)

x1 = Conv2D(128, (3,3), activation='relu')(x1)  # FIXED (was wrong before)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D()(x1)
x1 = Flatten()(x1)

# ----- EfficientNet branch -----
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=input_layer)
base_model.trainable = True   # MUST match final training state

x2 = GlobalAveragePooling2D()(base_model.output)

# ----- Merge branches -----
merged = Concatenate()([x1, x2])
merged = Dense(256, activation='relu')(merged)
merged = Dropout(0.5)(merged)
output = Dense(7, activation='softmax')(merged)

model = Model(inputs=input_layer, outputs=output)

# ================= LOAD TRAINED WEIGHTS =================
model.load_weights("hybrid_skin_weights2.h5")
print("✅ Model loaded successfully!")

# ================= LOAD DISEASE LABELS =================
with open("disease_labels.json") as f:
    labels = {int(k): v for k, v in json.load(f).items()}

# ================= DISEASE INFO =================
disease_info = {
    "Melanocytic nevi (NV)": {"symptoms":"Brown/black moles.","advice":"Monitor changes."},
    "Melanoma (MEL)": {"symptoms":"Irregular borders, color variation.","advice":"URGENT: Consult dermatologist."},
    "Benign keratosis (BKL)": {"symptoms":"Scaly, rough skin patches.","advice":"Usually harmless."},
    "Basal cell carcinoma (BCC)": {"symptoms":"Pearly or waxy bump.","advice":"Seek medical treatment."},
    "Actinic keratoses (AKIEC)": {"symptoms":"Dry sun-damaged patches.","advice":"Pre-cancerous, see doctor."},
    "Vascular lesions (VASC)": {"symptoms":"Red/blue/purple marks.","advice":"Usually harmless."},
    "Dermatofibroma (DF)": {"symptoms":"Firm small bump.","advice":"Generally benign."}
}

# ================= GRAD-CAM FUNCTION =================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ================= ROUTES =================
@app.route('/')
def login():
    return render_template("login.html")

@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/upload')
def upload():
    return render_template("upload.html")

@app.route('/skin-info')
def skin_info():
    return render_template('skin_info.html')


@app.route('/predict', methods=['POST'])
def predict():
    prediction = None
    confidence = None
    img_path = None
    symptoms = ""
    advice = ""
    gradcam_path = None

    file = request.files['file']
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype("float32") / 255.0
        img = np.reshape(img, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

        pred = model.predict(img)
        class_idx = np.argmax(pred)

        prediction = labels[class_idx]
        confidence = round(float(np.max(pred)) * 100, 2)
                   
        # -------- GRAD-CAM --------
        last_conv_layer = "top_conv"
        heatmap = make_gradcam_heatmap(img, model, last_conv_layer)

        heatmap = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = heatmap * 0.4 + (img[0] * 255)
        gradcam_path = os.path.join(UPLOAD_FOLDER, "gradcam_" + file.filename)
        cv2.imwrite(gradcam_path, superimposed_img)

        img_path = filepath

        info = disease_info.get(prediction, {})
        symptoms = info.get("symptoms", "")
        advice = info.get("advice", "")

    return render_template("predict.html",
                       prediction=prediction,
                       confidence=confidence,
                       img_path=img_path,
                       gradcam_path=gradcam_path,
                       symptoms=symptoms,
                       advice=advice)


# ================= RUN =================


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)

