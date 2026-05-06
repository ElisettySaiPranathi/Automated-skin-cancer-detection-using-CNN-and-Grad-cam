import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                                     BatchNormalization, GlobalAveragePooling2D,
                                     Concatenate, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping

# ================= PARAMETERS =================
DATA_PATH = r"D:\skin_cancer\data"
IMAGE_SIZE = 128
EPOCHS = 10
BATCH_SIZE = 32

# ================= LOAD METADATA =================
df = pd.read_csv(os.path.join(DATA_PATH, "HAM10000_metadata.csv"))

label_dict = {'nv':0,'mel':1,'bkl':2,'bcc':3,'akiec':4,'vasc':5,'df':6}
df['label'] = df['dx'].map(label_dict)

disease_names = {
    0: "Melanocytic nevi (NV)",
    1: "Melanoma (MEL)",
    2: "Benign keratosis (BKL)",
    3: "Basal cell carcinoma (BCC)",
    4: "Actinic keratoses (AKIEC)",
    5: "Vascular lesions (VASC)",
    6: "Dermatofibroma (DF)"
}
class_labels = list(disease_names.values())

with open("disease_labels.json", "w") as f:
    json.dump(disease_names, f, indent=4)

df['disease_name'] = df['label'].map(disease_names)

# ================= DATASET ANALYSIS =================
plt.figure(figsize=(10,5))
sns.countplot(x='disease_name', data=df)
plt.xticks(rotation=30)
plt.title("Skin Cancer by Class")
plt.savefig("class_distribution.png")
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x='sex', hue='disease_name', data=df)
plt.title("Skin Cancer by Sex")
plt.savefig("sex_distribution.png")
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(data=df, x='age', hue='disease_name', multiple='stack', bins=20)
plt.title("Skin Cancer by Age")
plt.savefig("age_distribution.png")
plt.show()

# ================= LOAD IMAGES =================
images, labels = [], []

for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    img_id = row['image_id']
    label = row['label']

    p1 = os.path.join(DATA_PATH, "HAM10000_images_part_1", img_id + ".jpg")
    p2 = os.path.join(DATA_PATH, "HAM10000_images_part_2", img_id + ".jpg")
    path = p1 if os.path.exists(p1) else p2

    img = cv2.imread(path)
    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype("float32") / 255.0

    images.append(img)
    labels.append(label)

X = np.array(images)
y = np.array(labels)

# ================= SPLIT DATA =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

y_train_cat = to_categorical(y_train, 7)
y_test_cat = to_categorical(y_test, 7)

# ================= CLASS WEIGHTS =================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: float(w) for i, w in enumerate(class_weights)}

# ================= AUGMENTATION =================
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# ================= MODEL BUILD =================
input_layer = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

x1 = Conv2D(32, (3,3), activation='relu')(input_layer)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D()(x1)

x1 = Conv2D(64, (3,3), activation='relu')(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D()(x1)

x1 = Conv2D(128, (3,3), activation='relu')(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D()(x1)
x1 = Flatten()(x1)

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=input_layer)
base_model.trainable = False
x2 = GlobalAveragePooling2D()(base_model.output)

merged = Concatenate()([x1, x2])
merged = Dense(256, activation='relu')(merged)
merged = Dropout(0.5)(merged)
output = Dense(7, activation='softmax')(merged)

model = Model(inputs=input_layer, outputs=output)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# ================= TRAIN =================
early = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early]
)
history_data = {
    "accuracy": history.history["accuracy"],
    "val_accuracy": history.history["val_accuracy"],
    "loss": history.history["loss"],
    "val_loss": history.history["val_loss"]
}

with open("training_history.json", "w") as f:
    json.dump(history_data, f, indent=4)

print("Training history JSON saved!")


# ================= SAVE WEIGHTS =================
model.save_weights("hybrid_skin_weights2.h5")
print("Model weights saved!")

# ================= TRAINING CURVES =================
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.savefig("accuracy.png")
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")
plt.savefig("loss.png")
plt.show()

# ================= CONFUSION MATRIX =================
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# ================= ROC CURVE =================
y_test_bin = label_binarize(y_test, classes=[0,1,2,3,4,5,6])
y_score = model.predict(X_test)

for i in range(7):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, label=f"{disease_names[i]} (AUC={auc(fpr, tpr):.2f})")

plt.plot([0,1],[0,1],'k--')
plt.legend(fontsize=7)
plt.title("ROC Curve")
plt.savefig("roc_curve.png")
plt.show()

print("Training Complete")
