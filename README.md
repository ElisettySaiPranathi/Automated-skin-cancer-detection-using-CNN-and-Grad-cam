# Automated-skin-cancer-detection-using-CNN-and-Grad-cam
Below is a **complete professional README.md template** you can copy directly into your repository on **GitHub**.
You only need to **replace the username and add screenshots if you want**.

---

# Automated Skin Cancer Detection using CNN and Grad-CAM

## Overview

This project presents a deep learning–based system for automated detection and classification of skin diseases from dermoscopic images. The model uses **Convolutional Neural Networks (CNN)** to learn image features and classify skin lesions into multiple disease categories. In addition, **Grad-CAM (Gradient-weighted Class Activation Mapping)** is implemented to provide visual explanations by highlighting the important regions of the image that influence the model's prediction.

The goal of this system is to support **early detection of skin cancer** and assist healthcare professionals by providing an explainable AI-based diagnostic tool.

---

## Features

* Automated skin lesion classification using CNN
* Explainable AI using Grad-CAM visualization
* Multi-class classification of skin diseases
* Web interface for image upload and prediction
* Real-time prediction with confidence score

---

## Dataset

This project uses the **HAM10000** dataset, which contains more than **10,000 dermoscopic images** of different skin lesions.

Dataset includes the following classes:

* Melanocytic Nevi
* Melanoma
* Benign Keratosis
* Basal Cell Carcinoma
* Actinic Keratoses
* Vascular Lesions
* Dermatofibroma

Download dataset from:

[https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

After downloading, place the dataset inside:

```
dataset/
 ├── HAM10000_images_part_1
 ├── HAM10000_images_part_2
 └── HAM10000_metadata.csv
```

---

## Model Architecture

The model is built using **Convolutional Neural Networks (CNN)** for feature extraction and classification.

Key steps include:

1. Image preprocessing (resizing and normalization)
2. Feature extraction using convolutional layers
3. Classification using fully connected layers
4. Grad-CAM visualization for interpretability

---

## Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 97.6% |
| Precision | 96.6% |
| Recall    | 96.2% |
| F1 Score  | 96.4% |

---

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Flask
* Scikit-learn

---

## Project Structure

```
skin-cancer-detection/
│
├── dataset_sample/
├── models/
│   └── cnn_model.h5
│
├── notebooks/
│   └── training.ipynb
│
├── app/
│   └── app.py
│
├── gradcam/
│   └── heatmap_output.png
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```
git clone https://github.com/yourusername/skin-cancer-detection.git
```

Navigate to the project folder

```
cd skin-cancer-detection
```

Install dependencies

```
pip install -r requirements.txt
```

Run the application

```
python app.py
```

---

## Grad-CAM Visualization

Grad-CAM highlights the important regions in the image that influence the model's decision, making the system more transparent and interpretable.

Example outputs include heatmaps overlayed on the original skin lesion images.

---

## Research Publication

This project is based on the research paper:

**“Automated Skin Cancer Detection Using CNN and Grad-CAM”**
Published in the  Second International Conference on Multi- Agent Systems for Collaborative Intelligence (ICMSCI), 2026. 

---

## Future Improvements

* Deploy the model as a mobile application
* Use larger dermatology datasets for improved generalization
* Integrate advanced deep learning architectures
* Enable cloud-based diagnosis system

---



