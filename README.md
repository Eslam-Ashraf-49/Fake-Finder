# FakeFinder: Real vs Fake Facial Image Detection

## Project Summary

FakeFinder is a deep learning-based web application that detects manipulated (fake) facial images. Built using TensorFlow and Keras, the system leverages MobileNetV2 for efficient and accurate binary classification. Users can upload facial images to verify their authenticity, helping protect individuals and institutions from misinformation, fraud, and privacy violations.

---

## Table of Contents

* Introduction
* Dataset Preparation
* Development Environment Setup
* Model Architecture
* Training Pipeline
* Evaluation and Results
* Streamlit Web App
* Usage Instructions
* Team and Contributions
* References

---

## Introduction

Face manipulation is now accessible to everyone through deepfake apps and image editing tools. This has created new challenges for social trust, law enforcement, and journalism. FakeFinder is our response to this threat—an automated detection tool that analyzes facial images and flags whether they are real or fake.

### Problem

* Fake images are easy to create
* Human eyes and traditional software can't reliably detect manipulation
* Existing forensic tools are technical and not user-friendly

### Solution

* Use MobileNetV2 as a base CNN model
* Train it on a large dataset of real and fake facial images
* Make detection accessible through a Streamlit web interface

---

## Dataset Preparation

### Structure

The dataset is divided into three folders:

```
real-vs-fake/
├── train_ds/
│   ├── real/
│   └── fake/
├── val_ds/
│   ├── real/
│   └── fake/
└── test_ds/
    ├── real/
    └── fake/
```

### Preprocessing

* Resized all images to 160x160 pixels
* Used `image_dataset_from_directory` for loading
* Normalized pixel values to \[0,1]
* Applied caching and prefetching to optimize training

```python
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'real-vs-fake/train_ds',
    image_size=(160, 160),
    batch_size=128,
    shuffle=True
)
```

---

## Development Environment Setup

### Tools Used

* Anaconda
* Python 3.x
* TensorFlow
* Keras
* OpenCV
* Flask (for backend integration)

### Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

```
tensorflow
keras
opencv-python
matplotlib
numpy
Pillow
streamlit
```

---

## Model Architecture

### Base Model

Used `MobileNetV2` pretrained on ImageNet.

```python
base_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
base_model.trainable = False
```

### Classification Head

```python
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Compile

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

---

## Training Pipeline

### Phase 1: Feature Extraction

* Freeze all layers in MobileNetV2
* Train only the top classifier

### Phase 2: Fine-Tuning

* Unfreeze top layers of MobileNetV2
* Continue training with a smaller learning rate

### Callbacks Used

* `ModelCheckpoint` to save best model
* `EarlyStopping` to prevent overfitting

### Results

* Accuracy: \~94% on validation set
* Loss: Low and stable after 10 epochs

---

## Evaluation and Testing

### Testing Interface

Used a batch of test images and manually uploaded images.
Displayed:

* Predicted label (Real/Fake)
* Confidence scores

### Metrics

* Confusion Matrix
* Classification Report

---

## Streamlit Web App

### Purpose

Let users upload a facial image and get a prediction (real/fake).

### Code Summary

```python
image = Image.open(uploaded_file).convert("RGB")
image_resized = image.resize((160, 160))
img_array = np.expand_dims(np.array(image_resized)/255.0, axis=0)
prediction = model.predict(img_array)
```

### Run

```bash
streamlit run app.py
```

### Output Example

```
This image is classified as: Real
Confidence: Real: 97.12% | Fake: 2.88%
```

---

## Usage Instructions

### Upload Image

* Go to the Streamlit interface
* Upload `.jpg`, `.jpeg`, or `.png` file
* Wait for prediction output

### Get Prediction

* Displays if image is Real or Fake
* Shows confidence levels

---

## Team and Contributions

* Eslam Ashraf Gaber Hassan
* Eslam Mohamed Mohamed Abd-Alghany
* Abdulrahman Ibrahim Mohamady Ibrahim
* Mohamed Osama Ibrahim Mohamed
* Mohamed Atef Ibrahim Ahmed

Supervised by:

* Dr. Fathy Al-Qazzaz
* Eng. Gamal Essam

Benha University, 2023

---

## References

* [TensorFlow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [Python](https://python.org/)
* [OpenCV](https://opencv.org/)
* [Streamlit](https://streamlit.io/)
