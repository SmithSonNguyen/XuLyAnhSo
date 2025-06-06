<<<<<<< HEAD
# XuLyAnhSo

Xem ở branch master
=======
# Digital Image Processing & Recognition Project

## Introduction

This is a course project for the subject **Digital Image Processing & Computer Vision**, focusing on practical applications such as:

- Face Recognition
- Fruit Detection
- Handwritten Digit Recognition
- Human Activity Recognition
- Basic Image Processing Techniques

The project leverages modern technologies such as **YOLOv8**, **OpenCV**, **TensorFlow/Keras**, **Mediapipe**, and builds user interfaces using **Streamlit** and **Tkinter**.

---

## Demo

[▶️ Watch Demo Video](https://youtu.be/7ufiylwsbGM)

---

## 🧠 Main Features

### 👤 Face Recognition (ONNX + OpenCV)

- Uses ONNX models combined with OpenCV to detect and recognize faces in real-time via webcam.
- Supports personal face data storage and model training.

### 🍎 Fruit Detection with YOLOv8

- Applies YOLOv8 model to detect and classify different types of fruits in images.
- A user-friendly Streamlit interface allows image upload and displays detection results visually.

### ✍️ Handwritten Digit Recognition (MNIST + CNN)

- Utilizes a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits from uploaded images.

### 🕺 Human Activity Recognition

- Uses **Mediapipe** to extract pose landmarks from webcam input.
- Collects and stores movement data, then trains an **LSTM** model to recognize actions like `HANDSWING`, `BODYSWING`, and `SWING`.

### 🧪 Basic Image Processing

- Includes functionalities such as negative transformation, color space conversions, noise filtering, etc.
- Implemented in `Chapter3.py` with a Tkinter interface to open, process, and save images.

---

## 📁 Project Structure

```plaintext
cuoiki/
├── app.py                         # Main file for the Streamlit interface
├── Chapter3.py                   # Basic image processing functions
├── requirements.txt              # Required Python libraries
├── pages/                        # Functional sub-pages for Streamlit
├── Nhan_dang_chu_so/            # Handwritten digit recognition module
├── NhanDangKhuonMat_onnx/       # Face recognition module
├── Human_Activity_Recognition/  # Human activity recognition module
├── ThucHanhXuLyAnh/             # Image processing via Tkinter
├── Toi_2_4_6_TraiCay640x640_yolov8/ # Trained YOLOv8 fruit model
└── image_main/                  # Example images and illustrations

```

---

## Getting Started

1. Install Dependencies

```python
pip install -r requirements.txt
```

2. Run the Streamlit Interface

```python
streamlit run app.py
```

3. Run Scripts for Image Processing or Data Collection
   Example: Human Activity Recognition

```python
python make_data.py
```

4. Launch Tkinter GUI for Image Processing

```python
python xu_ly_anh_so.py
```
>>>>>>> master
