#  Face Mask Compliance Monitoring System

A fully offline, deployable computer vision module that detects faces and classifies mask compliance into **three categories**:  
**With Mask** · 
**Improper Mask** · 
**No Mask**

Designed for air‑gapped environments, this system runs **without any internet** after initial setup, making it ideal for controlled facilities (offices, labs, etc.).

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-3.x-D00000?logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Deep_Learning-CNN-blueviolet?logo=pytorch&logoColor=white" alt="Deep Learning">
  <img src="https://img.shields.io/badge/Inference-Real--time-success?logo=clockify&logoColor=white" alt="Real-time">
</p>

---

##  Table of Contents

1. [Objective](#-objective)
2. [System Architecture](#-system-architecture)
3. [Features & Bonus](#-features--bonus)
4. [System Requirements](#-system-requirements)
5. [Installation & Local Deployment](#-installation--local-deployment)
6. [Usage](#-usage)
   - [Real‑time Webcam](#real-time-webcam)
   - [Static Image Inference](#static-image-inference)
   - [Violation Dashboard](#violation-dashboard)
7. [Models & Training](#-models--training)
8. [Results & Metrics](#-results--metrics)
9. [Project Structure](#-project-structure)
10. [Acknowledgements](#-acknowledgements)

---

## 🎯 Objective

Build a deployable module (not just a model) that automatically monitors mask compliance in real time. The system must:

- Detect **multiple faces** in a video stream or static image
- Classify each face as **With Mask**, **Improper Mask** (chin only, nose exposed, etc.), or **No Mask**
- Operate **completely offline** (air‑gapped) – all dependencies and model files stored locally
- Provide a robust **alert system** with logs, screenshots, and a live violation window

---

##  System Architecture

The pipeline follows a modular design, separating concerns into distinct components:

```text
Camera / Image
    │
    ▼
┌─────────────────────────────┐
│       Face Detector         │
│  OpenCV DNN (SSD+ResNet10)  │
│      (detector.py)          │
└─────────────┬───────────────┘
              │ Face Coordinates
              ▼
┌─────────────────────────────┐
│         Face Crop           │
│      ROI Extraction         │
└─────────────┬───────────────┘
              │ Cropped Face
              ▼
┌─────────────────────────────┐
│      Mask Classifier        │
│ Custom CNN / MobileNetV2    │
│   (.keras or .tflite)       │
│     (classifier.py)         │
└─────────────┬───────────────┘
              │ Label + Confidence
              ▼
┌─────────────────────────────┐
│        Alert Logger         │
│ Logs, Screenshots, Display  │
│       (logger.py)           │
└─────────────┬───────────────┘
              │
              ▼
     Bounding Box + Label
      Displayed on Frame
```

All components are located inside the `modules/` directory and are loaded by either `main_webcam.py` or `main_image.py`.

---

# Features & Bonus

| Feature | Status |
|----------|--------|
| **Real-Time Webcam Inference** | ✅ |
| **Static Image Inference** | ✅ |
| **Three-Class Classification** | ✅ With Mask · Improper Mask · No Mask |
| **Multiple Face Detection** | ✅ |
| **Air-Gapped Deployment** | ✅ No internet required after setup |
| **Confidence Thresholding** | ✅ Filters uncertain predictions |
| **Alert System** | ✅ Log file + Screenshot capture + Live log window |
| **Model Optimization** | ✅ Quantized TFLite models for faster inference |
| **Pruning Exploration** | ✅ API demonstrated and ready for fine-tuning |
| **Violation Dashboard** | ✅ Pandas-based statistics dashboard |
| **Domain Adaptation** | ✅ Fine-tuning using self-captured images |

---

# System Requirements

- **Operating System:** Windows 10/11, macOS, or Linux (tested on Windows)
- **Python:** 3.11
- **Camera:** Any USB webcam (for real-time inference)
- **Storage:** ~1 GB free space
- **Internet:** Required only once for cloning and installing dependencies

---

# ⚙️ Installation & Local Deployment

Follow these steps on the target machine.

## 1. Install Python 3.11

Download Python from:

https://www.python.org/downloads/release/python-3119/

During installation:

✅ Check **"Add Python to PATH"**

Verify installation:

```bash
python --version
```

Expected output:

```text
Python 3.11.x
```

---

## 2. Clone the Repository

```bash
git clone https://github.com/hitakshijoshi20072911/Face_Mask_Detection_v2.git
cd Face_Mask_Detection_v2
```

---

## 3. Create and Activate a Virtual Environment

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** All trained models (`.keras`, `.tflite`) and OpenCV face detector files (`.prototxt`, `.caffemodel`) are already included in the repository. No additional downloads are required.

---

#  Usage

## Real-Time Webcam Inference

```bash
python main_webcam.py --model cnn_tflite
```

### Available Models

| Option | Description |
|----------|------------|
| `cnn` | Custom CNN (.keras) |
| `mbv2` | MobileNetV2 (.keras) |
| `cnn_tflite` | Quantized Custom CNN |
| `mbv2_tflite` | Quantized MobileNetV2 |

Enable debug mode:

```bash
python main_webcam.py --model cnn_tflite --debug
```

Press **Q** to exit.

### Output

- Bounding boxes around detected faces
- Class labels with confidence scores
- Color-coded predictions:
  - 🟢 Green → With Mask
  - 🟡 Yellow → Improper Mask
  - 🔴 Red → No Mask
- FPS counter
- Live violation log window

When a violation is detected:

- Screenshot saved in:

```text
logs/violations/
```

- Event appended to:

```text
logs/violations.log
```

---

## Static Image Inference

```bash
python main_image.py --image "path/to/image.jpg" --model cnn_tflite
```

With debug output:

```bash
python main_image.py --image "path/to/image.jpg" --model cnn_tflite --debug
```

The processed image is displayed with bounding boxes and predictions.

---

## Violation Dashboard

After violations have been logged:

```bash
python dashboard.py
```

Displays:

- Total violations
- Violations per day
- Violations per category
- Statistical summaries

---

#  Models & Training

Two classification models are provided.

## 1. Custom CNN

### Architecture

```text
Conv2D(32)
    ↓
MaxPooling
    ↓
Conv2D(64)
    ↓
MaxPooling
    ↓
Conv2D(128)
    ↓
MaxPooling
    ↓
Flatten
    ↓
Dense(256, ReLU)
    ↓
Dropout(0.5)
    ↓
Dense(3, Softmax)
```

### Training Details

- 10 epochs
- Data augmentation:
  - Rotation
  - Zoom
  - Shear
  - Horizontal Flip

---

## 2. MobileNetV2 (Transfer Learning)

### Base Model

- MobileNetV2 pretrained on ImageNet
- Frozen backbone

### Classification Head

```text
GlobalAveragePooling2D
        ↓
Dense(128)
        ↓
Dropout(0.5)
        ↓
Dense(3)
```

### Training Details

- 5 epochs
- Low learning rate fine-tuning

---

## Dataset

Models were trained using:

- FMD_DATASET (~14.5k images)

Additionally fine-tuned with self-captured improper-mask images to improve real-world performance.

---

## Retraining

```bash
python prepare_data.py
python train_custom_cnn.py
python train_mobilenetv2.py
python fine_tune_cnn.py
python quantize_and_prune.py
```

---

# 📊 Results & Metrics

Evaluation performed on a held-out test set of **1,431 images**.


## Confusion Matrix – Custom CNN

<img width="600" height="500" alt="custom_cnn after fine tuning of improper" src="https://github.com/user-attachments/assets/2873b18d-ea0d-46c6-b81e-7a5ccf3fd4fe" />


## Confusion Matrix – MobileNetV2
<img width="600" height="500" alt="Confusion Matrix_Mbv2_after fine tuning" src="https://github.com/user-attachments/assets/b8c9a550-0392-460f-b1fa-a49dee1f7466" />


## Classification Report – Custom CNN
<img width="531" height="247" alt="Classification Report _ Custom cnn (after fine tuning and self capture of improper mask)" src="https://github.com/user-attachments/assets/7705aab4-92ef-4478-ad38-4f99cdf0a114" />

## Classification Report – MobileNetV2
<img width="528" height="241" alt="Classification-report-mbv2-afterfine tuning" src="https://github.com/user-attachments/assets/14dc5649-fa13-472f-95e7-f986011083d3" />

---

# 📂 Project Structure

```text
Face_Mask_Detection_v2/
│
├── models/
│   ├── face_detector/
│   │   ├── deploy.prototxt
│   │   └── res10_300x300_ssd_iter_140000.caffemodel
│   │
│   ├── custom_cnn/
│   │   ├── mask_detector_cnn.keras
│   │   ├── mask_detector_cnn.tflite
│   │   └── class_labels.json
│   │
│   └── mobilenetv2/
│       ├── mask_detector_mbv2.keras
│       ├── mask_detector_mbv2.tflite
│       └── class_labels.json
│
├── modules/
│   ├── detector.py
│   ├── classifier.py
│   └── logger.py
│
├── main_webcam.py
├── main_image.py
├── train_custom_cnn.py
├── train_mobilenetv2.py
├── fine_tune_cnn.py
├── capture_improper.py
├── evaluate.py
├── quantize_and_prune.py
├── dashboard.py
├── prepare_data.py
├── requirements.txt
├── .gitignore
└── README.md
```

---


##  Acknowledgements

- **Dataset:** [FMD_DATASET (Kaggle)](https://www.kaggle.com/datasets/shiekhburhan/face-mask-dataset?select=FMD_DATASET)
- **Face Detector:** OpenCV DNN with SSD + ResNet-10
- **Deep Learning Framework:** TensorFlow / Keras
- **Optimization:** TensorFlow Lite Quantization
