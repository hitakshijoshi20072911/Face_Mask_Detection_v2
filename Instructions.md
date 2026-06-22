

# 🛡️ Face Mask Detection v2 – Offline Deployment Guide

This guide explains how to set up and run the Face Mask Compliance Monitoring System
on a **blank Windows computer**, either with or without internet.

---

## 📋 Prerequisites

- **Operating System:** Windows 10/11
- **Python:** **3.11** (exact version 3.11.9 recommended)  
  > TensorFlow currently does **not** support Python 3.12 or 3.13.  
  > The program will **not work** with Python 3.13.
- **Camera:** Any built‑in or USB webcam (for live inference)
- **Storage:** ~2 GB free space (for the environment and models)

---

## 📁 Project Structure (after extraction)

```
Face_Mask_Detection_v2/
├── models/
│   ├── face_detector/
│   │   ├── deploy.prototxt
│   │   └── res10_300x300_ssd_iter_140000.caffemodel
│   ├── custom_cnn/
│   │   ├── mask_detector_cnn.keras
│   │   ├── mask_detector_cnn.tflite
│   │   └── class_labels.json
│   └── mobilenetv2/
│       ├── mask_detector_mbv2.keras
│       ├── mask_detector_mbv2.tflite
│       └── class_labels.json
├── modules/
│   ├── detector.py
│   ├── classifier.py
│   └── logger.py
├── main_webcam.py
├── main_image.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🐍 1. Install Python 3.11

1. Download the **Python 3.11.9** installer from the official website:  
   https://www.python.org/downloads/release/python-3119/
2. Run the installer.
3. ✅ **Check the box “Add Python to PATH”** before clicking Install.

Verify the installation by opening a new Command Prompt and typing:

```cmd
py -3.11 --version
```

You should see `Python 3.11.9`.

---

## 📥 2. Obtain the Project Files

**Option A – Download ZIP from GitHub (internet required once)**

1. On a computer with internet, go to the repository.
2. Click **Code → Download ZIP**.
3. Save the ZIP file and transfer it to the target machine (CD/USB/local network).
4. Extract the ZIP to a short path, e.g., `C:\Face_Mask_Detection_v2`.

**Option B – Clone with Git (if Git is installed)**

```cmd
git clone https://github.com/hitakshijoshi20072911/Face_Mask_Detection_v2.git
cd Face_Mask_Detection_v2
```

---

## 📦 3. Install Python Dependencies

### 3.1 With internet (easiest)

Open Command Prompt inside the project folder and run:

```cmd
cd C:\Face_Mask_Detection_v2
py -3.11 -m venv venv311
venv311\Scripts\activate
pip install -r requirements.txt
```

All packages will be downloaded and installed automatically.

### 3.2 Without internet (air‑gapped deployment)

Because the TensorFlow wheel is very large (~350 MB) and cannot be stored on GitHub,
you must **pre‑download** the wheels on a machine that has internet and carry them
on a CD or USB drive.

**Step 1 – On a connected computer (prepare the wheel bundle)**

1. Clone or extract the project.
2. Open Command Prompt in the project folder.
3. Create a temporary virtual environment and download all wheels:

```cmd
py -3.11 -m venv temp
temp\Scripts\activate
pip wheel -w wheels -r requirements.txt
deactivate
```

4. This creates a `wheels\` folder containing every package (including TensorFlow).
5. Burn the **whole project folder (with `wheels\`)** onto a CD, or copy it to a USB drive.

**Step 2 – On the offline target machine**

1. Insert the CD/USB and copy the project folder to a local drive (e.g., `C:\Face_Mask_Detection_v2`).
2. Open Command Prompt **as Administrator** (right‑click → “Run as administrator”) inside that folder.
3. Install Python 3.11 if not already present.
4. Create a fresh virtual environment and install from the local wheel folder:

```cmd
cd C:\Face_Mask_Detection_v2
py -3.11 -m venv venv311
venv311\Scripts\activate
pip install --no-index --find-links=wheels -r requirements.txt
```

> **Note:** The `--no-index --find-links=wheels` tells pip to look only inside the
> `wheels\` directory and **not** to contact the internet. This is 100% offline.

---

## 🚀 4. Run the Program

Make sure the virtual environment is still activated (`(venv311)` appears in the prompt).

### 4.1 Static Image Inference

```cmd
python main_image.py --image "Test cases/test 2.jpg" --model mbv2
```

Replace the image path with any JPEG/PNG file on your machine.

**Available model choices (`--model`):**

| Argument      | Model file used                         | Type          |
|---------------|-----------------------------------------|---------------|
| `cnn`         | custom_cnn/mask_detector_cnn.keras      | Full Keras    |
| `cnn_tflite`  | custom_cnn/mask_detector_cnn.tflite     | Quantized     |
| `mbv2`        | mobilenetv2/mask_detector_mbv2.keras    | Full Keras    |
| `mbv2_tflite` | mobilenetv2/mask_detector_mbv2.tflite   | Quantized     |

### 4.2 Real‑time Webcam Inference

```cmd
python main_webcam.py --model mbv2
```

- A window titled **“Face Mask Compliance System”** will open.
- Bounding boxes will appear:
  - 🟢 Green = `with_mask`
  - 🟡 Yellow = `incorrect_mask`
  - 🔴 Red = `without_mask`
- A dark blue **FPS counter** is shown at the top‑left.
- A separate **“Violation Log”** window displays recent alerts.
- Press **`q`** on the webcam window to quit.

---

## 📊 5. Interpreting the Output

- Each detected face gets a label and a confidence percentage.
- If a face is too close to the top edge of the frame, the text may be hidden.
  This is a known visual bug – the classification still works.
- The system logs every violation (no‑mask or improper‑mask) in `logs/violations.log`
  and saves a screenshot in `logs/violations/`.

---

## 🛠 Troubleshooting

| Symptom | Likely cause | Solution |
|---------|--------------|----------|
| `py -3.11` not found | Python 3.11 not installed | Install Python 3.11.9 from python.org |
| `pip install` fails with “No matching distribution” | Wheels folder missing or incorrect path | Ensure the command is run from the project root containing `wheels\` |
| Webcam shows black screen | Camera in use by another app or wrong camera index | Close other apps; try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `main_webcam.py` |
| `face_recognition` errors | This library is **not** required for v2 | Your current repository does not use `face_recognition`; ignore related errors |
| Text label missing on image | Face too close to top border | The classification still works; the text is drawn at a negative Y coordinate (harmless) |

---

## ⚙️ Exact Dependency Versions (as tested)

The following `requirements.txt` was used successfully with Python 3.11:

```
absl-py==1.4.0
astunparse==1.6.3
attrs==26.1.0
certifi==2026.6.17
charset-normalizer==3.4.7
contourpy==1.3.3
cycler==0.12.1
dm-tree==0.1.10
flatbuffers==25.12.19
fonttools==4.63.0
gast==0.7.0
google-pasta==0.2.0
grpcio==1.81.1
h5py==3.14.0
idna==3.18
joblib==1.5.3
keras==3.14.1
kiwisolver==1.5.0
libclang==18.1.1
markdown-it-py==4.2.0
matplotlib==3.11.0
mdurl==0.1.2
ml_dtypes==0.5.4
mock==5.2.0
namex==0.1.0
narwhals==2.22.1
numpy==2.5.0
opencv-python==4.13.0.92
opt_einsum==3.4.0
optree==0.19.1
packaging==26.2
pandas==3.0.3
pillow==12.2.0
protobuf==7.35.1
Pygments==2.20.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
requests==2.34.2
rich==15.0.0
scikit-learn==1.9.0
scipy==1.18.0
setuptools==82.0.1
six==1.17.0
tensorflow==2.21.0
tensorflow-model-optimization==0.8.1
termcolor==3.3.0
tf_keras==2.21.0
threadpoolctl==3.6.0
typing_extensions==4.15.0
tzdata==2026.2
urllib3==2.7.0
wheel==0.47.0
wrapt==2.2.2
```

---

## 📌 Final Notes

- This version (v2) is the **stable, proven baseline**. Use MobileNetV2 (`--model mbv2`) for the most reliable improper‑mask detection.
- The system runs **fully offline** once the virtual environment is prepared.
- All model files and the face detector are already included in the repository – no extra downloads needed.

For further assistance, refer to the full project README or contact the author.
```
