# 💰 CurrencyVision AI
### Indian Currency Note Recognition using Deep Learning & Explainable AI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red.svg)](https://streamlit.io/)
[![Grad-CAM](https://img.shields.io/badge/Explainable_AI-Grad--CAM-purple.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview

**CurrencyVision AI** is an enterprise-grade AI solution designed to automatically identify, classify, and verify Indian banknote denominations (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000) using a **Custom Convolutional Neural Network (CNN)** built from scratch with TensorFlow/Keras.

Unlike typical academic projects, **CurrencyVision AI** incorporates:
- **Explainable AI (Grad-CAM)**: Heatmap activation visualizer highlighting specific note features (security threads, watermarks, numerals) that drove the classification.
- **Real-Time Webcam Scanning**: Integrated camera capture for instant live banknote predictions.
- **Multi-Format Export Engine**: Automatically generates audit-ready prediction reports in **PDF**, **CSV**, and **JSON**.
- **Apple + OpenAI Inspired Glassmorphism UI**: High-end user interface featuring dynamic dark/light themes, smooth transitions, and animated statistics cards.
- **Educational Banknote Security Guide**: Interactive breakdown of official Reserve Bank of India (RBI) banknote security indicators.

---

## 🧠 Custom CNN Architecture

The model uses a custom 4-block CNN architecture initialized with **He Normal (`he_normal`)** weights and multi-stage **Dropout** to prevent overfitting:

```
INPUT TENSOR (128x128x3 RGB)
│
├── Conv2D (32 filters, 3x3) → BatchNormalization → ReLU → MaxPool (2x2)
├── Conv2D (64 filters, 3x3) → BatchNormalization → ReLU → MaxPool (2x2)
├── Conv2D (128 filters, 3x3) → BatchNormalization → ReLU → Dropout (0.3)
├── Conv2D (128 filters, 3x3) [Grad-CAM Target] → BatchNormalization → ReLU → Dropout (0.4)
│
├── Flatten
├── Dense (256 units) → ReLU → Dropout (0.5)
└── Dense (7 units) → Softmax Output (Classes: 10, 20, 50, 100, 200, 500, 2000)
```

---

## 📂 Project Directory Structure

```
CurrencyVisionAI/
├── app.py                      # Main Streamlit Multi-Page Dashboard
├── train.py                    # Training script with dynamic dataset scanner
├── predict.py                  # Predictor class & Grad-CAM integration
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── model/
│   ├── currency_model.h5       # Trained Keras CNN model weights
│   └── class_indices.json      # Class label mapping file
├── dataset/
│   ├── train/                  # Subfolders per denomination (10, 20, 50, 100, 200, 500, 2000)
│   └── validation/             # Validation split directories
├── assets/
│   ├── icons/                  # SVG Icons & graphics
│   └── images/                 # Banner & reference images
├── styles/
│   └── custom.css              # Glassmorphic Apple + OpenAI theme stylesheet
└── utils/
    ├── model_utils.py          # Custom CNN architecture & training callbacks
    ├── preprocessing.py        # Dataset dynamic scanner & image normalizer
    ├── plots.py                # Plotly confidence gauge, bar charts & confusion matrix
    ├── gradcam.py              # Keras Grad-CAM heatmap generator
    ├── report_generator.py     # PDF, CSV, and JSON report export engine
    ├── camera.py               # Streamlit live webcam processing module
    └── theme.py                # Dark / Light mode session state manager
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/CurrencyVisionAI.git
cd CurrencyVisionAI
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 📊 Dataset Structure (Kaggle Compatible)

You can populate `dataset/train` and `dataset/validation` with any Indian Currency dataset downloaded from Kaggle. The script **automatically detects folder names as class labels**:

```
dataset/
├── train/
│   ├── 10/
│   ├── 20/
│   ├── 50/
│   ├── 100/
│   ├── 200/
│   ├── 500/
│   └── 2000/
└── validation/
    ├── 10/
    ├── 20/
    ├── 50/
    ├── 100/
    ├── 200/
    ├── 500/
    └── 2000/
```

*Note: If no dataset is present, `train.py` automatically initializes starter sample note templates so the codebase remains executable out of the box.*

---

## 🚀 Model Training

To train the custom CNN model:
```bash
python train.py --epochs 25 --batch_size 32
```
During training:
- **`EarlyStopping`** halts training when validation loss stops improving.
- **`ReduceLROnPlateau`** dynamically reduces learning rate upon loss plateaus.
- Best model weights are automatically saved to `model/currency_model.h5`.

---

## 💻 Launching the Application

Run the Streamlit web dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 👁️ Explainable AI (Grad-CAM)

CurrencyVision AI generates a **Grad-CAM (Gradient-weighted Class Activation Map)** overlay for every input image.
1. Computes feature maps of the target convolutional layer (`target_conv_layer`).
2. Calculates gradients of the predicted class score with respect to feature maps.
3. Produces a JET color-mapped visual overlay highlighting key discriminative regions (e.g. RBI emblem, watermark, security thread).

---

## 🛡️ Indian Banknote Security Marks Supported
The educational module documents security features across denominations:
- **Watermark & Electrotype:** Portrait of Mahatma Gandhi & denomination numeral.
- **Security Thread:** Color-shifting thread (green to blue) with 'RBI' lettering.
- **Micro-Lettering:** Micro-printed text under magnification.
- **Latent Image:** Hidden denomination number visible at a 45° angle.
- **RBI Seal & Signature:** Official Reserve Bank seal and Governor signature.
- **Optically Variable Ink:** Color shifting numerals on high-value notes.

---

## 📜 License & Author

- **Author:** Senior AI Engineer & UI/UX Specialist
- **License:** MIT License — free for educational and commercial use.
