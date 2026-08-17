# AI Brain Tumor Detection

> Medical Image Classification using PyTorch Convolutional Neural Networks (CNN) and Streamlit for automated MRI anomaly screening.

[Repository](https://github.com/SanyogSingh07/AI-Brain-Tumor-Detection)

---

## Overview

**AI Brain Tumor Detection** is a computer vision research prototype designed to classify brain MRI scans into distinct diagnostic categories (e.g., Glioma, Meningioma, Pituitary tumor, or No Tumor). The repository includes a PyTorch CNN model pipeline and an interactive Streamlit inference dashboard.

---

## Problem & Motivation

Neurological MRI scan evaluation requires specialized radiological analysis. Automated triage prototypes can support educational and research workflows by demonstrating how computer vision models extract spatial features from medical imaging data.

---

## Technical Pipeline

```
[ Input MRI Scan ] ──► [ Resize & Normalization ] ──► [ Data Augmentation ]
                                                               │
                                                               ▼
[ Class Predictions ] ◄── [ Softmax Probabilities ] ◄── [ PyTorch CNN Model ]
```

---

## Dataset & Preprocessing

- **Dataset**: Public Brain MRI Image Dataset formatted into 4 multi-class diagnostic categories.
- **Preprocessing Pipeline**:
  - Image resizing to $224 \times 224$ pixel spatial dimensions.
  - Normalization using ImageNet mean ($[0.485, 0.456, 0.406]$) and standard deviation ($[0.229, 0.224, 0.225]$).
- **Data Augmentation**: Random horizontal flipping, subtle affine rotations ($\pm 15^\circ$), and color jittering during training.

---

## Model Architecture

- **Backbone**: Custom 4-Layer Convolutional Neural Network (CNN) / ResNet-18 Transfer Learning backbone in PyTorch.
- **Activation Functions**: ReLU activation after feature extraction blocks; Softmax layer for output class probabilities.
- **Loss Function**: Cross-Entropy Loss (`nn.CrossEntropyLoss()`).
- **Optimizer**: Adam Optimizer (`lr=0.001`, `weight_decay=1e-4`).

---

## Evaluation & Results

> [!NOTE]
> Evaluation metrics are calculated on the test split.

| Category | Status / Metric |
|:---|:---|
| **Diagnostic Classes** | Glioma, Meningioma, Pituitary, No Tumor |
| **Model Evaluation** | Evaluated on validation/test split |
| **Inference Interface** | Interactive Streamlit web interface |

---

## Installation & Usage

```bash
git clone https://github.com/SanyogSingh07/AI-Brain-Tumor-Detection.git
cd AI-Brain-Tumor-Detection
pip install -r requirements.txt

# Launch Local Streamlit Web Interface
streamlit run app.py
```

---

## Limitations & Non-Clinical Disclaimer

> [!CAUTION]
> **Research & Educational Disclaimer**:  
> This software is an academic engineering project intended solely for computer vision research and educational purposes. **It is NOT a clinical diagnostic device, medical software, or healthcare tool**, and must never be used for real-world medical diagnosis, patient assessment, or treatment decisions.
