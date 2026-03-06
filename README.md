# AI Brain Tumor Detection

This project uses Deep Learning to detect and classify brain tumors from MRI images.

The system is built using PyTorch and deployed as a web application using Streamlit.

---

## Problem Statement

Brain tumors are abnormal growths of cells inside the brain that can disrupt normal brain functions. Early detection is critical for treatment planning.

Manual MRI analysis is time-consuming and depends heavily on radiologists. This project uses AI to assist doctors by automatically detecting tumors from MRI scans.

---

## Dataset

Brain MRI Images for Brain Tumor Detection (Kaggle)

Classes:
- Tumor
- No Tumor

---

## Model Architecture

Convolutional Neural Network (CNN)

Pipeline:

MRI Image  
↓  
Image Preprocessing  
↓  
CNN Feature Extraction  
↓  
Fully Connected Layer  
↓  
Classification

---

## Results

Training Accuracy: ~XX%

Evaluation Metrics:

- Accuracy
- Loss
- Confusion Matrix

Graphs are stored in the results folder.

---

## Deployment

The model is deployed using Streamlit.

Run locally:

pip install -r requirements.txt

streamlit run app.py

---

## Project Structure

AI-Brain-Tumor-Detection
│
├── train.py
├── app.py
├── model/
├── results/
├── dataset/
├── requirements.txt
└── README.md

---

## Future Improvements

- Multi-class tumor classification
- Tumor localization
- Cloud deployment
- Explainable AI (Grad-CAM)
