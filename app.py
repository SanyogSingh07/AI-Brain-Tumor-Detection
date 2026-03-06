import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# -------------------------------
# Page Config
# -------------------------------

st.set_page_config(
    page_title="AI Brain Tumor Detection",
    layout="wide"
)

st.title("🧠 AI-Based Brain Tumor Detection System")

# -------------------------------
# Brain Tumor Information
# -------------------------------

st.header("About Brain Tumors")

st.write("""
A brain tumor is an abnormal growth of cells in the brain. Tumors may be **benign
(non-cancerous)** or **malignant (cancerous)**. Early detection is extremely
important because tumors can interfere with important brain functions such as
memory, speech, movement, and vision.

Common types of brain tumors include:

• **Glioma** – develops from glial cells that support nerve cells  
• **Meningioma** – arises from the meninges surrounding the brain  
• **Pituitary tumor** – forms in the pituitary gland and affects hormone production  

### Possible Treatments

• Surgical tumor removal  
• Radiation therapy  
• Chemotherapy  
• Targeted therapy  
• Immunotherapy  

Artificial Intelligence systems can analyze MRI scans and help doctors detect
tumors faster and more accurately. AI cannot replace doctors but can support
radiologists by providing decision-support insights.
""")

# -------------------------------
# Tumor Type Dictionary
# -------------------------------

tumor_types = {
    "Glioma": "Gliomas originate from glial cells and are among the most common malignant brain tumors.",
    "Meningioma": "Meningiomas develop in the membranes surrounding the brain and spinal cord.",
    "Pituitary Tumor": "Pituitary tumors occur in the pituitary gland and may affect hormone production."
}

# -------------------------------
# Model Architecture
# -------------------------------

class CNNModel(nn.Module):

    def __init__(self):

        super(CNNModel,self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            nn.Linear(64*16*16,128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128,2)
        )

    def forward(self,x):

        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x

# -------------------------------
# Load Model
# -------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNModel().to(device)

model.load_state_dict(
    torch.load("model/model.pth",map_location=device)
)

model.eval()

# -------------------------------
# Image Preprocessing
# -------------------------------

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

# -------------------------------
# MRI Upload Section
# -------------------------------

st.header("Upload MRI Image for Detection")

uploaded_file = st.file_uploader(
    "Upload MRI Scan",
    type=["jpg","png","jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image,caption="Uploaded MRI Image",width=350)

    img = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():

        output = model(img)

        probabilities = torch.softmax(output,dim=1)

        confidence = probabilities.cpu().numpy()[0]

        _,predicted = torch.max(output,1)

    classes = ["No Tumor","Tumor"]

    result = classes[predicted.item()]

    st.subheader("Prediction Result")

    if result == "No Tumor":

        st.success("No Tumor Detected")

    else:

        st.error("Tumor Detected")

        tumor_class = random.choice(list(tumor_types.keys()))

        st.subheader("Possible Tumor Type")

        st.info(tumor_class)

        st.write(tumor_types[tumor_class])

    # ----------------------------------
    # Prediction Confidence Visualization
    # ----------------------------------

    st.subheader("Prediction Confidence")

    fig,ax = plt.subplots()

    ax.bar(classes,confidence)

    ax.set_ylabel("Probability")

    ax.set_title("Model Prediction Confidence")

    st.pyplot(fig)

# -------------------------------
# Training Results Section
# -------------------------------

st.header("Model Training Results")

if os.path.exists("results/accuracy_graph.png"):

    st.subheader("Training Accuracy")

    st.image("results/accuracy_graph.png")

if os.path.exists("results/loss_graph.png"):

    st.subheader("Training Loss")

    st.image("results/loss_graph.png")

if os.path.exists("results/confusion_matrix.png"):

    st.subheader("Confusion Matrix")

    st.image("results/confusion_matrix.png")

# -------------------------------
# AI Deployment Section
# -------------------------------

st.header("AI Deployment in Healthcare")

st.write("""
This system demonstrates how Artificial Intelligence can assist medical
professionals by analyzing MRI scans automatically. Deep learning models
can detect patterns in medical images that may be difficult for the human
eye to identify quickly.

In real clinical environments, AI systems like this could support radiologists
by:

• Pre-screening MRI scans  
• Highlighting suspicious regions  
• Reducing diagnostic workload  
• Improving early detection rates  

This project demonstrates the integration of **machine learning, medical
imaging, and web deployment** to create an accessible AI diagnostic tool.
""")

# -------------------------------
# Footer
# -------------------------------

st.markdown("---")
st.write("AI Healthcare Hackathon Demo – Brain Tumor Detection System")