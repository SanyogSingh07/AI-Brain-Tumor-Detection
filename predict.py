import tensorflow as tf
import numpy as np
import cv2
import os
import random

# Load model
model = tf.keras.models.load_model("models/brain_tumor_model.h5")

yes_path = r"C:\Users\sanyo\OneDrive\Desktop\brain_tumor_ai\dataset\yes"
no_path = r"C:\Users\sanyo\OneDrive\Desktop\brain_tumor_ai\dataset\no"

# Randomly choose folder
folder_choice = random.choice(["yes", "no"])

if folder_choice == "yes":
    folder_path = yes_path
    actual_label = "Tumor"
else:
    folder_path = no_path
    actual_label = "No Tumor"

# Pick random image
images = os.listdir(folder_path)
random_image = random.choice(images)

img_path = os.path.join(folder_path, random_image)

print("\nSelected Image:", random_image)
print("Actual Label:", actual_label)

# Read image
img = cv2.imread(img_path)
img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Prediction
prediction = model.predict(img)[0][0]

# Determine label
if prediction >= 0.5:
    predicted_label = "Tumor"
else:
    predicted_label = "No Tumor"

print("Predicted Label:", predicted_label)
print("Tumor Probability:", round(prediction*100,2), "%")