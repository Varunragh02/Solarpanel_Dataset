import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# Load CNN classification model
model = load_model('D:/Guvi/Solarpanel_Dataset/solar_panel_model.h5')

# Load YOLO model
yolo_model = YOLO("runs/detect/solar_panel_detector/weights/best.pt")

# Class indices (must match training)
class_indices = {
    'Bird-drop': 0,
    'Clean': 1,
    'Dusty': 2,
    'Electrical-damage': 3,
    'Physical-damage': 4,
    'Snow-Covered': 5
}
class_labels = [None] * len(class_indices)
for label, idx in class_indices.items():
    class_labels[idx] = label

# Preprocessing function
def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction function
def predict(img_path):
    processed_img = preprocess_img(img_path)
    preds = model.predict(processed_img)[0]
    top_indices = preds.argsort()[-3:][::-1]
    results = [(class_labels[i], preds[i]) for i in top_indices]
    return results

# Panel detection function
def detect_panels(img_path):
    results = yolo_model(img_path)
    results[0].save(filename="detection.jpg")  # Save detection result
    return "detection.jpg", results[0].boxes.data.tolist()  # Return path and raw predictions

# === Streamlit UI ===
st.title("☀️ Solar Panel Image Analyzer")

uploaded_file = st.file_uploader("Upload a solar panel image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save temporarily
    temp_path = "uploaded_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(temp_path, caption="Uploaded Image", use_container_width=True)


    # Classification
    if st.button("🧠 Classify Panel Condition"):
        top_predictions = predict(temp_path)
        st.subheader("🧪 Condition Classification Results:")
        for i, (label, conf) in enumerate(top_predictions, 1):
            st.write(f"**Top {i}: {label}** — Confidence: `{conf:.2f}`")

    # Detection
    if st.button("🔍 Detect Panels"):
        st.subheader("📦 Detected Panels:")
        detection_path, boxes = detect_panels(temp_path)
        st.image(detection_path, caption="Detected Panels", use_container_width=True)

        st.write("Detected Objects (Raw Bounding Box Data):")
        for i, box in enumerate(boxes, 1):
            st.write(f"Panel {i}: {box}")
