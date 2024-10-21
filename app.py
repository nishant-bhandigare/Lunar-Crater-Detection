import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="Lunar Crater Detection", page_icon="🌕", layout="wide")

# Title of the app
st.title("Lunar Crater Detection using YOLOv8")

# Description
st.write("""
This web app allows you to upload an image of the lunar surface, and the trained YOLOv8 model will detect craters in the image.
""")

# Load the YOLOv8 model (you can adjust the path to your trained model)
model = YOLO('best_model.pt')  # Ensure the correct model path

# Sidebar for image upload
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Upload an image of the lunar surface", type=["jpg", "jpeg", "png"])

# Function to predict craters using the YOLOv8 model
def predict_and_visualize(image):
    # Convert the uploaded PIL image to OpenCV format (numpy array)
    img = np.array(image)

    # Convert from RGB to BGR for OpenCV processing
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Run prediction on the image using YOLOv8 model
    results = model(img_bgr)

    # Get the detected bounding boxes
    for result in results[0].boxes:
        x_min, y_min, x_max, y_max = result.xyxy[0].tolist()  # Get bounding box coordinates
        conf = result.conf[0].item()  # Get confidence score

        # Draw bounding box and confidence score on the image
        cv2.rectangle(img_bgr, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        cv2.putText(img_bgr, f'{conf:.2f}', (int(x_min), int(y_min)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Convert BGR back to RGB for displaying
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return img_rgb

# If an image is uploaded
if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.subheader("Uploaded Image")
    st.image(image, caption="Uploaded Lunar Surface Image", width=400)

    # Perform crater detection and visualize the result
    st.subheader("Detected Craters")
    st.write("The image below shows the detected craters with bounding boxes:")

    # Run prediction and display the result
    prediction_image = predict_and_visualize(image)

    # Display the prediction image
    st.image(prediction_image, caption="Crater Detection Results", width=400)
else:
    st.write("Please upload an image to detect craters.")
