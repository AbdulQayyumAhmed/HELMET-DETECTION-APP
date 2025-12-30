# helmet_detection_app.py

from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np

st.title("Helmet Detection App")

# Load the YOLO model (make sure best.pt is in the same folder)
model = YOLO("best.pt")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Run prediction
    results = model.predict(source=image_np)

    # Annotate image with bounding boxes
    annotated_frame = results[0].plot()  # returns OpenCV BGR image

    # Display the annotated image
    st.image(annotated_frame, caption="Prediction", use_container_width=True)

    # Display detected objects
    st.subheader("Detected objects:")
    for result in results:
        for box in result.boxes:
            cls = model.names[int(box.cls)]
            conf = float(box.conf)
            st.write(f"{cls}: {conf:.2f}")