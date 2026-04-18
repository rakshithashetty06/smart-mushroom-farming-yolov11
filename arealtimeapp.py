import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

st.set_page_config(page_title="Mushroom Detection", layout="centered")

st.title("🍄 Mushroom Detection (Live Webcam)")
st.write("Real-time edible vs poisonous mushroom detection")

# Load YOLO model
model = YOLO("best.pt")

# Confidence slider
conf = st.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.9,
    value=0.25,
    step=0.05
)

# Webcam toggle
run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

cap = None

if run:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Webcam not accessible")
    else:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame")
                break

            # Run YOLO prediction
            results = model.predict(
                source=frame,
                conf=conf,
                stream=True
            )

            for r in results:
                annotated_frame = r.plot()

            # Convert BGR to RGB
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            FRAME_WINDOW.image(annotated_frame)

        cap.release()
