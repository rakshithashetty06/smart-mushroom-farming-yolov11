import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Load model
model = YOLO("best.pt")

st.title("Mushroom Detection (Edible vs Poisonous)")

# Confidence slider
conf = st.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.9,
    value=0.25,
    step=0.05
)

# Upload image
uploaded_file = st.file_uploader(
    "Upload a mushroom image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        img_path = tmp.name

    # Run prediction with confidence
    results = model.predict(
        source=img_path,
        conf=conf,
        save=False
    )

    # Show results
    for r in results:
        im_array = r.plot()  # bounding boxes drawn
        st.image(im_array, caption="Prediction Result", use_column_width=True)
