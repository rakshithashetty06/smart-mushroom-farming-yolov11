import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load model
model = YOLO("best.pt")

# Title
st.title("🍄 Smart Milky Mushroom Monitoring System")
st.caption("AI + IoT Based Smart Farming System")

# Upload image
uploaded_file = st.file_uploader("Upload Mushroom Image", type=["jpg", "png"])

# Environmental inputs
temperature = st.slider("Temperature (°C)", 10, 50, 30)
humidity = st.slider("Humidity (%)", 0, 100, 80)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze"):
        results = model.predict(image)

        # ❌ No detection at all
        if len(results[0].boxes) == 0:
            st.error("❌ No mushroom detected. Please upload a milky mushroom image.")

        else:
            # ✅ Get confidence score
            conf = float(results[0].boxes.conf[0])

            # ❌ Low confidence → not mushroom
            if conf < 0.5:
                st.error("❌ This does not look like a milky mushroom. Please upload correct image.")

            else:
                # Class mapping (IMPORTANT ORDER)
                class_names = ["Contaminated", "Healthy", "PoorGrowth"]

                label = int(results[0].boxes.cls.tolist()[0])
                predicted_class = class_names[label]

                # 🌡️ Smart logic
                if predicted_class == "Contaminated":
                    suggestion = "❌ Disease detected → Remove infected bag immediately"

                elif predicted_class == "PoorGrowth":
                    if humidity < 80:
                        suggestion = "⚠️ Low humidity → Increase misting"
                    else:
                        suggestion = "⚠️ Growth issue → Check temperature & ventilation"

                elif predicted_class == "Healthy":
                    if 25 <= temperature <= 35 and 80 <= humidity <= 90:
                        suggestion = "✅ Healthy growth conditions"
                    else:
                        suggestion = "⚠️ Healthy but environment needs adjustment"

                # Display result
                st.success(f"Prediction: {predicted_class}")
                st.info(suggestion)