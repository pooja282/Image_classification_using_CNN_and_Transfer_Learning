import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
str
# Define the rice classes
class_names = {
    0: "Jasmine",
    1: "Arborio",
    2: "Karacadag",
    3: "Ipsala",
    4: "Basmati"
}

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.VGG16.h5")
    return model

model = load_model()

# App UI
st.title("🌾 Rice Grain Classifier")
st.write("Upload an image of a rice grain to predict its type.")

# Show class labels
st.subheader("🔍 Rice Types and Their Labels:")
for key, value in class_names.items():
    st.write(f"**{key}:** {value}")

# File uploader
uploaded_file = st.file_uploader("Upload a rice grain image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # Change size if your model expects different dimensions
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_names[predicted_class]

    st.subheader("🧠 Prediction Result:")
    st.write(f"The model predicts this is **{predicted_label}** rice.")



