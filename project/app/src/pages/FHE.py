from concrete.ml.deployment import FHEModelClient
import base64
import streamlit as st
import numpy as np
import requests
from PIL import Image

# API Endpoints
URL = "http://localhost:8000"

# Initialize FHE Model Client
temp_dir = '../tmp'
client = FHEModelClient(path_dir=f'{temp_dir}/fhe_models', key_dir=f'{temp_dir}/keys_client')
eval_keys = client.get_serialized_evaluation_keys()

# Function to encrypt image
def encrypt_image(image_np):
    encrypted_img = client.quantize_encrypt_serialize(image_np)
    return encrypted_img

# Function to send image to API
def get_api_results(url, payload):
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return {"result":response.json()}
        return {"result": response.reason, "reason": "Error"}
    except:
        return {"result":"", "reason":"Error"}

# Function to encode bytes to base64
def encode(el: bytes):
    return base64.b64encode(el).decode("utf-8")

# Streamlit UI
st.title("Image Processing & API Integration")

# Create two tabs
tab1, tab2 = st.tabs(["Normal Model", "FHE Model"])

# Tab 1: Normal Model
with tab1:
    st.header("Normal Model")
    uploaded_file_normal = st.file_uploader("Upload an Image for Normal Model", type=["png", "jpg", "jpeg"], key="normal")

    if uploaded_file_normal:
        image = Image.open(uploaded_file_normal)
        image_rescaled = image.resize((30, 30))  # Resize to 30x30
        image_np = np.transpose(np.array(image_rescaled), (2, 0, 1))
        image_np = np.expand_dims(image_np, axis=0)  # Now it's (1, 3, 30, 30)
        image_np = image_np / 255

        # Send to /predict endpoint
        api_results = get_api_results(f"{URL}/predict", {"data": image_np.tolist()})

        # Display Images
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", width=300)
        with col2:
            st.image(image_rescaled, caption="Rescaled Image (30x30)", width = 300)

        # Display Results
        st.subheader("Results from Normal Model")
        st.write(api_results["result"])

# Tab 2: FHE Model
with tab2:
    st.header("FHE Model")
    uploaded_file_fhe = st.file_uploader("Upload an Image for FHE Model", type=["png", "jpg", "jpeg"], key="fhe")

    if uploaded_file_fhe:
        image = Image.open(uploaded_file_fhe)
        image_rescaled = image.resize((30, 30))  # Resize to 30x30
        image_np = np.transpose(np.array(image_rescaled), (2, 0, 1))
        image_np = np.expand_dims(image_np, axis=0)  # Now it's (1, 3, 30, 30)
        image_np = image_np / 255

        # Encrypt the image
        encrypted_image = encrypt_image(image_np)

        # Send to /predict/fhe endpoint
        api_results_fhe = get_api_results(
            f"{URL}/predict/fhe",
            {"data": {"image": encode(encrypted_image), "key": encode(eval_keys)}}
        )

        # Display Images
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", width = 300)
        with col2:
            st.image(image_rescaled, caption="Rescaled Image (30x30)", width = 300)

        str_encrypted = encode(encrypted_image)
        st.text("Encrypted Image (base64)")
        st.text(str_encrypted[:200] + " \n ... \n" + str_encrypted[200:400])

        # Display Results
        st.subheader("Results from FHE Model")
        st.write(api_results_fhe["result"])
