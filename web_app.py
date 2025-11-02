import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os

IMG_WIDTH=30
IMG_HEIGHT=30
  

model = tf.keras.models.load_model('model.keras')


def preprocess_image(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    train = np.load("train.npz")
    x_train = train["X"]
    x_train_mean, x_train_std = np.mean(x_train, axis=0), np.std(x_train, axis=0)
    img = np.array(img)-x_train_mean
    img = (img - x_train_mean) / x_train_std
    return img

def predict(img):
    processed_image = preprocess_image(img)
    processed_image = np.expand_dims(processed_image, axis=0)
    prediction = model.predict(processed_image)
    return prediction



# Streamlit UI
st.title("Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", width=100)

    prediction = predict(image)
    confidence = np.max(prediction)
    if confidence > 0.9:
        confidence_msg = "✅ Very Confident"
    elif confidence > 0.7:
        confidence_msg = "🟡 Moderately Confident"
    else:
        confidence_msg = "🔴 Not Very Confident"
    pred_class=np.argmax(prediction)
    predicted_image_path=os.path.join("data/Meta", f"{pred_class}.png")
    st.write(f"Confidence: {confidence:.2%} ({confidence_msg})")
    st.write(f"Predicted class: {pred_class}, whose prototype image is")
    predicted_image=Image.open(predicted_image_path)
    st.image(predicted_image, width=100)