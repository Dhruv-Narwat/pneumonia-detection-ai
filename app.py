import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Medical Diagnosis",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 AI Pneumonia Detection Dashboard")
st.markdown("Deep Learning based medical imaging system for detecting pneumonia from chest X-rays.")

# Sidebar
st.sidebar.title("AI Medical Assistant")
st.sidebar.info(
"""
This AI model analyzes **Chest X-ray images** and predicts whether the patient has **Pneumonia**.

Model: CNN / Transfer Learning  
Input Size: 224 × 224  
Explainability: Grad-CAM
"""
)

model = load_model("pneumonia_model.h5")

dummy = np.zeros((1,224,224,3))
model.predict(dummy)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    last_conv_layer = model.get_layer(last_conv_layer_name)

    conv_model = tf.keras.models.Model(model.inputs, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input

    for layer in model.layers[model.layers.index(last_conv_layer) + 1:]:
        x = layer(x)

    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:

        conv_output = conv_model(img_array)
        tape.watch(conv_output)

        preds = classifier_model(conv_output)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_output = conv_output[0]

    heatmap = conv_output @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap,0) / tf.reduce_max(heatmap)

    return heatmap.numpy()


uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    img_resized = cv2.resize(img,(224,224))
    img_resized = img_resized / 255.0
    img_array = np.expand_dims(img_resized, axis=0)

    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded X-ray")
        st.image(image)

    if confidence > 0.7:
        result = "Pneumonia Detected"
        st.error(result)
    else:
        result = "Normal Lung"
        st.success(result)

    st.subheader("🧠 AI Diagnosis")

    st.metric("Prediction", result)
    st.metric("Confidence Score", f"{confidence:.2f}")

    # Probability chart
    normal_prob = 1 - confidence
    pneumonia_prob = confidence

    fig, ax = plt.subplots()

    labels = ["Normal", "Pneumonia"]
    values = [normal_prob, pneumonia_prob]

    ax.bar(labels, values)
    ax.set_ylabel("Probability")
    ax.set_title("Disease Prediction Probability")

    st.pyplot(fig)

    # GradCAM
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)

    heatmap = cv2.resize(heatmap,(224,224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img_resized*255

    with col2:
        st.subheader("🔥 Grad-CAM Explanation")
        st.image(superimposed_img.astype("uint8"))

st.markdown("---")
st.caption("Medical AI System • Computer Vision Project")