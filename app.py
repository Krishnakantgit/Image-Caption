import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import Model
import warnings
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("imagemodel.keras")

# Load the tokenizer
@st.cache_data
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as handle:
        return pickle.load(handle)

# Generate a caption
def generate_caption(model, tokenizer, image_feature, max_length=40):
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        y_pred = model.predict([image_feature, sequence], verbose=0)
        y_pred = np.argmax(y_pred[0])
        word = tokenizer.index_word.get(y_pred, None)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace("startseq", "").replace("endseq", "").strip()

# Load everything
st.title("🖼️ Image Caption Generator using AI")
st.write("Upload an image, and the AI will generate a caption for it.")

model = load_model()
tokenizer = load_tokenizer()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess image for VGG16
        image = image.resize((224, 224))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Feature extraction using VGG16
        base_model = VGG16(weights='imagenet')  # Fully connected layers included
        feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)  # Reshape for model input
        image_feature = feature_extractor.predict(image)

        # Generate caption
        caption = generate_caption(model, tokenizer, image_feature)
        st.subheader("📝 Generated Caption:")
        st.write(caption)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
