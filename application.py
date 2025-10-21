import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re

# Ensure upload directory exists
os.makedirs('uploads', exist_ok=True)

# Initialize detector and model
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Load precomputed data
filenames = pickle.load(open('filenames.pkl', 'rb'))
feature_list = pickle.load(open('embedding_vggface.pkl', 'rb'))

# Save uploaded image
def save_uploaded_file(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

# Extract facial features
def extract_feature(img_path, model, detector):
    img = np.array(Image.open(img_path))
    results = detector.detect_faces(img)
    if not results:
        return None
    x, y, width, height = results[0]['box']
    face = img[y:y+height, x:x+width]
    face_image = Image.fromarray(face).resize((224, 224))
    face_array = np.asarray(face_image, dtype='float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    test_feature = model.predict(preprocessed_img).flatten()
    return test_feature

# Find the closest celebrity
def recommend(feature_list, features):
    similarity = [cosine_similarity(features.reshape(1, -1), f.reshape(1, -1))[0][0] for f in feature_list]
    index_pos = np.argmax(similarity)
    return index_pos

# Streamlit UI
st.title("Which Bollywood Celebrity Are You?")

uploaded_image = st.file_uploader("Choose an image")

if uploaded_image is not None:
    if save_uploaded_file(uploaded_image):
        display_image = Image.open(uploaded_image)
        st.image(display_image, caption="Uploaded Image", use_column_width=True)

        features = extract_feature(os.path.join('uploads', uploaded_image.name), model, detector)
        if features is not None:
            index_pos = recommend(feature_list, features)
            filename = os.path.basename(filenames[index_pos])
            name_without_ext = os.path.splitext(filename)[0]
            cleaned_name = re.sub(r'[\W\d_]+$', '', name_without_ext)
            predicted_actor = cleaned_name.replace("_", " ").strip()
            

            col1, col2 = st.columns(2)
            with col1:
                st.header("Your Uploaded Image")
                st.image(display_image, width=300)
            with col2:
                st.header(f"Looks Like: {predicted_actor}")
                st.image(filenames[index_pos], width=300)
        else:
            st.warning("No face detected! Please upload a clearer image.")
