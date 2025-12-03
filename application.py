import streamlit as st
from PIL import Image
import os
from mtcnn import MTCNN
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import time
import re

# Ensure upload directory exists
os.makedirs('uploads', exist_ok=True)

# Initialize detector and model
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False,
                input_shape=(224, 224, 3), pooling='avg')

# Load precomputed data
filenames = pickle.load(open('filenames.pkl', 'rb'))
feature_list = pickle.load(open('embedding_vggface.pkl', 'rb'))

# Save uploaded image
def save_uploaded_file(uploaded_image):
    try:
        file_path = os.path.join('uploads', uploaded_image.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return file_path
    except:
        return None

# Extract facial features
def extract_feature(img_path):
    img = np.array(Image.open(img_path).convert('RGB'))
    results = detector.detect_faces(img)

    if not results:
        return None

    x, y, width, height = results[0]['box']
    x, y = abs(x), abs(y)

    face = img[y:y + height, x:x + width]
    face_image = Image.fromarray(face).resize((224, 224))

    face_array = np.asarray(face_image, dtype='float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    return model.predict(preprocessed_img).flatten()

# Top 5 recommendations + similarity score
def recommend_top_5(feature_list, features):
    similarity = cosine_similarity([features], feature_list)[0]
    top_5_index = np.argsort(similarity)[-5:][::-1]
    return top_5_index, similarity[top_5_index]

def clean_name(name):
    # remove trailing numbers like: "Vijay Sethupathi 13" → "Vijay Sethupathi"
    name = re.sub(r'\s*\d+$', '', name)

    # replace _ with space if any
    name = name.replace("_", " ")

    # remove extra spaces
    name = re.sub(' +', ' ', name)

    return name.strip().title()

# App Title
st.title("🎭 Celebrity Look Alike Finder")

# Upload image
uploaded_image = st.file_uploader("Upload Your Photo", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    saved_path = save_uploaded_file(uploaded_image)
    display_image = Image.open(uploaded_image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Image")
        st.image(display_image, width=300)

    if saved_path:
        with st.spinner("🔍 Finding your celebrity match..."):
            features = extract_feature(saved_path)

        if features is not None:
            top5_index, similarity_scores = recommend_top_5(feature_list, features)

            # Main match
            main_index = top5_index[0]
            confidence = round(similarity_scores[0] * 100, 2)

            main_filename = os.path.basename(filenames[main_index])
            main_name = os.path.splitext(main_filename)[0]
            main_name = clean_name(main_name)
            with col2:
                st.subheader(f"**Match :** {main_name}")
                st.image(filenames[main_index], width=300)
                st.write(f"### 🔥 Similarity: {confidence}%")
                st.progress(confidence / 100)

            # Top 5 matches
            st.subheader("Top 5 Matches")

            for index, score in zip(top5_index, similarity_scores):
                fname = os.path.basename(filenames[index])
                celebrity = os.path.splitext(fname)[0]
                celebrity = clean_name(celebrity)
                percent = round(score * 100, 2)

                c1, c2 = st.columns([1, 3])
                with c1:
                    st.image(filenames[index], width=80)
                with c2:
                    st.write(f"**{celebrity}** — {percent}%")

        else:
            st.error("❌ No face detected. Please upload a clearer image.")
