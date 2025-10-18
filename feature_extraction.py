# This code loads all image paths from filenames.pkl, passes each image through a pre-trained VGGFace (ResNet50) model to extract a 2048-dimensional face embedding.
# It then saves all these embeddings into embedding_vggface.pkl for later face similarity comparison.

import pickle
import numpy as np
from tqdm import tqdm

# TensorFlow/Keras image preprocessing utilities
from tensorflow.keras.preprocessing import image

# VGGFace utilities
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

# ===============================
# 1. Load list of image filenames
# ===============================
filenames = pickle.load(open('filenames.pkl', 'rb'))

# ===============================
# 2. Initialize VGGFace model (ResNet50 backbone)
# Exclude top layer to get embeddings
# Use Global Average Pooling
# ===============================
model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')

# ===============================
# 3. Define feature extraction function
# ===============================
def feature_extractor(img_path, model):
    """
    Extract 2048-dim VGGFace embedding for a given image.
    """
    # Load image and resize to 224x224
    img = image.load_img(img_path, target_size=(224,224))
    
    # Convert to array
    img_array = image.img_to_array(img)
    
    # Expand dims to add batch axis
    expanded_img = np.expand_dims(img_array, axis=0)
    
    # Preprocess for VGGFace
    preprocessed_img = preprocess_input(expanded_img)
    
    # Get embedding
    embedding = model.predict(preprocessed_img).flatten()
    
    return embedding

# ===============================
# 4. Extract features for all images
# ===============================
features = []

for file in tqdm(filenames, desc="Extracting VGGFace embeddings"):
    try:
        feat = feature_extractor(file, model)
        features.append(feat)
    except Exception as e:
        print(f"Error processing {file}: {e}")

# ===============================
# 5. Save embeddings
# ===============================
pickle.dump(features, open('embedding_vggface.pkl', 'wb'))
print("Feature extraction completed. Saved as embedding_vggface.pkl")
