# This code scans all subfolders inside the dataset directory and collects the full paths of every image file.
# It then saves those image paths into a filenames.pkl file so they can be quickly loaded later without rescanning the folders.

import os
import pickle

# Directory containing actor subfolders
dataset_dir = 'dataset'

# List of all actor folders
actors = os.listdir(dataset_dir)

filenames = []

# Loop through each actor's folder inside the 'dataset' directory
for actor in actors:
    actor_path = os.path.join(dataset_dir, actor)
    if os.path.isdir(actor_path):  # Make sure it's a folder
        # Loop through each file (image) in the current actor's folder
        for file in os.listdir(actor_path):
            file_path = os.path.join(actor_path, file)
            filenames.append(file_path)

# Save (serialize) the 'filenames' list into 'filenames.pkl'
with open('filenames.pkl', 'wb') as f:
    pickle.dump(filenames, f)

print(f"Total images saved: {len(filenames)}")
