import os
import numpy as np
import cv2
import time
from sklearn.decomposition import SparsePCA
import matplotlib.pyplot as plt

# Define directories
directories = [
    "C:/Users/abhis/Downloads/degraded_images/green_tint",
    "C:/Users/abhis/Downloads/degraded_images/blue_tint",
    "C:/Users/abhis/Downloads/degraded_images/red_tint",
    "C:/Users/abhis/Downloads/degraded_images/noisy",
    "C:/Users/abhis/Downloads/degraded_images/blurry",
    "C:/Users/abhis/Downloads/degraded_images/hazy",
    "C:/Users/abhis/Downloads/degraded_images/high_contrast",
    "C:/Users/abhis/Downloads/degraded_images/low_illumination"
]

# Resize dimensions
resize_dim = (128, 128)

# Function to extract Sparse PCA features
def extract_sparse_pca_color_features(image, n_components=3):
    image_resized = cv2.resize(image, resize_dim)
    image_flattened = image_resized.reshape(-1, 3)
    spca = SparsePCA(n_components=n_components, random_state=42)
    sparse_pca_features = spca.fit_transform(image_flattened)
    return sparse_pca_features.flatten()

# List to hold all extracted features and labels
all_features = []
all_labels = []

start_time = time.time()
processed_images = 0

for folder in directories:
    print(f"Processing folder: {folder}")
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            # Read the image
            image = cv2.imread(filepath)
            if image is not None:
                # Extract Sparse PCA color features
                sparse_pca_color_features = extract_sparse_pca_color_features(image)
                all_features.append(sparse_pca_color_features)
                all_labels.append(folder)
                processed_images += 1
                print(f"Processed {filename} successfully.")
            else:
                print(f"Failed to read {filename}.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

        # Print progress update
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            current_speed = processed_images / elapsed_time
            print(f"Current processing speed: {current_speed:.2f} images/sec")
            if current_speed < 10:
                print(f"Warning: Processing speed is below target. Current speed: {current_speed:.2f} images/sec")

    print(f"Finished processing folder: {folder}")

# Save the extracted features and labels
np.save("SparsePCAextractedcolor.npy", np.array(all_features))
np.save("SparsePCA_labels.npy", np.array(all_labels))

print("Feature extraction complete.")

# Print 10 random images with their Sparse PCA features
import random

sample_indices = random.sample(range(len(all_features)), min(10, len(all_features)))
for idx in sample_indices:
    print(f"Image: {all_labels[idx]}")
    print(f"Sparse PCA Features: {all_features[idx]}")

    # Visualization (optional)
    img_folder = [d for d in directories if os.path.basename(d) == os.path.basename(all_labels[idx])][0]
    img_files = os.listdir(img_folder)
    if img_files:
        img_path = os.path.join(img_folder, img_files[0])
        plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        plt.title(f"Sparse PCA Features for {os.path.basename(img_path)}")
        plt.show()
    else:
        print(f"No images found in {img_folder}")
