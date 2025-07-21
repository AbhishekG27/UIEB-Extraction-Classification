import os
import cv2
import numpy as np
from sklearn.decomposition import SparsePCA, PCA
from skimage.feature import hog
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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

# Function to extract HOG features
def extract_hog_features(image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), orientations=9):
    image_resized = cv2.resize(image, resize_dim)
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        gray_image, 
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        orientations=orientations,
        visualize=False
    )
    return hog_features

# Function to apply PCA on color features
def extract_pca_color_features(image, n_components=3):
    image_resized = cv2.resize(image, resize_dim)
    image_flattened = image_resized.reshape(-1, 3)
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(image_flattened)
    return pca_features.flatten()

# Function to apply SparsePCA on color features
def extract_sparse_pca_color_features(image, n_components=3):
    image_resized = cv2.resize(image, resize_dim)
    image_flattened = image_resized.reshape(-1, 3)
    spca = SparsePCA(n_components=n_components, random_state=42)
    spca_features = spca.fit_transform(image_flattened)
    return spca_features.flatten()

# Function to process a single image
def process_image(filepath, folder):
    try:
        image = cv2.imread(filepath)
        if image is not None:
            hog_features = extract_hog_features(image)
            pca_color_features = extract_pca_color_features(image)
            sparse_pca_color_features = extract_sparse_pca_color_features(image)
            combined_features = np.concatenate((hog_features, pca_color_features, sparse_pca_color_features))
            return combined_features, folder, filepath
        else:
            print(f"Failed to read {filepath}.")
            return None, None, filepath
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None, filepath

# List to hold all extracted features and labels
all_features = []
all_labels = []
processed_images = 0

start_time = time.time()

# Using ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = []
    for folder in directories:
        print(f"Processing folder: {folder}")
        filepaths = [os.path.join(folder, filename) for filename in os.listdir(folder)]
        for filepath in filepaths:
            futures.append(executor.submit(process_image, filepath, folder))
    
    for future in as_completed(futures):
        features, label, filepath = future.result()
        if features is not None and label is not None:
            all_features.append(features)
            all_labels.append(label)
            processed_images += 1
            print(f"Processed {filepath} successfully.")
        
        elapsed_time = time.time() - start_time
        if elapsed_time > 0 and processed_images / elapsed_time < 10:
            print(f"Warning: Processing speed is below target. Current speed: {processed_images / elapsed_time:.2f} images/sec")

print(f"Finished processing all folders. Total processed images: {processed_images}")

# Save the extracted features and labels
np.save("sparse_pca_hog_features.npy", np.array(all_features))
np.save("sparse_pca_hog_labels.npy", np.array(all_labels))

print("Feature extraction complete.")
