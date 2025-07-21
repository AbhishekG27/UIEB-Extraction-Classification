import os
import random
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Define HSV histogram feature extraction method
def extract_hsv_histogram(image, bins=(16, 16, 16)):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Load images from folder
def load_images(folder_path, num_images=50):
    print(f"Loading images from {folder_path}")
    image_paths = glob(os.path.join(folder_path, '*'))
    selected_images = random.sample(image_paths, min(num_images, len(image_paths)))
    images = []
    for image_path in selected_images:
        print(f"Processing image: {os.path.basename(image_path)}")
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)
        else:
            print(f"Failed to load image: {os.path.basename(image_path)}")
    return images

# Process each folder and extract features
def process_folder(folder_path, feature_extraction_function):
    images = load_images(folder_path)
    features = []
    for idx, image in enumerate(images):
        feature = feature_extraction_function(image)
        features.append(feature)
        print(f"Extracted features from image {idx + 1}/{len(images)}")
    return features

# Pad or truncate features to the same length
def pad_features(features, max_length):
    padded_features = []
    for feature in features:
        if len(feature) < max_length:
            feature = np.pad(feature, (0, max_length - len(feature)), 'constant')
        else:
            feature = feature[:max_length]
        padded_features.append(feature)
    return np.array(padded_features)

# Classifiers to evaluate
classifiers = {
    "GaussianNB": GaussianNB(),
    "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=3),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVC": SVC(kernel='linear'),
    "MLPClassifier": MLPClassifier(max_iter=1000),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100),
    "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=100),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}

# Define degradation folders
folders = [
    "green_tint",
    "blue_tint",
    "red_tint",
    "noisy",
    "blurry",
    "hazy",
    "high_contrast",
    "low_illumination"
]

# Load data, extract features, and classify
def main():
    base_path = "C:/Users/abhis/Downloads/degraded_images/"
    best_results = {"HSV Histogram": (None, 0)}
    accuracy_results = {"HSV Histogram": {}}

    print(f"\nExtracting features using HSV Histogram...")

    all_features = []
    all_labels = []

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        print(f"  Processing folder: {folder_path}")

        # Extract features
        features = process_folder(folder_path, extract_hsv_histogram)
        max_length = max(len(f) for f in features)
        features = pad_features(features, max_length)
        labels = np.full(len(features), folder)

        all_features.extend(features)
        all_labels.extend(labels)

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    # Split data
    print("  Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)

    # Evaluate each classifier
    for clf_name, classifier in classifiers.items():
        print(f"    Training and evaluating {clf_name}...")
        classifier.fit(X_train, y_train)

        # Evaluate classifier
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"    Accuracy using HSV Histogram with {clf_name}: {accuracy * 100:.2f}%")

        # Keep track of the best classifier for this feature extraction method
        if accuracy > best_results["HSV Histogram"][1]:
            best_results["HSV Histogram"] = (clf_name, accuracy)

        accuracy_results["HSV Histogram"][clf_name] = accuracy

    # Output best classifier for the method
    best_clf, accuracy = best_results["HSV Histogram"]
    print(f"\nThe best classifier for HSV Histogram is {best_clf} with accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
