# UIEB Dataset Analysis: Underwater Image Enhancement Classification

A comprehensive analysis of underwater image degradation classification using various feature extraction methods and machine learning classifiers on the Underwater Image Enhancement Benchmark (UIEB) dataset.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Applications](#applications)
- [Degradation Types](#degradation-types)
- [Feature Extraction Methods](#feature-extraction-methods)
- [Classification Results](#classification-results)
- [Key Findings](#key-findings)
- [Usage](#usage)
- [Requirements](#requirements)
- [Contributing](#contributing)

## 🌊 Overview

The UIEB (Underwater Image Enhancement Benchmark) dataset is a specialized collection designed to address challenges in underwater imaging, including color distortion, low contrast, and blurring caused by light scattering and absorption in water environments.

This project implements and compares multiple feature extraction techniques combined with various machine learning classifiers to automatically identify different types of underwater image degradations.

## 📊 Dataset

The UIEB dataset contains underwater images with various degradation types that commonly occur in underwater photography and imaging systems.

## 🎯 Applications

- **Marine Research**: Enhanced visual data analysis for marine biology studies
- **Underwater Exploration**: Archaeological documentation and artifact analysis
- **Photography & Videography**: Aesthetic quality improvement for underwater media
- **Autonomous Underwater Vehicles (AUVs)**: Enhanced performance in mapping, inspection, and surveillance

## 🔍 Degradation Types

The analysis covers eight main types of underwater image degradations:

- **Low Illumination**: Images with overall low brightness
- **High Contrast**: Images with extreme light-dark differences
- **Hazy**: Images with reduced clarity due to haze
- **Blurry**: Out-of-focus images
- **Noisy**: Images with significant visual noise
- **Red Tint**: Images dominated by red color cast
- **Blue Tint**: Images dominated by blue color cast
- **Green Tint**: Images dominated by green color cast

## 🛠 Feature Extraction Methods

### Structural Features
- **Sparse PCA**: Identifies important color features contributing to overall color variation
- **HOG (Histogram of Oriented Gradients)**: Captures structural and texture features
- **Sparse + HOG**: Combined sparse coding with HOG features for enhanced discriminative power
- **HOG + PCA**: Comprehensive feature set combining texture and color information

### Color-Based Features
- **Dominant Color Descriptor**: Captures prominent colors using k-means clustering
- **Color Coherence Vector**: Measures spatial coherence of colors
- **HSV Histogram**: Color distribution in HSV color space
- **Color Layout Descriptor**: Spatial color representation using grid averaging
- **Color Name Descriptor**: Maps colors to predefined color names
- **Opponency Color Features**: Captures color channel differences (red-green, yellow-blue)

## 📈 Classification Results

### Best Performing Combinations

| Method | Best Classifier | Accuracy |
|--------|----------------|----------|
| Color Coherence Vector | Gradient Boosting | 97.25% |
| HSV Histogram | Logistic Regression | 95.00% |
| SVM + Colour Histogram | SVM | 93.11% |
| Dominant Color | Gradient Boosting | 92.50% |

### Overall Method Comparison

| Classifier | Sparse PCA | HOG+PCA | HOG+Colour | Colour Histogram |
|------------|------------|---------|------------|------------------|
| Random Forest | 78.51% | 37% | 97% | 97.40% |
| Gradient Boost | 76.12% | 67% | 99% | 96.83% |
| KNN | 75.91% | 13% | 73% | 90.24% |
| Decision Tree | 74.15% | 56% | 89% | 92.69% |

### Color Method Classification Results
When using adaptive feature extraction based on degradation type:
- **Random Forest Classifier**: 100% accuracy
- **Gradient Boosting Classifier**: 100% accuracy
- **Gaussian Naive Bayes**: 97.92% accuracy
- **Logistic Regression**: 97.92% accuracy
- **MLP Classifier**: 97.92% accuracy

## 🔑 Key Findings

1. **Ensemble methods** (Random Forest, Gradient Boosting) consistently outperform individual classifiers
2. **Color-based features** generally provide better discrimination than structural features alone
3. **HOG + Colour Histogram** combination achieves exceptional results (99% with Gradient Boosting)
4. **Adaptive feature selection** based on degradation type can achieve perfect classification
5. **Color Coherence Vector** with Gradient Boosting shows the highest single-method performance (97.25%)

## 🚀 Usage

```python
# Example usage structure
# 1. Load and preprocess UIEB dataset
# 2. Extract features using chosen method
# 3. Train classifier
# 4. Evaluate performance

# Feature extraction example
features = extract_color_coherence_vector(images)
# or
features = extract_hog_color_histogram(images)

# Classification
classifier = GradientBoostingClassifier()
classifier.fit(features_train, labels_train)
accuracy = classifier.score(features_test, labels_test)
```

## 📋 Requirements

- Python 3.x
- OpenCV
- scikit-learn
- NumPy
- Matplotlib
- scikit-image

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{underwater_image_classification,
  author = {Abhishek G},
  title = {UIEB Dataset Analysis: Underwater Image Enhancement Classification},
  year = {2024},
}
}
```

## 📧 Contact

**Author**: Abhishek G  
**Degree**: BTech in Data Science

---

*This project demonstrates the effectiveness of combining multiple feature extraction techniques with ensemble learning methods for underwater image degradation classification, contributing to the advancement of underwater image enhancement technologies.*
