# Cats vs Dogs Classification using Support Vector Machine (SVM)

## 1. Introduction
In this project, we aim to classify images of cats and dogs using the Support Vector Machine (SVM) algorithm. Image classification is a fundamental task in computer vision, and distinguishing between different species is a common application of this technique.

## 2. Objective
The goal of the project is to build a machine learning model that can accurately classify whether an image contains a cat or a dog.

## 3. Dataset
The dataset used for this project is the popular "Cats vs Dogs" dataset, which contains thousands of labeled images of cats and dogs. 

- **Source**: [Kaggle's Cats and Dogs dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
- **Training Set**: 25,000 images (12,500 cats and 12,500 dogs)
- **Test Set**: Separate test set for evaluation

## 4. Data Preprocessing
Before feeding the data into the SVM classifier, we performed the following preprocessing steps:

1. **Resizing**: All images were resized to a uniform size (e.g., 64x64 or 128x128 pixels).
2. **Grayscale Conversion**: Convert the colored images to grayscale for simplicity.
3. **Normalization**: Scale pixel values to the range [0, 1].
4. **Flattening**: Flatten each image into a 1D array to prepare it for SVM input.
5. **Data Splitting**: Split the dataset into training and testing sets.

```python
# Example of preprocessing code
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Flatten and normalize images
X = np.array([img.flatten() for img in images]) / 255.0

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
