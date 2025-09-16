# Task-03:
Implement a support vector machine (SVM) to classify images of cats and dogs from the Kaggle dataset.

## Overview

This project implements a **Support Vector Machine (SVM)** to classify images of **cats and dogs** using a Kaggle dataset. The goal is to build a complete machine learning pipeline that extracts image features, trains an SVM model, and predicts whether a given image is a cat or a dog.

---

## Dataset

**Dataset used:** [Dog and Cat Classification (PetImages) – Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

* Contains a single folder `PetImages` with two subfolders: `Cat` and `Dog`.
* Each class contains **12,499 images**.
* The dataset is preprocessed to remove corrupted images, ensuring clean data for training.

---

## Project Steps

### 1. Dataset Overview & EDA

* Counted the number of images in each class.
* Displayed random sample images from each category.
* Visualized key features to understand patterns in the dataset.

### 2. Feature Extraction

* Extracted **HOG (Histogram of Oriented Gradients) features** and **color histograms** for each image.
* Converted images into structured feature vectors suitable for SVM training.

### 3. Train-Test Split & Scaling

* Split the dataset into **training and testing sets**.
* Scaled features to improve model performance and accuracy.

### 4. Train SVM with Hyperparameter Tuning

* Trained an SVM classifier with **hyperparameter tuning** to find the optimal parameters.
* Selected the best model based on cross-validation performance.

### 5. Model Evaluation

* Evaluated the model on the test set.
* Generated **accuracy scores** and a **classification report** showing precision, recall, and F1-score.

### 6. Predictions on Test Images

* Predicted labels for random test images.
* Compared predicted labels with actual labels to verify model performance.

### 7. Predict External Images

* Built a pipeline to process external images.
* Successfully predicted whether new images are cats or dogs.

---

## Key Learnings

* Combining **HOG and color histogram features** improves classification accuracy.
* **SVM** is effective for image classification tasks when features are properly extracted.
* Hyperparameter tuning (C, gamma, kernel) is crucial for optimal performance.
* Handling datasets with corrupted images is essential for building robust ML pipelines.
* Developed a **full end-to-end workflow**: data preprocessing → feature extraction → model training → evaluation → external prediction.

---

## Tools & Libraries

* Python 3.x
* OpenCV
* NumPy
* Matplotlib
* scikit-image
* scikit-learn

---

If you want, I can also **draft a LinkedIn-ready version** summarizing this README in **short paragraphs with hashtags** for your post along with the video.

Do you want me to do that?
