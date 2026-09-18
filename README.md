# CIFAR-10 Image Classification — ANN vs CNN

A hands-on deep-learning study comparing a fully connected neural network (ANN) with a convolutional neural network (CNN) for CIFAR-10 image classification.

> ⚠️ **Important:** The notebook currently contains a train/test leakage issue in the CNN training step. The reported CNN accuracy should **not** be presented as a valid generalization benchmark until the notebook is corrected and rerun.

## 🎯 Objective

Demonstrate the architectural difference between an ANN, which flattens images, and a CNN, which preserves spatial structure through convolution and pooling.

## 📊 Dataset

CIFAR-10 contains 60,000 RGB 32×32 images across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## 🧠 Models

### ANN Baseline
A fully connected network using flattened image input and dense layers.

### CNN
A convolutional architecture using convolution, max-pooling, flattening, and dense classification layers.

## ⚠️ Evaluation Integrity

The current notebook trains the CNN using the test data and then evaluates on that same data. This is **data leakage** and makes the reported test score unsuitable as an unbiased estimate of generalization.

The CNN should instead be trained on the training split and evaluated only on the untouched test split:

```python
cnn.fit(X_train_scaled, y_train, epochs=10)
cnn.evaluate(X_test_scaled, y_test)
```

A validation split should also be used for model selection and overfitting monitoring.

## 🛠️ Technology Stack

Python · TensorFlow/Keras · NumPy · Pandas · Scikit-learn · Matplotlib · Seaborn · Jupyter/Colab

## 🔮 Recommended Improvements

- Fix train/test leakage
- Add a validation split
- Standardize preprocessing
- Use appropriate multi-class output activation
- Add data augmentation
- Add dropout/batch normalization
- Re-run and document clean test metrics

## 📌 Project Type

Deep Learning · Computer Vision · Image Classification · TensorFlow

**Author:** Fahad Ur Rehman
