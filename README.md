# Image-Classification-using-CNN
Implemented CNN on a dataset of keras that has 50000 data records or images data.


<div align="center">

# 🖼️ CIFAR-10 Image Classification
### ANN vs. CNN — A Hands-On Comparison

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-lightgrey)
![Status](https://img.shields.io/badge/Status-Learning%20Project-yellow)
![License](https://img.shields.io/badge/License-Educational-green)

*A side-by-side study of why Convolutional Neural Networks beat plain fully-connected networks on image data.*

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model 1: Baseline ANN](#-model-1-baseline-ann)
- [Model 2: CNN](#-model-2-cnn)
- [Why CNN Wins](#-why-cnn-outperforms-ann)
- [Known Issues & Next Steps](#-known-issues--next-steps)
- [Requirements](#-requirements)
- [How to Run](#-how-to-run)

---

## 🎯 Overview

This project classifies 32×32 RGB images into **10 categories**, and demonstrates *why* CNNs outperform plain fully-connected networks on image data. Two models are trained and evaluated side by side:

| Model | Description |
|---|---|
| 🧮 **ANN (baseline)** | Fully-connected network — no spatial awareness |
| 🧠 **CNN** | Convolutional network — learns edges → shapes → objects |

<div align="center">

### 📊 Results at a Glance

| Model | Test Accuracy | Test Loss |
|:---:|:---:|:---:|
| ANN (baseline) | 🟥 `47.8%` | `1.467` |
| **CNN** | 🟩 **`75.2%`** | `0.744` |

**+27.4 percentage points** gained by adding convolution and pooling layers.

</div>

> ⚠️ **Before you cite that 75.2%** — see [Known Issues](#-known-issues--next-steps). There's a real bug worth fixing first.

---

## 📊 Dataset

**CIFAR-10** — 60,000 32×32 color images across 10 classes, loaded via `tensorflow.keras.datasets.cifar10`.

| Split | Images | Shape |
|:---:|:---:|:---:|
| 🏋️ Train | 50,000 | `(50000, 32, 32, 3)` |
| 🧪 Test | 10,000 | `(10000, 32, 32, 3)` |

**Classes:** `✈️ airplane` · `🚗 automobile` · `🐦 bird` · `🐱 cat` · `🦌 deer` · `🐶 dog` · `🐸 frog` · `🐴 horse` · `🚢 ship` · `🚚 truck`

Pixel values are normalized to `[0, 1]` by dividing by 255 before training.

---

## 🧮 Model 1: Baseline ANN

A fully-connected network with **no convolutional layers** — included specifically to expose the limits of treating image pixels as a flat feature vector.

```python
ann = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='sigmoid')
])

ann.compile(optimizer='SGD',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
```

**Training setup:** 5 epochs · SGD optimizer · full 50,000-image training set

**Result:** `47.8%` test accuracy — barely better than a coin-flip-plus-heuristic across 10 classes, with wildly inconsistent per-class performance (17% recall on class 4 vs. 84% recall on class 6). This is exactly what you'd expect from flattening spatial data: the network loses all information about *where* pixels sit relative to each other.

<details>
<summary>📋 Full classification report (ANN)</summary>

```
              precision    recall  f1-score   support

           0       0.66      0.45      0.53      1000
           1       0.62      0.61      0.61      1000
           2       0.36      0.35      0.35      1000
           3       0.37      0.28      0.32      1000
           4       0.60      0.17      0.27      1000
           5       0.47      0.30      0.37      1000
           6       0.33      0.84      0.48      1000
           7       0.64      0.48      0.55      1000
           8       0.58      0.67      0.62      1000
           9       0.51      0.63      0.57      1000

    accuracy                           0.48     10000
   macro avg       0.51      0.48      0.47     10000
weighted avg       0.51      0.48      0.47     10000
```
</details>

---

## 🧠 Model 2: CNN

A compact convolutional architecture that learns spatial hierarchies — edges and textures first, then shapes, then whole objects.

```python
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
```

**Architecture flow:**

```
Conv2D + ReLU → MaxPooling → Conv2D + ReLU → MaxPooling → Flatten → Dense → Softmax
```

**Training setup:** 10 epochs · Adam optimizer

**Result:** `75.2%` accuracy — a major jump over the ANN, with far more balanced performance across classes (recall ranges 50–89% instead of 17–84%).

<details>
<summary>📋 Full classification report (CNN)</summary>

```
              precision    recall  f1-score   support

           0       0.67      0.88      0.76      1000
           1       0.84      0.89      0.86      1000
           2       0.70      0.58      0.64      1000
           3       0.73      0.50      0.59      1000
           4       0.67      0.67      0.67      1000
           5       0.73      0.66      0.69      1000
           6       0.79      0.81      0.80      1000
           7       0.73      0.86      0.79      1000
           8       0.87      0.81      0.84      1000
           9       0.79      0.85      0.82      1000

    accuracy                           0.75     10000
   macro avg       0.75      0.75      0.75     10000
weighted avg       0.75      0.75      0.75     10000
```
</details>

---

## ⚖️ Why CNN Outperforms ANN

| | 🧮 ANN | 🧠 CNN |
|---|---|---|
| Spatial awareness | ❌ None — image flattened to 1D immediately | ✅ Preserved via 2D convolutions |
| Parameter efficiency | ❌ Very high (wide dense layers) | ✅ Lower — shared kernel weights |
| Feature learning | Raw pixel correlations | Hierarchical: edges → textures → shapes → objects |
| Translation invariance | ❌ None | ✅ Yes, via pooling |
| **Result** | 47.8% accuracy | **75.2% accuracy** |

---

## ⚠️ Known Issues & Next Steps

This is a learning-focused notebook, and a few things are worth fixing before treating the CNN's 75.2% as a trustworthy benchmark:

1. **🐛 Train/test leakage in the CNN cell** — `cnn.fit(X_test, y_test, epochs=10)` trains on the **test set** instead of `X_train, y_train`. Since `cnn.evaluate()` is then also run on `X_test, y_test`, the reported 75.2% reflects performance on data the model already trained on, not true generalization.
   **Fix:** `cnn.fit(X_train_scaled, y_train, epochs=10)`, then re-evaluate on the untouched test set.
2. **🔢 Inconsistent normalization** — `X_train_scaled`/`X_test_scaled` (pixel values ÷ 255) are computed but it's not confirmed the CNN actually trains on the scaled versions.
3. **📉 No validation split** — no held-out validation set to monitor overfitting during training.
4. **🔄 No data augmentation** — random flips/crops/rotations typically add several more points of accuracy on CIFAR-10.
5. **🏗️ Shallow architecture** — only two convolutional blocks; a third block, batch normalization, or dropout would likely push accuracy past 80%.
6. **🎯 Activation mismatch** — the ANN uses `sigmoid` on its output layer; with `sparse_categorical_crossentropy`, `softmax` (as used in the CNN) is the correct choice for multi-class, single-label output.

---

## ⚙️ Requirements

```
tensorflow>=2.x
numpy
pandas
matplotlib
scikit-learn
```

## ▶️ How to Run

1. Open the notebook in **Jupyter** or **Google Colab** (GPU runtime recommended — CIFAR-10 training is significantly faster on GPU).
2. Run all cells top to bottom. The dataset downloads automatically via `keras.datasets.cifar10`.
3. Compare the ANN and CNN sections to see the accuracy/architecture trade-off directly.

---

<div align="center">

📄 *For educational/portfolio use.*

</div>
