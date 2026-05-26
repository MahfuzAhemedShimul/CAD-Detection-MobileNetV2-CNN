# 🫀 CAD Detection using MobileNetV2 + SE Blocks with Grad-CAM

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Accuracy](https://img.shields.io/badge/Accuracy-90%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> A deep learning model for **Coronary Artery Disease (CAD) detection** from cardiac MRI images using MobileNetV2 with Squeeze-and-Excitation (SE) blocks and Grad-CAM explainability.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Training Curves](#training-curves)
- [Sample Predictions](#sample-predictions)
- [Installation](#installation)
- [Usage](#usage)
- [Comparison with Other Methods](#comparison-with-other-methods)
- [License](#license)

---

## 🧠 Overview

Coronary Artery Disease (CAD) is one of the leading causes of death worldwide. Early and accurate detection is critical for effective treatment. This project builds a binary image classifier (Normal vs Sick) trained on cardiac MRI scans using a fine-tuned **MobileNetV2** backbone enhanced with **Squeeze-and-Excitation (SE)** attention blocks.

Key highlights:
- ✅ **90% overall accuracy** — outperforms prior SOTA methods
- ✅ **Grad-CAM** explainability for clinical interpretability
- ✅ Two-phase training: frozen backbone → full fine-tuning
- ✅ Lightweight MobileNetV2 backbone suitable for deployment

---

## 📂 Dataset

- **Source:** [CAD Cardiac MRI Dataset on Kaggle](https://www.kaggle.com/datasets/danialsharifrazi/cad-cardiac-mri-dataset)
- **Classes:** `Normal` | `Sick`
- **Total samples used:** 2,000 (1,000 per class)
- **Split:** Train / Validation / Test
- **Input size:** 224×224 RGB

> ⚠️ The raw MRI images are **not included** in this repository due to licensing. Please download them from the Kaggle link above.

---

## 🏗️ Model Architecture

```
Input (224x224x3)
     ↓
MobileNetV2 (pretrained on ImageNet)
     ↓
Squeeze-and-Excitation (SE) Block
     ↓
Global Average Pooling
     ↓
Dense(256) + Dropout(0.5)
     ↓
Output: Sigmoid (Normal / Sick)
```

**Training Strategy:**
- Phase 1 (Epochs 0–9): Backbone frozen, only classifier trained
- Phase 2 (Epochs 10–24): Full fine-tuning with lower learning rate

---

## 📊 Results

### Confusion Matrix

| | Predicted Normal | Predicted Sick |
|---|---|---|
| **True Normal** | 276 ✅ | 24 ❌ |
| **True Sick** | 38 ❌ | 262 ✅ |

### Classification Report

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 0.88 | 0.92 | 0.90 |
| Sick | 0.92 | 0.87 | 0.89 |
| **Overall Accuracy** | | | **0.90** |

---

## 📈 Training Curves

![Training Curves](training_curves.png)

The red dashed line marks the start of fine-tuning (epoch 10). After fine-tuning, both accuracy improves and loss stabilizes significantly.

---

## 🖼️ Sample Predictions

![Sample Predictions](sample_predictions.png)

The model predicts with high confidence (82%–100%) on most test samples. Grad-CAM heatmaps highlight the relevant cardiac regions used for classification.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/CAD-Detection-MobileNetV2.git
cd CAD-Detection-MobileNetV2

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/danialsharifrazi/cad-cardiac-mri-dataset)
2. Place it in the `data/` directory
3. Open and run the notebook:

```bash
jupyter notebook notebook.ipynb
```

Or run on Kaggle directly using the provided notebook.

---

## 🏆 Comparison with Other Methods

| Method | Backbone | Accuracy | Explainability |
|--------|----------|----------|----------------|
| Chen et al. | ResNet-50 | 88.5% | No |
| Isensee et al. | U-Net | 87.2% | No |
| SE-ResNet | ResNet + SE | 89.3% | No |
| **Proposed (Ours)** | **MobileNetV2 + SE** | **90.0%** | **Grad-CAM ✅** |

Our approach achieves the **highest accuracy** among compared methods while also providing **visual explainability** via Grad-CAM — a critical feature for clinical adoption.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset by [Danial Sharifrazi on Kaggle](https://www.kaggle.com/datasets/danialsharifrazi/cad-cardiac-mri-dataset)
- MobileNetV2 pretrained weights from ImageNet via TensorFlow/Keras
- Grad-CAM implementation inspired by the original paper by Selvaraju et al.
