# 🔬 Polyp Segmentation with EfficientNet-B5 + U-Net++

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Kvasir--SEG-green)
![Dice Score](https://img.shields.io/badge/Test%20Dice-91.09%25-brightgreen)
![IoU](https://img.shields.io/badge/Test%20IoU-84.60%25-brightgreen)
![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)

A medical image segmentation pipeline for colonoscopy polyp detection using **EfficientNet-B5** as encoder with a **U-Net++ decoder** and **Deep Supervision**, trained on the Kvasir-SEG dataset.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Training Setup](#-training-setup)
- [Results](#-results)
- [XAI — Grad-CAM](#-xai--grad-cam)
- [Notebook Structure](#-notebook-structure)

---

## 🧠 Overview

This notebook implements a high-performance polyp segmentation system using a CNN-based encoder-decoder architecture. EfficientNet-B5 provides rich multi-scale feature extraction, while the U-Net++ decoder with deep supervision ensures precise segmentation masks.

| Property         | Value                                      |
|------------------|--------------------------------------------|
| **Model**        | EfficientNet-B5 + U-Net++ (Deep Supervision ✅) |
| **Task**         | Binary Semantic Segmentation               |
| **Dataset**      | Kvasir-SEG                                 |
| **Image Size**   | 384 × 384                                  |
| **Batch Size**   | 8 (Mixed Precision / AMP)                  |
| **Seed**         | 42                                         |

---

## 🏗️ Architecture

```
Input Image (384×384)
        │
   EfficientNet-B5 Encoder
   ├── Block 1 → E1 features
   ├── Block 2 → E2 features
   ├── Block 3 → E3 features
   ├── Block 4 → E4 features
   └── Block 5 → E5 features (bottleneck)
        │
   U-Net++ Decoder (Dense skip connections)
   ├── Nested dense blocks
   └── Deep Supervision (aux outputs per level)
        │
   Segmentation Head
        │
   Output Mask (384×384)
```

**Key Components:**
- **Encoder:** EfficientNet-B5 — compound-scaled CNN with excellent feature hierarchy
- **Decoder:** U-Net++ — nested dense skip connections for fine-grained boundary recovery
- **Deep Supervision:** Auxiliary BCE loss (weight=0.3) at intermediate decoder levels
- **Loss:** `DiceLoss × 0.5 + BCEWithLogitsLoss × 0.5` (main) + `0.3 × BCE` (aux)

---

## 📂 Dataset

**Kvasir-SEG** — A publicly available colonoscopy dataset for polyp segmentation.

| Split    | Images | Percentage |
|----------|--------|------------|
| Train    | 700    | 70%        |
| Val      | 150    | 15%        |
| Test     | 150    | 15%        |
| **Total**| **1000** | **100%** |

**Augmentations (Training):**
- Random horizontal & vertical flip
- Random rotation
- Color jitter (brightness, contrast)
- Normalization (ImageNet mean/std)

---

## ⚙️ Training Setup

| Parameter        | Value                                        |
|------------------|----------------------------------------------|
| Optimizer        | AdamW (lr=1e-4, weight_decay=1e-4)          |
| Scheduler        | ReduceLROnPlateau (factor=0.5, patience=5)  |
| Epochs           | 100 (max)                                    |
| Early Stopping   | 20 epochs patience                           |
| Best Epoch       | 49                                           |
| Precision        | Mixed (fp16 + fp32 AMP)                     |

---

## 📊 Results

### Validation Results (Best Epoch: 49)

| Metric        | Score   |
|---------------|---------|
| Dice Score    | **91.04%** |
| IoU Score     | **84.87%** |
| Precision     | 92.82%  |
| Sensitivity   | 91.05%  |
| Specificity   | 98.76%  |

### Test Results

| Metric        | Score   |
|---------------|---------|
| Dice Score    | **91.09%** |
| IoU Score     | **84.60%** |
| Precision     | 95.31%  |
| Sensitivity   | 88.05%  |
| Specificity   | 99.10%  |

### Multi-Threshold Evaluation (Test Set)

| Threshold | Dice   | IoU    | Precision | Sensitivity | Specificity |
|-----------|--------|--------|-----------|-------------|-------------|
| 0.3       | 0.9114 | 0.8591 | 0.9359    | 0.9161      | 0.9888      |
| 0.4       | 0.9118 | 0.8596 | 0.9416    | 0.9112      | 0.9898      |
| **0.5**   | **0.9119** | **0.8597** | **0.9467** | **0.9064** | **0.9906** |
| 0.6       | 0.9115 | 0.8588 | 0.9514    | 0.9011      | 0.9914      |
| 0.7       | 0.9104 | 0.8571 | 0.9564    | 0.8945      | 0.9922      |

> 🏆 **Best Threshold:** 0.5 → Dice = 0.9119, IoU = 0.8597

---

## 🔍 XAI — Grad-CAM

Gradient-weighted Class Activation Mapping (**Grad-CAM**) is applied to visualize the model's attention regions during polyp segmentation.

- **Library:** `grad-cam` (pytorch-grad-cam)
- Heatmaps highlight the areas the model considers most important for each prediction
- Overlaid on original colonoscopy images for clinical interpretability

---

## 📓 Notebook Structure

| Cell | Description |
|------|-------------|
| 1    | GPU Check |
| 2    | Seed Setup (seed=42) |
| 3    | Install Libraries |
| 4    | Kaggle Setup & Kvasir-SEG Download |
| 5    | Path Setup |
| 6    | 70/15/15 Train-Val-Test Split |
| 7    | Dataset Class & DataLoaders |
| 8    | Sample Visualization |
| 9    | Model Setup (EfficientNet-B5 + U-Net++ + Deep Supervision) |
| 10   | Loss, Optimizer, Scheduler Setup |
| 11   | Training Loop |
| 12   | Training Curves |
| 13   | Evaluation (Val + Test) |
| 13b  | Multi-Threshold Evaluation |
| 14   | Test Predictions Visualization |
| 15   | XAI (Grad-CAM) |

---

## 🛠️ Dependencies

```bash
pip install torch torchvision
pip install segmentation-models-pytorch albumentations
pip install grad-cam kaggle
```

---

## 📜 License

Dataset: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — Kvasir-SEG (SimulaMet)
