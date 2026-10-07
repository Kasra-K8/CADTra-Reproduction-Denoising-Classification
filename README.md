# CADTra Medical Image Denoising Reproduction

## Deep-Learning Autoencoder Denoising for Medical Image Diagnosis

This repository contains a reproduction and experimental evaluation of a deep-learning computer-aided diagnosis pipeline studied as part of a graduate **Digital Image Processing** course during my M.Sc. studies in Biomedical Engineering at K. N. Toosi University of Technology.

The project investigates whether applying a **Convolutional Denoising Autoencoder** before image classification can improve the diagnosis of pulmonary disease from chest medical images.

Two imaging modalities are considered:

- **Chest X-ray:** three-class classification of COVID-19, Pneumonia, and Normal cases
- **Chest CT:** binary classification of COVID-19 and Normal cases

The implementation reproduces the main ideas of the **CADTra** framework described in the reference study and compares image classification performance **before and after autoencoder-based denoising**.

---

## Project Overview

The complete experimental pipeline consists of three main stages:

1. **Baseline classification** on the original medical images
2. **Convolutional denoising autoencoder training and evaluation**
3. **Classification after denoising** to quantify the effect of the denoising stage

The project evaluates:

- A custom four-layer CNN (**FCNN**)
- **VGG16** with transfer learning
- **ResNet50** with transfer learning
- Convolutional denoising autoencoders
- X-ray and CT medical imaging datasets
- Gaussian, Salt-and-Pepper, and Speckle noise
- PSNR and SSIM image-quality metrics

---

## Medical Imaging Tasks

### Chest X-ray Classification

The X-ray experiment performs three-class classification:

- COVID-19
- Pneumonia
- Normal

The reproduced dataset contained **8,567 X-ray images** after dataset matching and subsampling.

### Chest CT Classification

The CT experiment performs binary classification:

- COVID-19
- Normal

The reproduced dataset contained **1,492 CT images**.

Because the exact historical dataset composition used by the original paper was not completely reproducible from the currently available public sources, the implementation uses the closest available dataset configuration while preserving the original class structure and experimental methodology.

---

## Data Preparation

The preprocessing pipeline includes:

1. Collection of chest X-ray and CT images from public datasets
2. Class-based subsampling to approximately reproduce the dataset composition of the reference study
3. Conversion to three-channel RGB images
4. Resizing all images to **224 × 224**
5. Stratified train/validation/test splitting
6. Pixel normalization
7. Data augmentation
8. Synthetic noise generation for denoising-autoencoder training

For the primary 80/20 experiment, an additional validation subset was separated from the training data to support model selection and early stopping.

### X-ray Split

| Split | Images |
|---|---:|
| Training | 6,167 |
| Validation | 686 |
| Test | 1,714 |

### CT Split

| Split | Images |
|---|---:|
| Training | 1,073 |
| Validation | 120 |
| Test | 299 |

---

## Synthetic Noise Generation

The denoising experiments evaluate three types of artificial image corruption.

### Gaussian Noise

Additive Gaussian noise is applied independently to image pixels.

### Salt-and-Pepper Noise

Random pixels are replaced with maximum or minimum intensity values.

### Speckle Noise

Multiplicative noise is applied relative to the original image intensity.

Five noise levels are evaluated:

```text
0.05
0.10
0.15
0.20
0.25
```

Noise is generated dynamically during dataset loading rather than storing separate noisy copies of every image.

This reduces storage requirements and exposes the autoencoder to different noise realizations during training.

---

## Convolutional Denoising Autoencoder

The denoising model consists of an encoder and decoder.

### Encoder

```text
Input Image
    ↓
Batch Normalization
    ↓
Conv 3×3 — 128 channels — ReLU
    ↓
Conv 3×3 — 64 channels — ReLU
    ↓
Conv 3×3 — 32 channels — ReLU
    ↓
Latent Representation
```

### Decoder

```text
Latent Representation
    ↓
Transposed Conv 3×3 — 32 channels — ReLU
    ↓
Transposed Conv 3×3 — 64 channels — ReLU
    ↓
Transposed Conv 3×3 — 128 channels — ReLU
    ↓
Output Conv
    ↓
Reconstructed Image
```

The autoencoder receives a noisy image and learns to reconstruct its corresponding clean image.

---

## Denoising Evaluation

Denoising quality is evaluated using:

- **PSNR — Peak Signal-to-Noise Ratio**
- **SSIM — Structural Similarity Index**

A total of **30 denoising configurations** were evaluated:

```text
2 imaging modalities
×
3 noise types
×
5 noise levels
=
30 configurations
```

Examples from the reproduced X-ray experiments include:

| Noise | Level | PSNR | SSIM |
|---|---:|---:|---:|
| Gaussian | 0.05 | 30.01 | 0.8093 |
| Gaussian | 0.10 | 28.69 | 0.7646 |
| Salt & Pepper | 0.05 | 35.69 | 0.9406 |
| Salt & Pepper | 0.10 | 35.19 | 0.9358 |
| Speckle | 0.05 | 32.48 | 0.8811 |
| Speckle | 0.20 | 30.68 | 0.8390 |

Several denoising configurations reproduced a substantial proportion of the image-quality performance reported in the reference study.

---

## Classification Models

Three primary classification architectures were evaluated.

### FCNN

A custom convolutional neural network trained from scratch.

The architecture includes:

- Batch Normalization
- Four convolutional layers
- ReLU activations
- Max Pooling
- Fully connected layers
- Dropout
- Softmax classification

### VGG16

A pretrained VGG16 backbone is used for transfer learning with a modified classification head.

### ResNet50

A pretrained ResNet50 backbone is used for transfer learning with a task-specific classifier.

---

# Phase 1 — Classification Before Denoising

The first experiment evaluates all classifiers directly on the original images without applying the denoising autoencoder.

## X-ray Results

| Model | Accuracy | F1 |
|---|---:|---:|
| FCNN | 87.98% | 83.56% |
| **VGG16** | **93.12%** | **89.25%** |
| ResNet50 | 89.67% | 86.37% |

The best X-ray classifier was **VGG16**, achieving **93.12% test accuracy**.

## CT Results

| Model | Accuracy | F1 |
|---|---:|---:|
| FCNN | 89.63% | 89.53% |
| **VGG16** | **91.97%** | **91.89%** |
| ResNet50 | 72.58% | 72.46% |

The best CT classifier was also **VGG16**, achieving **91.97% test accuracy**.

---

# Phase 2 — Autoencoder Denoising

Independent denoising autoencoders were trained across the two imaging modalities, three noise types, and five noise levels.

The experiment evaluates whether the convolutional autoencoder can recover clinically relevant image structure while suppressing artificial noise.

The reconstructed images are evaluated quantitatively using PSNR and SSIM before being passed to the downstream classifiers.

---

# Phase 3 — Classification After Denoising

The same FCNN, VGG16, and ResNet50 classifiers were evaluated using images processed by the denoising pipeline.

## X-ray Results After Denoising

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| FCNN | 87.28% | 88.57% | 80.66% | 83.31% |
| **VGG16** | **92.30%** | **93.80%** | **86.04%** | **88.81%** |
| ResNet50 | 89.03% | 91.15% | 82.74% | 85.57% |

## CT Results After Denoising

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| FCNN | 89.97% | 90.34% | 89.67% | 89.86% |
| **VGG16** | **91.30%** | **91.98%** | **90.93%** | **91.18%** |
| ResNet50 | 55.52% | 60.90% | 57.37% | 52.45% |

---

## Before vs After Denoising

One of the main goals of the project was to test whether autoencoder-based denoising actually improves downstream classification.

| Modality | Model | Original Accuracy | Denoised Accuracy | Change |
|---|---|---:|---:|---:|
| X-ray | FCNN | 87.98% | 87.28% | -0.70 pp |
| X-ray | VGG16 | 93.12% | 92.30% | -0.82 pp |
| X-ray | ResNet50 | 89.67% | 89.03% | -0.64 pp |
| CT | FCNN | 89.63% | **89.97%** | **+0.34 pp** |
| CT | VGG16 | 91.97% | 91.30% | -0.67 pp |
| CT | ResNet50 | 72.58% | 55.52% | -17.06 pp |

Unlike the improvement reported in the reference study, the reproduced experiments did **not show a consistent classification benefit from denoising**.

Only the CT FCNN showed a small improvement in classification accuracy.

For most configurations, classification performance remained similar or decreased after denoising.

This result highlights an important distinction between:

- improving pixel-level reconstruction quality, and
- preserving discriminative features required for disease classification.

A denoised image can achieve strong reconstruction metrics while still removing or modifying subtle features that are useful to a downstream classifier.

---

## Key Findings

### 1. VGG16 provided the strongest classification performance

The best baseline results were:

- **93.12% accuracy on chest X-ray**
- **91.97% accuracy on chest CT**

### 2. Denoising quality and classification accuracy are not equivalent

The autoencoder successfully improved noisy-image reconstruction according to PSNR and SSIM, but this did not consistently translate into improved classification.

### 3. Dataset reproduction strongly affects results

The exact dataset composition used by the original study was not fully recoverable.

The reproduced project therefore used:

- approximately **93%** of the original reported X-ray dataset size
- approximately **54%** of the original reported CT dataset size

This difference is particularly important for the smaller CT dataset.

### 4. Reproduction produced different conclusions from the reference study

The reference study reported small but consistent improvements after denoising.

In this reproduction, the effect was model-dependent and generally neutral or negative.

This demonstrates the importance of independently reproducing published machine-learning pipelines instead of assuming that reported improvements automatically generalize to different dataset versions or experimental settings.

---

## Key Concepts Explored

This project covers:

- Digital Image Processing
- Medical Image Analysis
- Computer-Aided Diagnosis
- Chest X-ray Classification
- CT Image Classification
- Convolutional Neural Networks
- Denoising Autoencoders
- Image Reconstruction
- Transfer Learning
- VGG16
- ResNet50
- Image Augmentation
- Gaussian Noise
- Salt-and-Pepper Noise
- Speckle Noise
- PSNR
- SSIM
- Stratified Data Splitting
- Early Stopping
- Deep Learning Reproducibility

---

## Repository Structure

```text
.
├── CADTra_Medical_Image_Denoising_Reproduction.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── report/
    └── CADTra_Medical_Image_Denoising_Report_FA.pdf
```

---

## Requirements

The main libraries used in the implementation include:

```text
torch
torchvision
numpy
pandas
matplotlib
seaborn
scikit-learn
scikit-image
opencv-python
Pillow
torchinfo
tqdm
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

---

## Data

The medical-image datasets are **not included in this repository**.

The project uses public chest X-ray and CT datasets corresponding to:

- COVID-19 chest X-ray images
- Pneumonia chest X-ray images
- Normal chest X-ray images
- COVID-19 CT images
- Non-COVID CT images

The implementation expects the datasets to be configured locally before preprocessing.

Large datasets, generated images, cached denoised images, and trained model checkpoints should remain excluded from Git version control.

---

## Reproducibility Notes

A fixed random seed of:

```text
42
```

was used for dataset preparation and experimental reproducibility.

Images are standardized to:

```text
224 × 224 × 3
```

The primary reported experiment uses an approximately:

```text
80% training / 20% testing
```

split, with a separate validation subset extracted from the training data.

---

## Academic Context

**Course:** Digital Image Processing  
**Project Type:** Graduate Course Project  
**Program:** M.Sc. Biomedical Engineering  
**University:** K. N. Toosi University of Technology  
**Author:** Kasra Attar Kashani

The complete Persian-language project report is available in the `report/` directory.

---

## Disclaimer

This project was completed for academic and research purposes.

It is a reproduction and experimental evaluation of a previously proposed medical-image computer-aided diagnosis framework and is **not intended for clinical use or medical decision-making**.

The results reported in this repository correspond to the reproduced experimental setup and available public datasets and should not be interpreted as clinical performance claims.
