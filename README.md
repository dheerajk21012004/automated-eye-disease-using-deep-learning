# automated-eye-disease-using-deep-learning

# Retinal Fundus Image Analysis for Multi-Label Ocular and Systemic Disease Classification

**Author:** Dheeraj Kodwani  
**Institution:** Anglia Ruskin University  
**Programme:** BSc (Hons) Artificial Intelligence  
**Supervisor:** Dr. Silvia Cirstea  
**Year:** 2025–2026  

---

## Overview

This project develops and evaluates a deep learning framework for the automated detection and classification of multiple ocular and systemic diseases from retinal fundus images. Rather than focusing on a single disease, this study frames the problem as a **multi-label classification task** — where a single retinal image can be associated with more than one diagnostic condition simultaneously. This reflects real-world clinical scenarios where patients frequently present with co-existing conditions such as diabetic retinopathy alongside hypertension or age-related macular degeneration.

The project sits within the emerging field of **oculomics** — the study of associations between retinal biomarkers and systemic health conditions — and adopts a **weakly supervised learning** approach, training models using only image-level diagnostic labels rather than costly and resource-intensive lesion-level annotations.

---

## Research Context

### Problem Statement

Traditional automated retinal disease detection systems predominantly focus on single-disease classification and rely on strongly annotated datasets requiring pixel-level or region-level labels. These approaches are difficult to scale in real-world clinical environments where:

- Detailed lesion annotations are rarely available
- Multiple diseases frequently co-exist within a single retinal image
- Dataset class imbalance is significant (e.g. hypertension appears in fewer than 10% of cases compared to diabetes)
- Model interpretability is critical for clinical trust

### Research Gaps Addressed

This project addresses two linked gaps identified in the literature:

1. **Limited weakly supervised pipelines** that avoid lesion-level annotation while maintaining clinical interpretability
2. **Absence of a unified multi-label framework** capable of detecting systemic diseases (diabetes, hypertension) as primary objectives while simultaneously reporting co-existing ocular conditions

---

## Datasets

### Primary Dataset — ODIR-5K

- **Source:** Ocular Disease Intelligent Recognition dataset (available on [Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k))
- **Size:** 6,392 retinal fundus images from 3,358 unique patients
- **Structure:** Multi-label, image-level annotations only (no pixel-level labels)
- **Disease Classes:** 8 categories — Normal (N), Diabetes (D), Glaucoma (G), Cataract (C), Age-Related Macular Degeneration (A), Hypertension (H), Myopia (M), Other Abnormalities (O)
- **Class Imbalance:** Significant — diabetes (1,105 patients) vs hypertension (103 patients), approximately a 10:1 ratio
- **Patient Distribution:** 3,034 patients contributed bilateral images; 324 contributed single eye images

### External Validation Dataset — FIVES

- **Source:** Fundus Image Dataset for Vessel Segmentation
- **Purpose:** Used exclusively for external testing to assess model generalisation under domain shift
- **Classes Used:** AMD, Diabetes, Glaucoma, Normal (4 overlapping categories with ODIR-5K)
- **Note:** Not integrated into training due to differences in label definitions and annotation structure

---

## Methodology

### Data Preprocessing

All images underwent the following preprocessing pipeline, applied consistently across all model architectures:

1. **Resizing** — All images resized to 256 × 256 pixels for consistent input dimensions and efficient batch processing
2. **Normalisation** — Pixel intensities normalised using ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]` to align with pretrained backbone expectations
3. **Data Augmentation** (training only) — Random horizontal flipping, minor rotations (±15°), random resized cropping (scale 0.85–1.0), and mild colour jitter to improve generalisation

### Data Splitting Strategy

Dataset was split at **patient level** (not image level) to prevent data leakage between subsets. This is critical in medical AI — splitting by image when bilateral images exist from the same patient leads to overly optimistic performance metrics.

| Split | Percentage | Patients |
|-------|-----------|---------|
| Training | 70% | ~2,350 |
| Validation | 15% | ~504 |
| Testing | 15% | ~504 |

All images from a single patient were assigned exclusively to one subset.

### Weakly Supervised Learning Strategy

Models were trained using **image-level diagnostic labels only** — no lesion-level or pixel-level annotations were used. Each retinal image was treated as an independent sample that may carry one or more disease labels, enabling multi-label classification under weak supervision. This approach improves scalability and reduces dependence on expert annotation.

---

## Model Architectures

Five deep learning architectures were implemented and evaluated:

### 1. EfficientNet-B2 Baseline (CNN)

- **Backbone:** EfficientNet-B2 pretrained on ImageNet
- **Head:** Custom multi-label classification head replacing the original softmax layer
- **Activation:** Sigmoid (independent per-class probability)
- **Training:** Two-stage — head-only training followed by full fine-tuning
- **Macro AUC:** 0.7986

### 2. EfficientNet-B2 Refined (CNN + Attention)

- **Architecture:** EfficientNet-B2 backbone with additional convolutional refinement layers and CBAM (Convolutional Block Attention Module) incorporating both channel and spatial attention
- **Purpose:** Improve localisation of disease-relevant retinal features
- **Macro AUC:** 0.8101 *(highest among all models on ODIR-5K)*

### 3. EfficientNet-B2 Gated (CNN + Attention + Gating)

- **Architecture:** Extended refined model with additional gating mechanisms to control information flow
- **Observation:** Performance declined despite increased complexity — macro AUC 0.7811 — demonstrating that architectural complexity does not guarantee improved performance on imbalanced datasets
- **Macro AUC:** 0.7811

### 4. ResNet50-Transformer Hybrid (CNN + Transformer)

- **Architecture:** ResNet50 backbone for deep spatial feature extraction, with features tokenised and passed through a Transformer encoder for global contextual modelling
- **Strength:** Best generalisation on external FIVES dataset (accuracy 0.5925)
- **Macro AUC:** 0.8013
- **Macro F1:** 0.5085

### 5. Vision Transformer — ViT (Pure Transformer)

- **Architecture:** Pure transformer — images processed as sequences of patches with self-attention mechanisms modelling global dependencies
- **Strength:** Highest F1-score overall (0.5362), strongest recall
- **Observation:** Tendency to over-predict disease presence (lower precision), fragmented Grad-CAM attention patterns
- **Macro AUC:** 0.7839

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Optimiser | AdamW (lr=6e-5, weight_decay=7e-4) |
| Loss Function | Weighted Binary Cross-Entropy (BCEWithLogitsLoss) |
| Class Weights | Computed per-class, capped at 10.0 |
| Mixed Precision | Automatic Mixed Precision (AMP) |
| Early Stopping | Patience = 3–6 epochs |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Batch Size | 8 |
| Input Resolution | 256 × 256 |
| GPU | CUDA-enabled GPU |

---

## Results

### ODIR-5K Performance (Test Set)

| Model | Macro AUC | Label Accuracy | Sensitivity | Specificity | F1-Score |
|-------|-----------|---------------|-------------|-------------|----------|
| EfficientNet-B2 (Baseline) | 0.7986 | 0.7525 | 0.7315 | 0.7331 | 0.4617 |
| EfficientNet-B2 (Refined) | **0.8101** | 0.7783 | 0.7125 | 0.7737 | 0.4224 |
| EfficientNet-B2 (Gated) | 0.7811 | 0.7405 | 0.7094 | 0.7318 | 0.3796 |
| ResNet50-Transformer (Hybrid) | 0.8013 | **0.8430** | 0.5869 | **0.8582** | 0.5085 |
| Vision Transformer (ViT) | 0.7839 | 0.8009 | **0.6373** | 0.8125 | **0.5362** |

### FIVES External Validation

| Model | Accuracy |
|-------|----------|
| EfficientNet-B2 (Refined) | 0.5725 |
| ResNet50-Transformer (Hybrid) | **0.5925** |
| Vision Transformer (ViT) | 0.5413 |

> All models substantially exceed the random chance baseline (25%) and majority-class baseline (~40%), confirming genuine generalisation capability.

### Key Findings

- **No single architecture dominated across all metrics** — CNN models offer stable discriminative performance, hybrid models offer best accuracy and generalisation, transformers achieve strongest F1
- **Hypertension was the most challenging class** across all models — F1-scores as low as 0.065 — reflecting severe class imbalance (103 patients vs 1,105 for diabetes)
- **Diabetic retinopathy showed consistent underperformance on FIVES** — likely reflecting domain shift in image acquisition protocols rather than a model-specific limitation
- **Normal class bias** — all models showed high recall for normal cases (up to 0.91), indicating a tendency to favour normal prediction that would require threshold calibration before clinical deployment

---

## Explainability Analysis

**Grad-CAM** (Gradient-weighted Class Activation Mapping) was applied to all five models to visualise which retinal regions influenced predictions.

| Model | Attention Pattern | Localisation Quality |
|-------|------------------|---------------------|
| EfficientNet-B2 (Baseline) | Diffuse, misaligned | Low |
| EfficientNet-B2 (Refined) | Concentrated | Moderate |
| EfficientNet-B2 (Gated) | Structured, broader | Moderate–High |
| ResNet50-Transformer | Weak, scattered | Low |
| Vision Transformer | Fragmented, distributed | Low |

**Critical finding:** Correct predictions did not always correspond to clinically meaningful attention regions. Several models predicted the correct disease category while focusing on peripheral or non-pathological areas of the retinal image. This highlights that quantitative performance metrics alone are insufficient to evaluate model suitability for clinical deployment.

---

## Repository Structure

```
├── Dissertation__13_.ipynb          # Main Jupyter notebook (all models)
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── data/
│   ├── index_train.csv              # Training split index
│   ├── index_val.csv                # Validation split index
│   └── index_test.csv               # Test split index
├── results/
│   ├── fives_effb2_results.csv      # FIVES results — EfficientNet
│   ├── fives_resnet50_results.csv   # FIVES results — Hybrid
│   ├── fives_vit_results.csv        # FIVES results — ViT
│   └── fives_model_comparison.csv   # FIVES comparison summary
└── models/
    └── (model weights not included — available on request)
```

---

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
Pillow>=9.5.0
tqdm>=4.65.0
timm>=0.9.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/retinal-disease-classification.git
cd retinal-disease-classification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the ODIR-5K dataset** from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) and place images in a local directory

4. **Update the DATA_DIR path** in the notebook to point to your local dataset folder

5. **Run the notebook** — each cell block is clearly labelled with the operation being performed

> **Note:** GPU acceleration is strongly recommended. Training on CPU will be significantly slower. The notebook was developed and tested using a CUDA-enabled GPU.

---

## Limitations

- Models trained on a relatively limited dataset — generalisation to broader and more diverse patient populations requires further validation
- FIVES external evaluation required label simplification to four overlapping categories — true cross-domain performance may differ
- Grad-CAM applied to single representative images per model — a more rigorous interpretability analysis would require systematic evaluation across multiple positive and negative cases per disease class
- No confidence intervals or statistical significance testing reported
- Decision threshold fixed at 0.5 — optimal threshold per class was not explored

---

## Future Work

- **Domain adaptation** — multi-centre datasets or federated learning to improve generalisation
- **Advanced explainability** — attention constraints or DINO-based self-supervised methods for better anatomical alignment
- **Class imbalance** — synthetic data generation (GAN-based) or specialised loss functions for minority classes
- **Clinical validation** — prospective evaluation with clinician involvement and regulatory compliance assessment

---

## Citation

If you use this work, please cite:

```
Kodwani, D. (2026) Retinal Fundus Image Analysis for Multi-Label Ocular 
and Systemic Disease Classification. BSc Dissertation, Anglia Ruskin University.
```

---

## License

This project is for academic purposes only. The ODIR-5K and FIVES datasets are subject to their own licensing terms — please refer to the original dataset sources before use.

---

## Acknowledgements

Supervised by **Dr. Silvia Cirstea**, Anglia Ruskin University. Additional academic support from Dr. Janan Faraz and Dr. Ashim Chakraborty.
