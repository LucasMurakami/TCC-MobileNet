# 📝 Research & Experiment Log: Dual-Perspective Evaluation & AUC-ROC Integration

**Date**: August 24, 2026  
**Project**: Comparative Benchmarking of MobileNet Generations (V1 to V5) on Dermoscopy and Clinical Smartphone Skin Lesions  
**Focus**: Dual-Perspective Evaluation Architecture (Solution 1), AUC-ROC Integration, and Cross-Dataset Taxonomy Harmonization  
**Hardware Environment**: NVIDIA GeForce RTX 5070 (11.9 GB VRAM / BFloat16) / Linux x86_64  

---

## 1. Context & Research Problem

A critical challenge identified in the benchmarking methodology is the taxonomic and distributional discrepancy between the training domain (**HAM10000**, 10,015 dermoscopy images) and the clinical test domain (**PAD-UFES-20**, 2,298 unconstrained smartphone photos):

| Diagnostic Class | HAM10000 (Dermoscopy) | PAD-UFES-20 (Smartphone) | Taxonomy Alignment |
| :--- | :---: | :---: | :--- |
| **Melanoma (MEL)** | 1,113 (11.1%) | 52 (2.3%) | Direct 1:1 match |
| **Basal Cell Carcinoma (BCC)** | 514 (5.1%) | 845 (36.8%) | Direct 1:1 match |
| **Nevus (NV / NEV)** | 6,705 (66.9%) | 244 (10.6%) | Direct 1:1 match |
| **Seborrheic Keratosis (BKL / SEK)** | 1,099 (11.0%) | 257 (11.2%) | Direct 1:1 match |
| **Actinic Keratosis / SCC (AKIEC)** | 327 (3.3%) | 730 (ACK) + 192 (SCC) | Mapped: ACK + SCC $\rightarrow$ AKIEC |
| **Dermatofibroma (DF)** | 115 (1.1%) | 0 (0.0%) | *Absent in PAD-UFES-20* |
| **Vascular Lesions (VASC)** | 142 (1.4%) | 0 (0.0%) | *Absent in PAD-UFES-20* |

### Key Issues with Naive Evaluation:
1. **Missing Classes**: Because PAD-UFES-20 lacks `df` and `vasc`, evaluating a strict 7-class accuracy penalizes the model on classes that physically do not exist in the test domain.
2. **Clinical Misalignment**: In real-world primary care triage, the fundamental clinical requirement is **high-sensitivity Melanoma detection** (minimizing False Negatives) and discriminating malignant lesions from benign ones, rather than equal-weight 7-class categorization.
3. **Missing AUC-ROC Metric**: AUC-ROC was specified as a primary analytical metric in the thesis proposal but had not yet been computed in the benchmark evaluation scripts.

---

## 2. Implemented Architecture: Dual-Perspective Evaluation (Solution 1)

To reconcile 7-class deep representation learning with clinical triage objectives, we implemented a **Dual-Perspective Evaluation Architecture**:

```text
               ┌──────────────────────────────────────────────┐
               │    HAM10000 (10,015 Dermoscopic Images)      │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │  7-Class Feature Representation Pretraining  │
               │  (Focal Loss γ=2.0 + AdamW + BFloat16 AMP)   │
               └──────────────────────┬───────────────────────┘
                                      │
                ┌─────────────────────┴─────────────────────┐
                │                                           │
                ▼                                           ▼
┌───────────────────────────────┐           ┌───────────────────────────────┐
│   Perspective 1: Fine-Grained │           │    Perspective 2: Clinical    │
│    Representation Learning    │           │    Melanoma Triage Mode       │
├───────────────────────────────┤           ├───────────────────────────────┤
│ • 7-Class HAM10000 In-Domain  │           │ • P(MEL) = p_mel              │
│ • 5-Class Harmonized OOD      │           │ • P(Non-MEL) = 1 - p_mel      │
│ • Macro OvR AUC-ROC           │           │ • Binary Melanoma AUC-ROC     │
│ • Per-Class Recall (MEL, BCC) │           │ • Sensitivity at Clinical Th. │
└───────────────────────────────┘           └───────────────────────────────┘
```

### Mathematical Formulation:
1. **Representation Learning (Stage 1 & 2)**:
   The neural network backbones (MobileNet V1–V5) output 7 logits: $\mathbf{z} \in \mathbb{R}^7$.  
   The softmax probability distribution is computed as:
   $$p_c = \frac{e^{z_c}}{\sum_{j=1}^{7} e^{z_j}}, \quad c \in \{\text{akiec}, \text{bcc}, \text{bkl}, \text{df}, \text{mel}, \text{nv}, \text{vasc}\}$$

2. **Melanoma Triage Projection**:
   The binary triage decision space (Melanoma vs. Non-Melanoma) is computed directly from the calibrated probabilities:
   $$\hat{P}(\text{Melanoma}) = p_{\text{mel}}, \quad \hat{P}(\text{Non-Melanoma}) = 1 - p_{\text{mel}}$$
   The Binary Melanoma AUC-ROC is computed via the Wilcoxon-Mann-Whitney formulation:
   $$\text{AUC}_{\text{MEL}} = \frac{1}{N_{\text{mel}} N_{\text{non-mel}}} \sum_{i \in \text{MEL}} \sum_{j \in \text{Non-MEL}} \mathbb{I}(p_{\text{mel}}^{(i)} > p_{\text{mel}}^{(j)})$$

3. **Multi-Class One-vs-Rest (OvR) Macro AUC-ROC**:
   $$\text{Macro AUC} = \frac{1}{|C_{\text{present}}|} \sum_{c \in C_{\text{present}}} \text{AUC}_c$$

4. **Harmonized 5-Class Diagnostic Evaluation**:
   Evaluates performance strictly across the 5 shared diagnostic categories ($C_{\text{shared}} = \{\text{akiec}, \text{bcc}, \text{bkl}, \text{mel}, \text{nv}\}$), eliminating the penalty from absent `df` and `vasc` samples in PAD-UFES-20.

---

## 3. Key Pipeline Modifications & Added Capabilities

### 1. `train_timm_models.py`
- Updated `evaluate_dataset()` to collect full Softmax probability matrices (`all_probs`).
- Integrated automated calculation of:
  - `mel_auc_roc`: Binary Melanoma Triage AUC-ROC.
  - `macro_auc_roc`: Multi-Class OvR Macro-averaged AUC-ROC.
  - `per_class_auc`: Per-class AUC-ROC dictionary for all active classes.
  - `harmonized_5class_acc`: Top-1 accuracy restricted to the 5 shared diagnostic categories.
- Added automated generation of **ROC curves** for in-domain HAM10000, out-of-domain PAD-UFES-20, and a **Dual-Domain ROC Comparison Panel** (`roc_curves_dual_domain.png`).

### 2. `visualize.py`
- Added `plot_roc_curves()`: High-contrast publication-grade ROC curves with per-class AUC annotations and primary melanoma target highlighting.
- Added `plot_dual_roc_comparison()`: Side-by-side dual-panel comparative ROC analysis (HAM10000 Dermoscopy vs. PAD-UFES-20 Smartphone).
- Updated `plot_domain_comparison()`: Includes Melanoma AUC-ROC and Macro AUC-ROC bars alongside Sensitivity and Accuracy.

### 3. `run_scenarios.py`
- Integrated `ham_mel_auc_roc`, `mel_auc_roc` (PAD), and `macro_auc_roc` into the master leaderboard CSV and markdown summary tables.
- Updated console execution logs to report dual-domain Melanoma AUC-ROC in real time.

---

## 4. Next Steps & Recommendations for Thesis Text

1. **Thesis Monograph Update**:
   - Update Chapter 3 (Metodologia) to explain the **Dual-Perspective Framework** (training on 7 classes for rich feature representations while evaluating binary melanoma triage and harmonized 5-class metrics).
   - Re-align mentions of the software stack to PyTorch / `timm` (reflecting native MobileNet V4 and V5 implementations).
2. **Next Milestone**:
   - Implement INT8 Post-Training Quantization (PTQ) export module for Edge Computing validation (PyTorch Mobile / ONNX / TFLite).
   - Measure on-device per-image inference latency (ms) and peak memory footprint across V1–V5.
