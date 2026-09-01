# 📝 Research & Experiment Log: MobileNetV5 LayerCAM & Operating Threshold Calibration

**Date**: August 31, 2026  
**Project**: Comparative Benchmarking of MobileNet Generations (V1 to V5) on Dermoscopy and Clinical Smartphone Skin Lesions  
**Focus**: MobileNetV5 MSFA LayerCAM Implementation, Decision Threshold Tuning Engine, Prior Shift Analysis, and Cross-Domain Triage Calibration  
**Hardware Environment**: NVIDIA GeForce RTX 5070 (11.9 GB VRAM / BFloat16) / Linux x86_64  

---

## 1. Context & Identified Challenges

Following full dual-domain benchmark runs on the MobileNet lineage (V1 to V5), two critical technical and methodological challenges emerged:

### 1. MobileNetV5 Spatial Saliency & Grad-CAM Artifacts:
- MobileNetV5 (`mobilenetv5_300m.gemma3n`) is a hybrid foundation vision model utilizing multi-axis token mixers and a Multi-Scale Feature Aggregation (MSFA) module.
- Standard Grad-CAM computed global channel average gradients, which resulted in scattered, diffuse heatmaps with token grid artifacts across non-lesion skin regions.
- **Requirement**: Implement a specialized saliency extraction method for V5 based on Jacob Gil's PyTorch-GradCAM framework without altering the validated Grad-CAM pipelines for MobileNet V1–V4.

### 2. The Multi-Class Argmax Prior Shift Barrier on PAD-UFES-20:
- When evaluating on out-of-domain smartphone photos (PAD-UFES-20), standard multi-class $\text{argmax}$ yielded low Melanoma recall (~17%–33%), despite high continuous AUC-ROC (0.7440 on V5).
- **Cause**: Extreme class prior shift between training (HAM10000: 67% Nevus, 11% Melanoma) and testing (PAD-UFES-20: 36.8% BCC, 40.1% Actinic Keratosis, only **2.3% / 52 Melanomas**).
- When a smartphone photo yields $p_{\text{mel}} = 0.35$ ($15\times$ higher than baseline prevalence), $\text{argmax}$ defaults to BCC or Nevus if $p_{\text{bcc}} = 0.40$, marking the case as a False Negative.
- **Requirement**: Build a configurable Decision Threshold Tuning engine into the pipeline to decouple high-stakes malignant lesion screening from the uncalibrated 7-class $\text{argmax}$ competition.

---

## 2. Technical Architecture & Implementations

```text
                                     ┌──────────────────────────────────────────────┐
                                     │         MobileNet V1–V5 Benchmarks           │
                                     └──────────────────────┬───────────────────────┘
                                                            │
                            ┌───────────────────────────────┴───────────────────────────────┐
                            ▼                                                               ▼
        ┌──────────────────────────────────────┐                        ┌──────────────────────────────────────┐
        │  1. MSFA LayerCAM for MobileNetV5    │                        │  2. Operating Threshold Engine       │
        ├──────────────────────────────────────┤                        ├──────────────────────────────────────┤
        │ • Target: raw_model.msfa.norm        │                        │ • CLI Flag: --mel-threshold          │
        │ • Shape: (1, 2048, 16, 16)           │                        │ • Preset in benchmark_scenarios.json │
        │ • Elementwise Pos-Grad Weighting:    │                        │ • Calibration: Youden, Sens90, Sens95│
        │   CAM = ReLU(sum(max(w,0) * A))      │                        │ • Zero-Shot In-to-Out Domain Transfer│
        │ • High contrast, sharp localization  │                        │ • Multi-Threshold Triage Curves      │
        └──────────────────────────────────────┘                        └──────────────────────────────────────┘
```

---

### 1. MobileNetV5 Jacob Gil MSFA LayerCAM (`visualize.py`)

1. **Target Layer Identification**:
   - Traced the forward computation graph of `mobilenetv5_300m.gemma3n`.
   - Identified that deep multi-scale spatial representations consolidate at `model.msfa.norm` (output tensor shape: `(1, 2048, 16, 16)`), preserving high spatial fidelity before pooling.
2. **LayerCAM Mathematical Formulation**:
   - Instead of spatially averaging gradients across the feature map ($\alpha_k = \frac{1}{Z}\sum_{i}\sum_{j} \frac{\partial Y^c}{\partial A_{ij}^k}$), LayerCAM weights each spatial activation elementwise by its positive gradient:
     $$w_{ij}^k = \max\left(\frac{\partial Y^c}{\partial A_{ij}^k}, \, 0\right)$$
     $$L_{\text{LayerCAM}}^c = \text{ReLU}\left(\sum_{k=1}^K w_{ij}^k \odot A_{ij}^k\right)$$
3. **Results**:
   - Eliminates diffuse background noise and resolves crisp boundary localization on dermoscopic and smartphone lesions.
   - MobileNet V1, V2, V3, and V4 Grad-CAM hooks remain strictly preserved and frozen.

---

### 2. Operating Sensitivity Threshold Tuning Engine (`train_timm_models.py` & `main.py`)

1. **Configuration Hierarchy (CLI Overrides JSON Always)**:
   $$\text{Threshold Source} = \text{CLI Flag} \;\succ\; \text{Model JSON Config} \;\succ\; \text{Scenario JSON Config} \;\succ\; \text{Default (0.15)}$$

2. **Supported Threshold Calibration Strategies**:
   - **`--mel-threshold <float>`** (e.g., `0.15`, `0.10`): Sets an explicit operating probability cutoff $\tau$.
   - **`--mel-threshold youden`** (or `auto`): Automatically computes Youden's Index on the In-Domain ROC curve:
     $$J(\tau) = \text{Sensitivity}(\tau) + \text{Specificity}(\tau) - 1 \implies \tau^* = \arg\max_\tau J(\tau)$$
   - **`--mel-threshold sens90`**: Calibrates the maximum threshold $\tau$ that guarantees $\ge 90\%$ Sensitivity on In-Domain validation.
   - **`--mel-threshold sens95`**: Calibrates the threshold for ultra-safe triage ($\ge 95\%$ Sensitivity).

3. **Zero-Shot Domain Transfer of Calibrated Thresholds**:
   - The optimal threshold $\tau_{\text{ham}}^*$ calibrated on In-Domain HAM10000 is transferred zero-shot to PAD-UFES-20 to simulate realistic real-world deployment without fine-tuning on smartphone targets.

4. **Multi-Threshold Reference Profiling**:
   - The evaluation engine logs complete triage operating points across:
     $$\tau \in [0.50, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02]$$
   - All metrics (`mel_triage_recall`, `mel_triage_spec`, `mel_triage_detected`, `mel_operating_points`) are automatically serialized into `results.json`, `master_leaderboard.csv`, and `SUMMARY.md`.

---

## 3. Empirical Findings & Mathematical Validation

### 1. Empirical Youden & PR Curve Calibration on Trained MobileNet Models

| Class | Clinical Classification | **Youden $\tau$ (ROC Curve)** | **Max-$F_1$ $\tau$ (PR Curve)** | In-Domain Sensitivity at Youden | In-Domain Specificity at Youden |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MEL** | Malignant (High Risk) | **0.1602** | 0.6553 | **91.0%** | **80.7%** |
| **BCC** | Malignant (Moderate Risk) | **0.1158** | 0.4736 | **93.2%** | **97.1%** |
| **AKIEC** | Pre-Malignant | **0.0110** | 0.4835 | **96.9%** | **90.8%** |
| **NV** | Benign (Dominant Class) | **0.3549** | 0.2185 | **93.7%** | **86.6%** |
| **BKL** | Benign | **0.1420** | 0.3450 | **91.4%** | **90.0%** |
| **DF** | Benign | **0.0297** | 0.7695 | **100.0%** | **98.5%** |
| **VASC** | Benign | **0.0322** | 0.5756 | **96.4%** | **99.6%** |

*Key Takeaway*: The optimal Youden threshold for Melanoma on In-Domain dermoscopy is **$\tau \approx 0.16$**, yielding $91.0\%$ sensitivity and $80.7\%$ specificity.

---

### 2. Impact on Out-of-Domain Smartphone Photos (PAD-UFES-20)

When testing MobileNet V5 on the 2,298 unconstrained smartphone images of PAD-UFES-20:

| Operating Threshold $\tau$ | Melanoma Sensitivity | Specificity | Melanomas Detected | Clinical Meaning |
| :--- | :---: | :---: | :---: | :--- |
| $\tau \ge 0.50$ (Default Argmax) | 17.3% | 98.4% | 9 / 52 | Suppressed by 7-class prior shift |
| $\tau \ge 0.15$ (Balanced Triage) | **63.5%** | 76.2% | **33 / 52** | $3.7\times$ sensitivity increase |
| $\tau \ge 0.10$ (Primary Care Screening) | **82.7%** | 61.4% | **43 / 52** | Catches $>8$ out of 10 smartphone melanomas |
| $\tau \ge 0.05$ (High-Sensitivity Triage) | **94.2%** | 42.1% | **49 / 52** | Ultra-safe teledermatology referral cutoff |

---

## 4. Summary of Codebase Modifications

1. **[`train_timm_models.py`](file:///home/lkm20/TCC/train_timm_models.py)**:
   - Implemented `evaluate_dataset()` threshold evaluation parameters, Youden's Index calculation, Sensitivity-target optimization, and triage point tracking.
   - Updated `train_single_model()` and `evaluate_dual_domain()` to record and display calibrated triage sensitivity alongside standard argmax metrics.
2. **[`main.py`](file:///home/lkm20/TCC/main.py)**:
   - Added `--mel-threshold` CLI argument to `argparse`.
   - Enforced CLI override hierarchy over `benchmark_scenarios.json`.
   - Updated `update_session_leaderboard()` to write triage columns into `master_leaderboard.csv` and `SUMMARY.md`.
3. **[`benchmark_scenarios.json`](file:///home/lkm20/TCC/benchmark_scenarios.json)**:
   - Added `"mel_threshold": 0.15` default across all scenario presets (`standard`, `medium`, `low`, `maximum`).
4. **[`visualize.py`](file:///home/lkm20/TCC/visualize.py)**:
   - Implemented Jacob Gil LayerCAM on `raw_model.msfa.norm` for MobileNetV5.
