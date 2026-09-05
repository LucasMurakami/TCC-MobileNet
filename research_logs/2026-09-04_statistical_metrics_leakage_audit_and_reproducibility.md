# 📝 Research & Experiment Log: Statistical Metrics Refactoring, Zero-Leakage 3-Way Splitting, and Full Pipeline Provenance

**Date**: September 4, 2026  
**Project**: Comparative Benchmarking of MobileNet Generations (V1 to V5) on Dermoscopy and Clinical Smartphone Skin Lesions  
**Focus**: Decoupled Metrics Engine (`metrics.py`), Cryptographic 3-Way Lesion-Grouped Splitting (`split_manifest.json`), Cross-Domain Threshold Degradation Analysis (`pad_oracle`), and Automated Provenance Tracking  
**Branch**: `test-branch` (commit `b957a5a`)  
**Hardware Environment**: NVIDIA GeForce RTX 5070 (11.9 GB VRAM / BFloat16) / Linux x86_64  

---

## 1. Motivation & Identified Methodological Gaps

Following initial dual-domain benchmark runs, four significant architectural and methodological opportunities were addressed to bring the experimental pipeline to top-tier academic and thesis rigor:

1. **Monolithic Metrics Tight Coupling**:
   - Diagnostic evaluation logic (Youden thresholding, continuous AUC calculation, multi-class reports) was previously intertwined inside the 950+ line `train_timm_models.py` training script.
   - **Resolution**: Extract all statistical and clinical evaluation logic into a standalone, pure, fully tested [`metrics.py`](../metrics.py) module.

2. **Validation-to-Test Data Snooping (2-Way vs. 3-Way Split)**:
   - In earlier iterations, HAM10000 was split into only two partitions: Train and Validation. Consequently, the validation set was utilized simultaneously for early stopping, threshold calibration, *and* final in-domain reporting.
   - **Resolution**: Implement a true **70% Train / 10% Tuning Validation / 20% Held-Out Test** lesion-grouped split. Operating thresholds are calibrated strictly on validation, while in-domain performance is reported exclusively on the untouched test partition.

3. **Verifiable Auditability & Zero-Leakage Guarantee**:
   - Although grouped splitting by `lesion_id` was intended, there was no cryptographic proof or audit log to guarantee that multiple dermoscopic images of the same lesion did not cross partition boundaries.
   - **Resolution**: Implement SHA256 partition hashing and create an immutable audit trail (`split_manifest.json`).

4. **Quantifying Cross-Domain Operating Point Degradation**:
   - Applying dermoscopy-calibrated decision thresholds ($\tau_{\text{HAM}}$) directly to smartphone images often led to high specificity (>93%) but reduced sensitivity due to lower model confidence on unstandardized smartphone photography.
   - **Resolution**: Implement an empirical **Oracle Thresholding** pass (`pad_oracle`) alongside the transferred threshold to isolate domain calibration loss from feature representation loss.

---

## 2. Architecture of the New Pipeline Components

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 New Pipeline Architecture                                   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                        ┌──────────────────────┴──────────────────────┐
                        ▼                                             ▼
       ┌─────────────────────────────────┐           ┌─────────────────────────────────┐
       │   1. Data & Provenance Layer    │           │   2. Modular Evaluation Engine  │
       ├─────────────────────────────────┤           ├─────────────────────────────────┤
       │ • dataset.py                    │           │ • metrics.py                    │
       │ • 70/10/20 Lesion-Grouped Split │           │ • evaluate_binary_triage        │
       │ • split_manifest.json (SHA256)  │           │ • restricted_class_accuracy (5) │
       │ • validate_image_paths (PIL)    │           │ • bootstrap_metric_ci (95% CI)  │
       │ • provenance.json (Git + Env)   │           │ • pad_oracle comparative pass   │
       └─────────────────────────────────┘           └─────────────────────────────────┘
                        │                                             │
                        └──────────────────────┬──────────────────────┘
                                               ▼
       ┌───────────────────────────────────────────────────────────────────────────────┐
       │                        3. Standardized Orchestration                          │
       ├───────────────────────────────────────────────────────────────────────────────┤
       │ • main.py / run_scenarios.py: Typed RunConfig Dataclass & Priority Hierarchy   │
       │ • train_timm_models.py: Stage 1 Warmup ➔ Stage 2 Fine-Tune (BFloat16 / RTX5070) │
       │ • tests/: Automated Unit Test Suite (pytest)                                  │
       └───────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technical Implementation Details

### A. Decoupled Evaluation & Metrics Engine ([`metrics.py`](../metrics.py))
- **Calibrated Binary Triage (`evaluate_binary_triage`)**:
  - Calculates operating points across fixed reference thresholds ($\tau \in [0.50, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02]$).
  - Dynamically computes optimal operating thresholds via Youden's $J$-statistic ($J = \text{TPR} - \text{FPR}$), 90% Sensitivity (`sens90`), or 95% Sensitivity (`sens95`).
  - Tracks threshold provenance: `threshold_source: "youden" | "sens90" | "float" | "fallback_default" | "clamped"`.
  - Implements safety clamping to $[0.01, 0.90]$ to avoid numerical extremes.
- **Restricted 5-Class Evaluation (`restricted_class_accuracy`)**:
  - HAM10000 spans 7 classes, but PAD-UFES-20 contains only 5 shared classes (`akiec`, `bcc`, `bkl`, `mel`, `nv`).
  - Evaluates target domain predictions conditionally across shared indices to prevent spurious penalties for missing classes (`df`, `vasc`).
- **Cluster-Aware Bootstrap Confidence Intervals (`bootstrap_metric_ci`)**:
  - Resamples patient/lesion clusters with replacement ($N = 1,000$ iterations, fixed seed 42) to produce non-parametric 95% confidence intervals ($2.5^{\text{th}}$ to $97.5^{\text{th}}$ percentiles) for all primary diagnostic metrics.

---

### B. Zero-Leakage 3-Way Partitioning ([`dataset.py`](../dataset.py))
- **Algorithm (`grouped_stratified_split`)**:
  ```python
  train_val_df, test_df = grouped_stratified_split(df, group_col='lesion_id', stratify_col='dx', test_size=0.20, random_state=42)
  train_df, val_df = grouped_stratified_split(train_val_df, group_col='lesion_id', stratify_col='dx', test_size=0.125, random_state=42)
  ```
  - **Resulting Partitions**:
    - **Train**: 7,012 images (70%)
    - **Tuning Validation (HAM)**: 1,008 images (10%)
    - **Held-Out Test (HAM)**: 1,995 images (20%)
    - **Out-of-Domain (PAD-UFES-20)**: 2,298 images
  - **Leakage Verification**: Explicit check verifying $\text{train} \cap \text{val} = \emptyset$, $\text{train} \cap \text{test} = \emptyset$, and $\text{val} \cap \text{test} = \emptyset$.
  - **Cryptographic Audit Manifest (`split_manifest.json`)**:
    Computes deterministic SHA256 checksums of sorted image IDs per partition:
    $$\text{Hash}_{\text{all}} = \text{SHA256}(\text{image\_id}_1 \parallel \text{image\_id}_2 \parallel \dots)$$

---

### C. Validation vs. Testing Separation ([`train_timm_models.py`](../train_timm_models.py))
1. **Stage 1 (Linear Warmup)**: Trains classification head for 3 epochs with frozen backbone ($lr = 1\times 10^{-3}$).
2. **Stage 2 (Full Fine-Tuning)**: Unfreezes all layers with reduced learning rate ($lr = 1\times 10^{-4}$). Early stopping monitors `ham_val_mel_auc_roc` with patience = 3 or 4.
3. **In-Domain Evaluation**: Restores best model checkpoint from validation peak, then calibrates triage thresholds ($\tau_{\text{mel}}, \tau_{\text{bcc}}, \tau_{\text{mal}}$) on `ham_val`. Evaluates final unbiased in-domain metrics on `ham_test`.
4. **Out-of-Domain Evaluation**: Evaluates `pad_val` using the HAM-calibrated thresholds. Additionally runs `pad_oracle` to measure maximum attainable transfer performance under target domain calibration.

---

### D. Typed Configuration & Provenance Capture ([`main.py`](../main.py))
- **`@dataclass class RunConfig`**: Replaces loose dictionary lookups with typed schema definitions.
- **`write_provenance`**: Dumps an environment stamp into every model run:
  - Installed packages and versions (`torch`, `torchvision`, `timm`, `numpy`, `pandas`, `scikit-learn`).
  - Git commit SHA and working directory dirty state.
  - Hardware specifications (GPU model, VRAM allocation, BFloat16 support).
  - SHA256 hash reference to `split_manifest.json`.

---

### E. Robust Dataset Ingestion & Image Integrity Validation ([`dataset.py`](../dataset.py))
- **Broken Image Screening (`validate_image_paths`)**:
  - Validates image file existence and executes header checks using PIL verify during initialization.
  - Automatically filters out corrupted entries with explicit logging rather than triggering hard training crashes mid-epoch.
- **PAD-UFES-20 Clinical Harmonization**:
  - Enforces explicit clinical mapping: `ack` $\to$ `akiec`, `bcc` $\to$ `bcc`, `mel` $\to$ `mel`, `nev` $\to$ `nv`, `scc` $\to$ `akiec`, `sek` $\to$ `bkl`.
  - Captures `lesion_id` and `patient_id` metadata from smartphone metadata tables to support future hierarchical mixed-effects modeling and patient-level clustering.
- **Mixup Parameterization**:
  - Replaced ambiguous minority-only mixup with standardized batch mixup ($\alpha = 0.20$), uniformly blending convex combinations during Stage 1 and Stage 2 training.

---

### F. Fault-Tolerant Orchestration & Resume ([`run_scenarios.py`](../run_scenarios.py))
- **Stateful Resumption (`is_experiment_completed`)**:
  - Inspects `results.json` in destination model run folders.
  - Automatically skips already completed experiments when recovering from unexpected hardware interrupts or timeouts.
- **Automated Summary Reporting**:
  - Generates consolidated session artifacts after every model run:
    - `master_leaderboard.csv`: Multi-metric comparison spreadsheet.
    - `SUMMARY.md`: Markdown summary table formatted for thesis documentation.
    - `summary_comparison.json`: Complete machine-readable hierarchical benchmark records.
- **Stream Redirection & Unbuffered Logging**:
  - Pipeline logs synchronously stream to `experiments/<session_id>/execution.log` under `PYTHONUNBUFFERED=1` to allow real-time terminal following without output buffer lag.

---

### G. Automated Pytest Unit Test Suite (`tests/`)
A dedicated automated testing suite was introduced under `tests/` to guarantee pipeline reproducibility and guard against regressions:
- [`tests/test_config.py`](../tests/test_config.py): Validates `RunConfig` dataclass defaults, JSON serialization, and CLI override priority.
- [`tests/test_dataset.py`](../tests/test_dataset.py): Validates grouped stratified splitting logic, mathematically verifies zero lesion overlap across all 3 partitions ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$), and verifies deterministic SHA-256 partition hashing.
- [`tests/test_metrics.py`](../tests/test_metrics.py): Validates Youden thresholding, continuous AUC calculation, restricted 5-class accuracy, and non-parametric bootstrap confidence interval estimation.
- [`tests/test_smoke.py`](../tests/test_smoke.py): Executes a synthetic fast end-to-end forward/backward training pass to verify loss computation and gradient backpropagation.

---

### H. Resolution of the Logit-Adjustment vs Balanced-Sampling Conflict
- In earlier benchmark iterations (`03_09_2026`), setting `logit_adjust: 1.0` simultaneously with `balanced_sampling: true` caused severe logit collapse, driving HAM10000 in-domain accuracy down to 8%–20%.
- **Root Cause**: Menon et al. logit adjustment adds class prior offsets $\log(\pi_y)$ under the assumption of an empirical long-tailed training distribution. When combined with dynamic balanced sampling (which already resamples all 7 classes uniformly to $1/K$), the model was double-penalizing majority classes and artificially over-boosting minority classes.
- **Resolution**: Disabling logit adjustment (`logit_adjust: 0.0`) when balanced sampling is active restored in-domain test accuracy to **77.5% – 81.3%** across all MobileNet architectures.

---

## 4. Empirical Verification: Single Model Smoke Run (`04_09_2026_v1_test`)

A verification run of MobileNet V1 (`low` scenario) was conducted to test the complete end-to-end integration:

| Metric | HAM10000 Test (In-Domain) | PAD-UFES-20 (Out-of-Domain) |
| :--- | :---: | :---: |
| **Accuracy** | **76.84%** | **21.50%** *(5-Class Restricted: 22.11%)* |
| **Melanoma AUC-ROC [95% CI]** | **0.8703** *[0.8435, 0.8957]* | **0.7358** *[0.6583, 0.8091]* |
| **BCC AUC-ROC [95% CI]** | **0.9717** *[0.9602, 0.9819]* | **0.6513** *[0.6260, 0.6763]* |
| **Macro AUC-ROC [95% CI]** | **0.9392** *[0.9289, 0.9480]* | **0.6618** *[0.6405, 0.6826]* |
| **Melanoma Triage Sensitivity** | **78.44%** *(Spec: 76.76%, $\tau=0.18$)* | **23.08%** *(Spec: 93.68%, $\tau=0.18$)* |
| **Melanoma Oracle Sensitivity** | — | **57.69%** *(Spec: 80.37%, $\tau=0.10$)* |
| **Malignancy Screening Recall** | **89.47%** *(Spec: 69.54%, $\tau=0.25$)* | **57.50%** *(Spec: 78.29%, $\tau=0.25$)* |

### Key Findings from Verification:
1. **Threshold Shift Quantification**: Under fixed HAM threshold ($\tau = 0.18$), PAD sensitivity is only 23.08% due to lower confidence scores on smartphone images. Under oracle calibration ($\tau = 0.10$), sensitivity surges to **57.69%** at 80.37% specificity.
2. **Execution Stability**: Full dual-domain pipeline completed in 3.5 minutes, generating all ROC curves, confusion matrices, Grad-CAM galleries, and 95% bootstrap confidence intervals with zero errors.

---

## 5. Live Production Benchmark Suite (Session `04_09_2026`)

The full benchmark across all scenarios (`STANDARD`, `MEDIUM`, `LOW`) was initiated with the complete 15-model matrix. Preliminary results from the completed models in the `STANDARD` scenario (20 Epochs, Patience 4):

| Architecture | In-Domain Test Acc (HAM) | OOD Acc (PAD Phone) | Domain Gap | In-Domain Mel AUC [95% CI] | PAD Mel AUC [95% CI] | Mel AUC Gap | Phone Mel Triage Sens ($\tau_{\text{HAM}}$) | Phone Malignancy Screening Sens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNet V3** | **79.35%** | **25.50%** | **53.85%** *(Lowest)* | **0.8715** *[0.8424, 0.8964]* | **0.7519** *[0.6833, 0.8172]* | **0.1196** *(Smallest)* | **57.69%** ($\tau=0.19$) | **69.71%** |
| **MobileNet V2** | **81.30%** *(Highest)* | 22.11% | 59.20% | 0.8679 *[0.8367, 0.8955]* | 0.7423 *[0.6684, 0.8137]* | 0.1256 | 32.69% ($\tau=0.20$) | 53.77% |
| **MobileNet V1** | 79.35% | 22.32% | 57.02% | 0.8639 *[0.8348, 0.8913]* | 0.7152 *[0.6360, 0.7975]* | 0.1487 | 51.92% ($\tau=0.11$) | 54.92% |
| **MobileNet V4** | 77.54% | 19.54% | 58.01% | 0.8693 *[0.8395, 0.8976]* | 0.6910 *[0.6121, 0.7705]* | 0.1783 | 38.46% ($\tau=0.26$) | **82.96%** *(Highest)* |
| **MobileNet V5** | *Training* | *Training* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |

---

## 6. Pipeline Terminal Monitoring & Telemetry Commands

For real-time terminal surveillance and benchmark provenance tracking, researchers can execute the following standard CLI commands:

```bash
# 1. Real-time log stream (tail follow)
tail -f experiments/04_09_2026/execution.log

# 2. Inspect the last 50 lines with continuous streaming
tail -n 50 -f experiments/04_09_2026/execution.log

# 3. View the live-updating Markdown leaderboard
watch -n 10 "cat experiments/04_09_2026/SUMMARY.md"

# 4. View tabular CSV leaderboard formatted for terminal
column -s, -t < experiments/04_09_2026/master_leaderboard.csv | less -S

# 5. Monitor GPU utilization, VRAM allocation, and thermal states
watch -n 1 nvidia-smi
```

