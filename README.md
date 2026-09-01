# Skin Lesion Classification: MobileNet Generations (V1 – V5) Benchmark & Interpretability Suite

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20%7C%20CUDA%2013-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-ImageNet%20%26%20Gemma-34D058)](https://github.com/huggingface/pytorch-image-models)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20Blackwell%20%7C%20RTX%205070%20%7C%20BF16-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com)
[![Validation](https://img.shields.io/badge/Validation-HAM10000%20%2B%20PAD--UFES--20-007ACC)](https://data.mendeley.com/datasets/zr7vgbcyr2/1)

Comprehensive deep learning benchmark framework comparing lightweight convolutional and foundation vision architectures across five generations of MobileNet (**MobileNet V1, V2, V3 Small, V3 Large, V4 Conv-Medium, V4 Conv-Large, and V5 300M**) on dermoscopic and clinical smartphone skin lesions.

---

## 📌 Project Overview & Research Scope

- **Primary In-Domain Training Dataset**: [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) (10,015 dermoscopy images across 7 diagnostic categories).
- **Out-of-Domain (OOD) Validation Dataset**: [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1) (2,298 unconstrained clinical smartphone photos collected from 1,373 Brazilian patients in Espírito Santo to evaluate teledermatology domain transfer and dataset shift).

### Cross-Dataset Taxonomy Harmonization:

| Diagnostic Class | HAM10000 (Dermoscopy) | PAD-UFES-20 (Smartphone) | Taxonomy Alignment & Mapping |
| :--- | :---: | :---: | :--- |
| **Melanoma (MEL)** | 1,113 (11.1%) | 52 (2.3%) | **Direct 1:1 Match** (Primary Target for Triage) |
| **Basal Cell Carcinoma (BCC)** | 514 (5.1%) | 845 (36.8%) | **Direct 1:1 Match** |
| **Nevus (NV / NEV)** | 6,705 (66.9%) | 244 (10.6%) | **Direct 1:1 Match** |
| **Seborrheic Keratosis (BKL / SEK)** | 1,099 (11.0%) | 257 (11.2%) | **Direct 1:1 Match** |
| **Actinic Keratosis / SCC (AKIEC)** | 327 (3.3%) | 730 (ACK) + 192 (SCC) | **Harmonized**: `ACK` + `SCC` $\rightarrow$ `AKIEC` |
| **Dermatofibroma (DF)** | 115 (1.1%) | *0 (0.0%)* | Absent in PAD-UFES-20 |
| **Vascular Lesions (VASC)** | 142 (1.4%) | *0 (0.0%)* | Absent in PAD-UFES-20 |

---

## 🔬 Dual-Perspective Evaluation Framework (Solution 1)

To reconcile 7-class deep representation learning with clinical screening and triage objectives, the framework evaluates models across two complementary perspectives:

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

1. **Perspective 1 — Fine-Grained Representation Learning**:
   - Trains backbones on all 7 HAM10000 classes to force convolutional and attention filters to learn nuanced morphological differences (pigment networks, vascular patterns, keratin structures).
   - Evaluates **Macro One-vs-Rest (OvR) AUC-ROC** and **Harmonized 5-Class Accuracy** on PAD-UFES-20 (bypassing penalties from absent classes `df`/`vasc`).

2. **Perspective 2 — Clinical Melanoma Triage Mode**:
   - Mathematically projects the 7-class probability simplex into binary triage space:
     $$\hat{P}(\text{Melanoma}) = p_{\text{mel}}, \quad \hat{P}(\text{Non-Melanoma}) = 1 - p_{\text{mel}}$$
   - Computes **Binary Melanoma AUC-ROC (`mel_auc_roc`)**, **Melanoma Sensitivity (Recall)**, and **Specificity** on both dermoscopy and clinical smartphone photos.

---

## 🏗️ Repository Architecture

```text
├── main.py                     # Unified CLI entrypoint, dataset orchestrator & benchmark summary logger
├── train_timm_models.py        # Core PyTorch + timm backbone engine, Focal Loss & Dual-Domain training loop
├── dataset.py                  # Dataset loaders, stratified splits, oversampling & PAD-UFES taxonomy mapping
├── run_scenarios.py            # Automated scenario runner (Maximum -> Medium -> Low) with date-versioned isolation
├── run_grid_search.py          # Multi-hyperparameter grid search orchestrator with auto-resume
├── visualize.py                # ROC curves, dual confusion matrices, Grad-CAM heatmaps & domain comparison charts
├── benchmark_scenarios.json    # Standardized hyperparameter configurations per scenario tier
├── infrastructure.md           # Mathematical and hardware architecture documentation
├── research_logs/              # Daily experimental logs and methodology documentation
│   ├── 2026-08-18_benchmark_summary.md
│   └── 2026-08-24_dual_perspective_auc_roc_and_taxonomy_harmonization.md
├── experiments/                # Date & session-isolated experiment runs
│   ├── GLOBAL_ARCHIVE_INDEX.md # Master catalog of all dated runs
│   └── 20_08_2026/             # Daily benchmark folder (leaderboard, checkpoints, curves, heatmaps)
└── data_cache/                 # Local cache for raw datasets (HAM10000, PAD-UFES-20)
```

---

## 📦 Supported Model Architectures (1 Flagship per Generation)

| CLI Model Flag | Architecture | Input Res. | Backbone / Pretrained Checkpoint | Primary Architectural Innovation |
|---|---|:---:|---|---|
| `--model v1` | **MobileNet V1** | 224 × 224 | ImageNet-1k (`mobilenetv1_100`) | Depthwise Separable Convolutions baseline |
| `--model v2` | **MobileNet V2** | 224 × 224 | ImageNet-1k (`mobilenetv2_100`) | Inverted Residuals & Linear Bottlenecks |
| `--model v3` | **MobileNet V3** | 224 × 224 | ImageNet-1k (`mobilenetv3_large_100`) | Hardware-Aware NAS + Squeeze-and-Excitation |
| `--model v4` | **MobileNet V4** | 256 × 256 | ImageNet-1k (`mobilenetv4_conv_medium`) | Universal Inverted Bottleneck (UIB) |
| `--model v5` | **MobileNet V5** | 256 × 256 | Google Gemma3n (`mobilenetv5_300m.gemma3n`) | Multi-Scale Feature Aggregation (MSFA) Foundation Backbone |

---

## 🚀 Quickstart & Unified CLI Benchmark Runner (`main.py`)

The pipeline features a unified CLI entrypoint [`main.py`](file:///home/lkm20/TCC/main.py) with dynamic scenario loading from [`benchmark_scenarios.json`](file:///home/lkm20/TCC/benchmark_scenarios.json), automatic hardware environment detection, date-versioned directory isolation, and global archive indexing.

### 1. Benchmark Execution Examples

```bash
# Run the complete benchmark on all 5 MobileNet generations under the Standard scenario
python main.py --scenario standard --model all

# Train only MobileNet V5 with Balanced Batch Sampling and Youden Threshold Calibration
python main.py --scenario standard --model v5 --balanced-sampling --mel-threshold youden

# Train MobileNet V4 with Logit Adjustment and Minority Mixup
python main.py --scenario medium --model v4 --logit-adjust 1.0 --mixup-minority 0.2

# Custom Fast Exploration Run
python main.py --scenario low --model v3 --epochs 15 --batch-size 32 --mel-threshold 0.15
```

---

## 🛠️ Complete CLI Options & Parameters Reference (`main.py --help`)

| Parameter | Type / Choices | Default | Description |
| :--- | :--- | :---: | :--- |
| `--model` | `v1, v2, v3, v4, v5, all` | `v1` | Model generation architecture to train, or `all` to benchmark all 5 generations. |
| `--scenario` | `standard, medium, low, maximum, custom` | `standard` | Preset from `benchmark_scenarios.json`. |
| `--scenarios-file` | `str` | `benchmark_scenarios.json` | Path to benchmark scenarios configuration JSON file. |
| `--session-dir` | `str` | `experiments/<DD_MM_YYYY>` | Custom date/session output directory. |
| `--epochs` | `int` | *from scenario* | Max epochs for Stage 2 (overrides scenario preset). |
| `--patience` | `int` | *from scenario* | Early stopping patience (overrides scenario preset). |
| `--batch-size` | `int` | *from scenario* | Target effective batch size (adaptive micro-batching). |
| `--lr-stage1` | `float` | *model default* | Stage 1 warmup learning rate (Head only). |
| `--lr-stage2` | `float` | *model default* | Stage 2 fine-tuning learning rate (Backbone). |
| `--mel-threshold` | `float, auto, youden, sens90, sens95` | `0.15` | Operating threshold for Melanoma clinical triage. CLI always overrides JSON. |
| `--balanced-sampling`| *flag* | `False` | Enables `WeightedRandomSampler` to eliminate 67% Nevus mini-batch dominance. |
| `--logit-adjust` | `float` | `0.0` | Post-hoc Bayesian logit adjustment strength $\tau$ (*ICLR 2021*) to strip prior bias. |
| `--mixup-minority` | `float` | `0.0` | Beta-distribution $\alpha$ parameter for synthetic minority class Mixup. |
| `--val-dataset` | `ham10000, pad-ufes-20, both` | `both` | Target validation evaluation scope. |
| `--seed` | `int` | `42` | Global random seed for PyTorch, NumPy, and CUDA. |

---

## ⚖️ Class Imbalance Mitigation Suite (Long-Tail Learning)

To combat the severe 67% Nevus dominance in HAM10000 and the 37% BCC / 2.3% Melanoma distribution shift in PAD-UFES-20, three modular solutions are integrated:

```text
┌───────────────────────────────────────┬───────────────────────────────────────┬───────────────────────────────────────┐
│ 1. Balanced Batch Sampling (Sol. A)   │ 2. Post-Hoc Logit Adjustment (Sol. B) │ 3. Minority Mixup Augmentation (Sol. C)│
├───────────────────────────────────────┼───────────────────────────────────────┼───────────────────────────────────────┤
│ Flag: --balanced-sampling             │ Flag: --logit-adjust 1.0              │ Flag: --mixup-minority 0.2            │
│ Samples each class with equal ~14.3%  │ Strips dataset prior penalty:         │ Interpolates rare malignant lesions   │
│ probability per mini-batch.           │ z_adj = z - tau * log(pi)             │ with diverse skin contexts.           │
└───────────────────────────────────────┴───────────────────────────────────────┴───────────────────────────────────────┘
```

---

## 📊 Visual & Analytical Artifacts Generated

Each benchmark execution automatically compiles comprehensive clinical evaluation artifacts:

- **`roc_curves_dual_domain.png`**: Side-by-side comparative ROC analysis contrasting in-domain (HAM10000 dermoscopy) against out-of-domain (PAD-UFES-20 smartphone) discrimination.
- **`roc_curves.png`**: Multi-class One-vs-Rest ROC curves with per-class AUC-ROC and highlighted Melanoma triage curve.
- **`domain_comparison.png`**: Side-by-side comparative bar chart tracking Accuracy, F1, Melanoma Recall, Melanoma AUC-ROC, and Macro AUC.
- **`confusion_matrix.png`**: Dual matrices showing raw sample counts and row-normalized sensitivity percentages.
- **`per_class_metrics.png`**: Diagnostic Precision, Recall, and F1-score breakdown per class.
- **`gradcam_heatmaps.png`**: Multi-column attention gallery (Grad-CAM for V1–V4, Jacob Gil MSFA LayerCAM for V5).
- **`training_curves.png`**: Multi-stage loss and accuracy trajectory over training epochs.
- **`results.json` & `classification_report.json`**: Exact numerical scores including `mel_auc_roc`, `macro_auc_roc`, `mel_triage_recall`, `pad_mel_triage_detected`, and per-class metrics.
- **`master_leaderboard.csv` & `SUMMARY.md`**: Automatically maintained leaderboard ranking all models by OOD accuracy and calibrated triage sensitivity.

