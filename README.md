# Skin Lesion Classification: MobileNet Benchmark & IncepX Ensemble

Deep learning framework for skin lesion classification comparing lightweight architectures (**MobileNet V1, V2, V3Small, V3Large, V4Conv, V4ConvL, and V5**) with official pre-trained weights against an **InceptionV3 + Xception (IncepX)** ensemble on dermoscopic and clinical datasets.

---

## 📌 Project Overview

- **Primary Dataset (Training)**: [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) (10,015 dermoscopy images across 7 diagnostic categories).
- **External Validation**: [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1) (clinical smartphone images collected in Brazil to evaluate out-of-domain generalization).
- **Diagnostic Classes**:
  1. `akiec` - Actinic Keratoses / Intraepithelial Carcinoma
  2. `bcc` - Basal Cell Carcinoma
  3. `bkl` - Benign Keratosis-like Lesions
  4. `df` - Dermatofibroma
  5. `mel` - Melanoma
  6. `nv` - Melanocytic Nevi
  7. `vasc` - Vascular Lesions

---

## 🏗️ Architecture & Modules

```text
├── dataset.py            # Centralized dataset pipeline, caching, loaders, Focal Loss & callbacks
├── train_mobilenets.py   # Unified MobileNet V1, V2, V3Small, V3Large, V4Conv, V4ConvL, V5 suite
├── train_timm_models.py  # PyTorch & timm trainer for Hugging Face pretrained V4 and V5 models
├── train_incepx.py       # InceptionV3 + Xception dual-backbone feature fusion ensemble
├── visualize.py          # Dual confusion matrix, per-class bar charts, Grad-CAM heatmaps & curves
├── requirements.txt      # Python dependencies
├── data_cache/           # Local cache for raw datasets (HAM10000, PAD-UFES-20)
└── tests/                # Smoke tests and loss validation scripts
    └── test_focal.py     # Focal Loss & model serialization tests
```

---

## 📦 Supported Models & Pretrained Weights

| CLI Model Flag | Architecture | Input Resolution | Pretrained Source / Checkpoint |
|---|---|---|---|
| `--model v1` | MobileNet V1 | 224 × 224 | **ImageNet-1k** (`timm/mobilenetv1_100`) |
| `--model v2` | MobileNet V2 | 224 × 224 | **ImageNet-1k** (`timm/mobilenetv2_100`) |
| `--model v3small` | MobileNet V3 Small | 224 × 224 | **ImageNet-1k** (`timm/mobilenetv3_small_100`) |
| `--model v3large` | MobileNet V3 Large | 224 × 224 | **ImageNet-1k** (`timm/mobilenetv3_large_100`) |
| `--model v4conv` | MobileNet V4 Conv-Medium | 256 × 256 | **ImageNet-1k** (`timm/mobilenetv4_conv_medium.e500_r256_in1k`) |
| `--model v4convl` | MobileNet V4 Conv-Large | 384 × 384 | **ImageNet-1k** (`timm/mobilenetv4_conv_large.e500_r384_in1k`) |
| `--model v5` | MobileNet V5 (300M) | 256 × 256 | **Google Gemma3n Vision Backbone** (`timm/mobilenetv5_300m.gemma3n`) |

---

### Key Technical Strategies
1. **Stratified Splitting & Training-Only Oversampling**: Prevents data leakage by splitting the raw dataset first, then oversampling only the training fold.
2. **Focal Loss ($\gamma=2.0$) & Class Weights**: Mitigates heavy class imbalance (e.g., thousands of `nv` vs. tens of `df`/`vasc`).
3. **Two-Stage Transfer Learning**:
   - **Stage 1 (Head Warmup)**: Frozen backbone, trains classification head ($lr \approx 10^{-3}$).
   - **Stage 2 (Fine-Tuning)**: Unfreezes convolutional layers with lower learning rate ($lr \approx 10^{-5}$) while keeping BatchNorm layers frozen for stable batch statistics.

---

## 🚀 Environment Setup

### 1. Create Virtual Environment & Install Dependencies

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Verify GPU Acceleration

```bash
.venv/bin/python -c "import tensorflow as tf; print('GPUs Available:', tf.config.list_physical_devices('GPU'))"
```

---

## 🏋️ Training & Usage

### 1. Train MobileNet V1 – V5 (`train_mobilenets.py`)

Run any MobileNet variant through the standardized dual-stage training & evaluation pipeline:

```bash
# MobileNet V1 (ImageNet pretrained)
.venv/bin/python train_mobilenets.py --model v1 --epochs 50 --batch-size 32

# MobileNet V2 (ImageNet pretrained)
.venv/bin/python train_mobilenets.py --model v2 --epochs 50 --batch-size 32

# MobileNet V3 (Small or Large)
.venv/bin/python train_mobilenets.py --model v3small --epochs 50
.venv/bin/python train_mobilenets.py --model v3large --epochs 50

# MobileNet V4 (Hugging Face / timm ImageNet pretrained)
.venv/bin/python train_mobilenets.py --model v4conv --epochs 50
.venv/bin/python train_mobilenets.py --model v4convl --epochs 50

# MobileNet V5 (Hugging Face / timm Google Gemma3n pretrained)
.venv/bin/python train_mobilenets.py --model v5 --epochs 50

# Train ALL MobileNet variants (V1 through V5) and generate multi-model comparison charts
.venv/bin/python train_mobilenets.py --model all --epochs 50
```

### 2. Train IncepX Ensemble (`train_incepx.py`)

```bash
.venv/bin/python train_incepx.py --epochs 50 --batch-size 32
```

---

## 🧪 External Validation on PAD-UFES-20

To evaluate clinical generalization by training on HAM10000 and validating on PAD-UFES-20:

```bash
# MobileNet V1 on PAD-UFES-20
.venv/bin/python train_mobilenets.py --model v1 --val-dataset pad-ufes-20 --epochs 50 --batch-size 32

# MobileNet V2 on PAD-UFES-20
.venv/bin/python train_mobilenets.py --model v2 --val-dataset pad-ufes-20 --epochs 50 --batch-size 32

# MobileNet V3 (Small or Large) on PAD-UFES-20
.venv/bin/python train_mobilenets.py --model v3small --val-dataset pad-ufes-20 --epochs 50
.venv/bin/python train_mobilenets.py --model v3large --val-dataset pad-ufes-20 --epochs 50

# MobileNet V4 (Conv Medium or Conv Large) on PAD-UFES-20
.venv/bin/python train_mobilenets.py --model v4conv --val-dataset pad-ufes-20 --epochs 50
.venv/bin/python train_mobilenets.py --model v4convl --val-dataset pad-ufes-20 --epochs 50

# MobileNet V5 (300M Gemma3n) on PAD-UFES-20
.venv/bin/python train_mobilenets.py --model v5 --val-dataset pad-ufes-20 --epochs 50

# Run All Models on PAD-UFES-20 and generate comparison charts
.venv/bin/python train_mobilenets.py --model all --val-dataset pad-ufes-20 --epochs 50
```

> [!IMPORTANT]
> **PAD-UFES-20 Setup**:
> Download the dataset archive (`pad-ufes-20.zip`) from **Mendeley Data**:
> - **Mendeley Data Link**: [https://data.mendeley.com/datasets/zr7vgbcyr2/1](https://data.mendeley.com/datasets/zr7vgbcyr2/1)
> 
> Extract `metadata.csv` and image subfolders (`imgs_part_1`, `imgs_part_2`, `imgs_part_3`) into:
> ```bash
> data_cache/pad_ufes_20_raw/
> ```

---

## 📊 Outputs & Visualizations

Training automatically generates rich visualization artifacts in the designated output folder (e.g., `./mobilenet_outputs/<model_name>/`):

- **`confusion_matrix.png`**: Dual side-by-side matrices (raw sample counts + normalized sensitivity percentages).
- **`per_class_metrics.png`**: Per-class Precision, Recall, and F1-score bar chart with exact value labels.
- **`training_curves.png`**: Multi-stage loss and accuracy curves with clear epoch markers.
- **`gradcam_heatmaps.png`**: Grad-CAM CNN attention heatmaps overlaid on sample lesion images.
- **`classification_report.json`**: Precision, recall, and F1-scores per diagnostic class.
- **`results.json`**: Key evaluation metrics (accuracy, weighted F1, macro F1, parameter counts).
- **`benchmark_comparison.png`**: Multi-model comparison bar chart generated when running `--model all`.
