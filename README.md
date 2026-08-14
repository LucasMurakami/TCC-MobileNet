# IncepX Ensemble (WSL + RTX)

This project trains an InceptionV3 + Xception ensemble on HAM10000.

## 1) Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Run training

```bash
python train_incepx.py --epochs 50 --batch-size 32
```

Useful options:

```bash
python train_incepx.py \
  --epochs 10 \
  --batch-size 16 \
  --cache-dir ./data_cache \
  --prepared-dir ./dataset_treino \
  --plot-path ./outputs/training_curves.png
```

## 3) Dataset caching behavior

- Raw Kaggle dataset is cached in `./data_cache/ham10000_raw`.
- If that folder already has the expected HAM10000 files, download is skipped.
- Prepared class folders are created in `./dataset_treino`.
- If `./dataset_treino/.ready` matches metadata image count, preparation is skipped.

## 4) GPU check

The script prints detected GPUs at startup:

```text
GPU Available: [...]
```

If empty, check NVIDIA driver and CUDA support in WSL first.
