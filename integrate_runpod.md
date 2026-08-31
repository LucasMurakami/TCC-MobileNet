# RunPod integration for this project

This project is already set up for a GPU training workflow. The easiest way to run it on RunPod is to start a GPU pod, install dependencies, mount or upload the dataset, and launch the training script directly.

## 1) Recommended RunPod setup

Use a RunPod GPU template with:

- CUDA-capable GPU
- Ubuntu base image
- at least 20–50 GB disk space
- Python 3.10+ recommended

A good default is:

- Template: PyTorch / CUDA image
- GPU: NVIDIA RTX 4090 / A6000 / 3060/4060 class
- Storage: 50 GB+ for dataset + model outputs

## 2) Prepare the project on the pod

In the RunPod terminal:

```bash
cd /workspace
git clone <your-repo-url>
cd TCC
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the repo is already mounted in the pod under another path, just use that path instead of `git clone`.

## 3) Upload or mount the dataset

This project expects the dataset under:

```bash
./dataset_treino
./data_cache
```

So on RunPod, make sure the same structure exists before training:

```bash
ls -la
ls -la dataset_treino
ls -la data_cache
```

If you are using a mounted volume, copy the dataset there first:

```bash
mkdir -p /workspace/TCC/data_cache
cp -r /mnt/data/dataset_treino /workspace/TCC/
cp -r /mnt/data/data_cache /workspace/TCC/
```

## 4) Verify the environment

Check Python, PyTorch and CUDA before training:

```bash
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

You should see at least one GPU listed.

## 5) Run training on RunPod

The training entry point is:

```bash
python train_timm_models.py --model v4 --epochs 50 --batch-size 32
```

With the main options:

```bash
python train_timm_models.py \
  --model v4 \
  --epochs 50 \
  --batch-size 32 \
  --lr-stage1 1e-3 \
  --lr-stage2 1e-4 \
  --patience 10 \
  --val-dataset pad-ufes-20 \
  --output-dir mobilenet_outputs
```

To run all generations (V1 through V5):

```bash
python train_timm_models.py \
  --model all \
  --epochs 50 \
  --batch-size 32 \
  --val-dataset pad-ufes-20 \
  --output-dir mobilenet_outputs
```

## 6) Output files

The script writes results to the output directory, for example:

```bash
mobilenet_outputs/v4/
  best_model.pth
  stage1_head_best.pth
  results.json
  classification_report.json
  confusion_matrix.png
  per_class_metrics.png
  training_curves.png
  roc_curves.png
  gradcam_heatmaps.png
```

These are created automatically by the PyTorch / timm trainer.

## 7) Long-running training on RunPod

For longer jobs, start the training in the background and log the output:

```bash
nohup bash -lc 'source .venv/bin/activate && cd /workspace/TCC && python train_timm_models.py --model all --epochs 50 --batch-size 32 --val-dataset pad-ufes-20 --output-dir mobilenet_outputs' > runpod_train.log 2>&1 &
```

Then monitor:

```bash
tail -f runpod_train.log
```

## 8) Download results after training

Once training completes, copy the output folder back to your local machine:

```bash
zip -r mobilenet_outputs.zip mobilenet_outputs
```

Then use RunPod file browser or `scp` to download the zip file.

## 9) Practical RunPod workflow

Recommended sequence:

1. Start a GPU pod
2. Clone or mount the repo
3. Install dependencies
4. Upload or mount datasets (`data_cache` with HAM10000 and PAD-UFES-20)
5. Run the model training command
6. Save and download `results.json`, metrics, and PyTorch model checkpoints (`best_model.pth`)

## 10) Fastest command for this repo

If you just want the quickest version:

```bash
cd /workspace/TCC
source .venv/bin/activate
python train_timm_models.py --model v3small --epochs 2 --batch-size 32
```

This uses the default arguments already configured in the script and runs a fast sanity check on RunPod.

## 11) Notes for this project

- The suite uses PyTorch / timm with automatic Mixed Precision (BFloat16 / FP16) and expects a CUDA-enabled environment.
- The dataset is automatically downloaded from Kaggle (HAM10000) or loaded from `data_cache/pad_ufes_20_raw` (PAD-UFES-20).
- If your project is stored on a mounted volume, keep the paths consistent to avoid training from a different folder than the one with the dataset.

## 12) Example full RunPod terminal flow

```bash
cd /workspace
git clone <repo-url>
cd TCC
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python train_timm_models.py \
  --model v4 \
  --epochs 50 \
  --batch-size 32 \
  --val-dataset pad-ufes-20 \
  --output-dir mobilenet_outputs
```

This is the standard RunPod integration flow for this repository.


