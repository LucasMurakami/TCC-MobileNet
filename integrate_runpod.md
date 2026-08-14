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

Check Python and CUDA before training:

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

You should see at least one GPU listed.

## 5) Run training on RunPod

The training entry point is:

```bash
python train_mobilenetv5.py
```

With the main options:

```bash
python train_mobilenetv5.py \
  --config b0 \
  --epochs-stage1 20 \
  --epochs-stage2 50 \
  --batch-size 32 \
  --img-size 224 \
  --data-dir dataset_treino \
  --output-dir mobilenetv5_output \
  --gpu 0
```

A larger model variant is also available:

```bash
python train_mobilenetv5.py \
  --config b1 \
  --epochs-stage1 20 \
  --epochs-stage2 50 \
  --batch-size 16 \
  --img-size 224 \
  --data-dir dataset_treino \
  --output-dir mobilenetv5_output_b1 \
  --gpu 0
```

## 6) Output files

The script writes results to the output directory, for example:

```bash
mobilenetv5_output/
  stage1_best.weights.h5
  stage2_best.weights.h5
  model.keras
  results.json
  stage1_evaluation.png
  stage2_evaluation.png
```

These are created automatically by the script.

## 7) Long-running training on RunPod

For longer jobs, start the training in the background and log the output:

```bash
nohup bash -lc 'source .venv/bin/activate && cd /workspace/TCC && python train_mobilenetv5.py --config b0 --epochs-stage1 20 --epochs-stage2 50 --batch-size 32 --img-size 224 --data-dir dataset_treino --output-dir mobilenetv5_output --gpu 0' > runpod_train.log 2>&1 &
```

Then monitor:

```bash
tail -f runpod_train.log
```

## 8) Download results after training

Once training completes, copy the output folder back to your local machine:

```bash
zip -r mobilenetv5_output.zip mobilenetv5_output
```

Then use RunPod file browser or `scp` to download the zip file.

## 9) Practical RunPod workflow

Recommended sequence:

1. Start a GPU pod
2. Clone or mount the repo
3. Install dependencies
4. Upload or mount the HAM10000 dataset
5. Run the model training command
6. Save and download `results.json`, metrics, and model weights

## 10) Fastest command for this repo

If you just want the quickest version:

```bash
cd /workspace/TCC
source .venv/bin/activate
python train_mobilenetv5.py --config b0 --batch-size 32 --gpu 0
```

This uses the default arguments already configured in the script and should work well as a baseline on RunPod.

## 11) Notes for this project

- The script uses TensorFlow/Keras and expects a CUDA-enabled environment.
- The dataset is not downloaded automatically inside the script for the MobileNetV5 trainer; it expects your dataset to already be prepared in `dataset_treino`.
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
ls dataset_treino
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python train_mobilenetv5.py \
  --config b0 \
  --epochs-stage1 20 \
  --epochs-stage2 50 \
  --batch-size 32 \
  --img-size 224 \
  --data-dir dataset_treino \
  --output-dir mobilenetv5_output \
  --gpu 0
```

This is the standard RunPod integration flow for this repository.
