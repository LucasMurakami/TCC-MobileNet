import argparse
import os
import shutil
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, Xception
from tensorflow.keras.layers import Concatenate, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight, resample
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# --- GPU STABILITY SETTINGS ---
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

from dataset import (
    NUM_CLASSES, CLASS_NAMES, prepare_dataset, get_data_generators, compute_class_weights
)

# PARAMS
IMG_SIZE = (299, 299)
CHANNELS = 3
INITIAL_LR = 1e-4
FINETUNE_LR = 5e-6


# MODEL ARCHITECTURE
def build_incepx_ensemble() -> Model:
    input_layer = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS))
    inc_base = InceptionV3(weights="imagenet", include_top=False, input_tensor=input_layer)
    x1 = GlobalAveragePooling2D()(inc_base.output)
    xcp_base = Xception(weights="imagenet", include_top=False, input_tensor=input_layer)
    x2 = GlobalAveragePooling2D()(xcp_base.output)
    merged = Concatenate()([x1, x2])
    x = Dense(512, activation="relu")(merged)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    output_layer = Dense(NUM_CLASSES, activation="softmax", dtype='float32')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    inc_base.trainable = False
    xcp_base.trainable = False
    return model


from visualize import (
    plot_training_curves, plot_confusion_matrices,
    plot_per_class_metrics, generate_gradcam_gallery
)

# EVALUATION
def evaluate_and_plot(model, val_gen, output_dir: Path, h1=None, h2=None, val_df=None):
    val_gen.reset()
    y_pred = model.predict(val_gen, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_gen.classes
    class_labels = list(val_gen.class_indices.keys())
    
    report = classification_report(y_true, y_pred_classes, target_names=class_labels, output_dict=True, zero_division=0)
    print("\nClassification Report (IncepX Ensemble):")
    print(classification_report(y_true, y_pred_classes, target_names=class_labels, zero_division=0))
    
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred_classes, target_names=class_labels, zero_division=0))
    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # 1. Dual Confusion Matrix
    plot_confusion_matrices(
        y_true, y_pred_classes, class_labels,
        output_path=output_dir / "confusion_matrix.png",
        model_name="IncepX Ensemble"
    )

    # 2. Per-Class Performance Bar Chart
    plot_per_class_metrics(
        report, class_labels,
        output_path=output_dir / "per_class_metrics.png",
        model_name="IncepX Ensemble"
    )

    # 3. Training Curves
    if h1 is not None and h2 is not None:
        plot_training_curves(
            [h1.history, h2.history],
            ["Stage 1 (Head)", "Stage 2 (Fine-Tune)"],
            output_path=output_dir / "training_curves.png",
            model_name="IncepX Ensemble"
        )

    # 4. Grad-CAM CNN Interpretability Heatmaps
    if val_df is not None:
        print('\nGenerating Grad-CAM CNN Attention Heatmaps...')
        try:
            val_sample_paths = val_df['path'].tolist()
            val_sample_labels = [class_labels.index(c) for c in val_df['dx']]
            generate_gradcam_gallery(
                model, val_sample_paths, val_sample_labels, class_labels,
                img_size=IMG_SIZE[0],
                output_path=output_dir / "gradcam_heatmaps.png",
                num_samples=6,
                model_name="IncepX Ensemble"
            )
            print(f'  Saved Grad-CAM heatmaps to {output_dir / "gradcam_heatmaps.png"}')
        except Exception as e:
            print(f'  Warning: Grad-CAM generation encountered {e}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16) 
    args = parser.parse_args()

    output_path = Path("./outputs_balanced")
    output_path.mkdir(parents=True, exist_ok=True)
    train_df, val_df = prepare_dataset(Path("./data_cache"), Path("./dataset_treino"))
    train_gen, val_gen = get_data_generators(train_df, val_df, img_size=IMG_SIZE[0], batch_size=args.batch_size)

    # STAGE 1
    model = build_incepx_ensemble()
    model.compile(optimizer=Adam(INITIAL_LR), loss="categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_gen, validation_data=val_gen, epochs=15, 
                   callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)],
                   verbose=1)

    # STAGE 2
    print("\n--- Deep Fine-Tuning ---")
    for layer in model.layers: layer.trainable = True
    model.compile(optimizer=Adam(FINETUNE_LR), loss="categorical_crossentropy", metrics=["accuracy"])
    
    checkpoint = ModelCheckpoint(output_path / "incepx_balanced_best.keras", save_best_only=True, monitor="val_accuracy")
    h2 = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, 
                   callbacks=[checkpoint, EarlyStopping(patience=10)],
                   verbose=1)
    
    best_model = tf.keras.models.load_model(output_path / "incepx_balanced_best.keras")
    evaluate_and_plot(best_model, val_gen, output_path, h1=h1, h2=h2, val_df=val_df)

if __name__ == "__main__":
    main()