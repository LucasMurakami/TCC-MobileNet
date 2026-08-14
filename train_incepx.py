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

# PARAMS
IMG_SIZE = (299, 299)
CHANNELS = 3
NUM_CLASSES = 7
INITIAL_LR = 1e-4
FINETUNE_LR = 5e-6

# DATA AUGMENTATION
def build_datagen() -> ImageDataGenerator:
    return ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="reflect"
    )

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

# DATASET PREPARATION WITH OVERSAMPLING
def prepare_balanced_df(cache_root: Path):
    raw_dir = cache_root / "ham10000_raw"
    if not (raw_dir / "HAM10000_metadata.csv").exists():
        print("[dataset] Downloading HAM10000...")
        downloaded_path = Path(kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000"))
        if raw_dir.exists(): shutil.rmtree(raw_dir)
        shutil.copytree(downloaded_path, raw_dir)

    df = pd.read_csv(raw_dir / "HAM10000_metadata.csv")
    
    # Map paths
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x 
                         for x in glob.glob(os.path.join(raw_dir, '*', '*.jpg'))}
    df['path'] = df['image_id'].map(imageid_path_dict)

    # SPLIT FIRST
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['dx'])
    
    # OVERSAMPLE (ONLY TRAINING)
    print("[dataset] Oversampling training set to balance classes...")
    max_size = train_df['dx'].value_counts().max()
    lst = []
    for class_index, group in train_df.groupby('dx'):
        lst.append(resample(group, replace=True, n_samples=max_size, random_state=42))
    train_df_balanced = pd.concat(lst)
    
    print(f"[dataset] Balanced Training Size: {len(train_df_balanced)} (was {len(train_df)})")
    print(f"[dataset] Original Validation Size: {len(val_df)}")
    
    return train_df_balanced, val_df

# EVALUATION
def evaluate_and_plot(model, val_gen, output_dir: Path):
    val_gen.reset()
    y_pred = model.predict(val_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_gen.classes
    class_labels = list(val_gen.class_indices.keys())
    
    report = classification_report(y_true, y_pred_classes, target_names=class_labels)
    print(report)
    with open(output_dir / "balanced_report.txt", "w") as f: f.write(report)

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_labels, yticklabels=class_labels)
    plt.savefig(output_dir / "balanced_confusion_matrix.png")

def plot_results(history, stage_name, output_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"], label="Train"); ax1.plot(history.history["val_accuracy"], label="Val")
    ax1.set_title(f"Acc - {stage_name}"); ax1.legend()
    ax2.plot(history.history["loss"], label="Train"); ax2.plot(history.history["val_loss"], label="Val")
    ax2.set_title(f"Loss - {stage_name}"); ax2.legend()
    plt.savefig(output_dir / f"curves_{stage_name}.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16) 
    args = parser.parse_args()

    output_path = Path("./outputs_balanced"); output_path.mkdir(exist_ok=True)
    train_df, val_df = prepare_balanced_df(Path("./data_cache"))
    
    datagen = build_datagen()
    # Rescale
    val_datagen = ImageDataGenerator(rescale=1.0/255)

    train_gen = datagen.flow_from_dataframe(train_df, x_col='path', y_col='dx', target_size=IMG_SIZE,
                                            batch_size=args.batch_size, class_mode='categorical')
    val_gen = val_datagen.flow_from_dataframe(val_df, x_col='path', y_col='dx', target_size=IMG_SIZE,
                                              batch_size=args.batch_size, class_mode='categorical', shuffle=False)

    # STAGE 1
    model = build_incepx_ensemble()
    model.compile(optimizer=Adam(INITIAL_LR), loss="categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_gen, validation_data=val_gen, epochs=15, 
                   callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)])

    # STAGE 2
    print("\n--- Deep Fine-Tuning ---")
    for layer in model.layers: layer.trainable = True
    model.compile(optimizer=Adam(FINETUNE_LR), loss="categorical_crossentropy", metrics=["accuracy"])
    
    checkpoint = ModelCheckpoint(output_path / "incepx_balanced_best.keras", save_best_only=True, monitor="val_accuracy")
    h2 = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, 
                   callbacks=[checkpoint, EarlyStopping(patience=10)])
    
    evaluate_and_plot(tf.keras.models.load_model(output_path / "incepx_balanced_best.keras"), val_gen, output_path)

if __name__ == "__main__":
    main()