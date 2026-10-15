"""CNN Model Architecture & Training Module (Member C).
Builds, compiles, and trains the GTSRB 43-class CNN model.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers, models

from src.config import (
    KERAS_MODEL_PATH,
    RESULTS_DIR,
    NUM_CLASSES,
    CLASSIFICATION_IMG_SIZE,
)
from src.data_pipeline.data_loader import load_split, class_weights


def build_traffic_sign_cnn(input_shape=(CLASSIFICATION_IMG_SIZE, CLASSIFICATION_IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """Build the official Member C Sequential CNN architecture."""
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Dense Classifier
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def train_cnn(epochs: int = 15, batch_size: int = 32, use_balanced: bool = True):
    """Train the CNN classifier on GTSRB data."""
    plots_dir = RESULTS_DIR / "classification" / "plots"
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(KERAS_MODEL_PATH.parent, exist_ok=True)

    split = "train_balanced" if use_balanced else "train"
    print(f"\n--- Training CNN Classifier on '{split}' ---")

    try:
        X_train, y_train = load_split(split, normalize=True)
        X_val, y_val = load_split("val", normalize=True)
    except FileNotFoundError:
        print("[TRAIN] Splits not found. Running split_data & augment first...")
        from src.data_pipeline.split_data import split_and_save_data
        from src.data_pipeline.augment import balance_and_augment
        split_and_save_data()
        balance_and_augment()
        X_train, y_train = load_split(split, normalize=True)
        X_val, y_val = load_split("val", normalize=True)

    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")

    model = build_traffic_sign_cnn()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        keras.callbacks.ModelCheckpoint(str(KERAS_MODEL_PATH), save_best_only=True, monitor="val_accuracy"),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    print(f"\n[SUCCESS] Model successfully saved to: {KERAS_MODEL_PATH}")

    # Plot Accuracy & Loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = plots_dir / "training_curves.png"
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"[TRAIN] Curves saved to: {curve_path}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train_cnn(epochs=args.epochs, batch_size=args.batch_size)
