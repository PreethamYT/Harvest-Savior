"""
train_model.py — CNN Training Script for Harvest Savior
=========================================================

PURPOSE:
  This script trains a Convolutional Neural Network (CNN) on the
  PlantVillage dataset and saves the trained model to:
    harvest-savior-ai/model/crop_disease_cnn.h5

  Once saved, the Flask microservice (app.py) automatically switches
  from DEMO MODE to REAL MODE and uses this model for predictions.

USAGE:
  C:\\hs_venv\\Scripts\\python.exe train_model.py

PREREQUISITES:
  1. Run setup_training_env.ps1 to install TensorFlow into C:\\hs_venv
  2. Run download_dataset.py to download the PlantVillage dataset
  3. Verify DATASET_ROOT below points to your dataset folder

===========================================================================
  FULL EXPLANATION FOR VIVA (read this before your mid-term evaluation)
===========================================================================

WHAT IS A CNN?
  A Convolutional Neural Network is a type of deep learning model
  specifically designed for processing grid-like data such as images.
  Unlike a plain neural network that treats every pixel independently,
  a CNN learns SPATIAL PATTERNS — edges, textures, colour blotches —
  by sliding small filters (kernels) across the image.

HOW TRAINING WORKS (step by step):
  1. FORWARD PASS: An image is passed through all layers. Each Conv2D
     layer applies learned filters to detect features. MaxPooling
     reduces the spatial size. The final Dense layer outputs a
     probability for each of the 38 disease classes (softmax).

  2. LOSS CALCULATION: The predicted probability is compared to the
     true label using Categorical Cross-Entropy loss.
     Loss = -log(probability of correct class)
     A perfect prediction gives loss ≈ 0.

  3. BACKPROPAGATION: The gradient of the loss with respect to every
     weight is calculated using the chain rule of calculus. This tells
     us: "if we tweak this weight slightly, does loss go up or down?"

  4. WEIGHT UPDATE: The Adam optimiser adjusts every weight in the
     direction that reduces the loss. This is repeated for every batch
     of images in every epoch until the model converges.

DATA AUGMENTATION:
  We apply random flips, rotations, and zoom to the training images.
  Why? To make the model robust to real-world variation — a farmer's
  photo of a leaf might be at a different angle, closer, or slightly
  blurry compared to the clean dataset photos. Augmentation simulates
  this variation without needing extra real images.

TRANSFER LEARNING (used here):
  Instead of training from scratch (which needs millions of images),
  we use MobileNetV2 — a model pre-trained on ImageNet (14 million
  photos). Its early layers already know how to detect edges, shapes,
  and textures. We only train the final classification layers for our
  38 disease classes. This converges in minutes instead of days.
"""

import os
import sys
import json
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')           # Non-interactive backend — works without a display
import matplotlib.pyplot as plt

# ── Import TensorFlow ─────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    print(f"✅ TensorFlow {tf.__version__} loaded successfully.")
except ImportError as e:
    print("=" * 60)
    print("  ERROR: TensorFlow is not installed in this Python environment.")
    print("=" * 60)
    print()
    print("  Fix: run the setup script first:")
    print("  powershell -ExecutionPolicy Bypass -File setup_training_env.ps1")
    print()
    print("  Then rerun using the venv Python:")
    print("  C:\\hs_venv\\Scripts\\python.exe train_model.py")
    sys.exit(1)


# ============================================================
#  CONFIGURATION — edit these if your paths differ
# ============================================================

# Path to the folder that contains one sub-folder per disease class.
# After running download_dataset.py this should be correct automatically.
DATASET_ROOT = os.path.join(os.path.dirname(__file__), 'dataset', 'PlantVillage')

# Where the trained model will be saved (Flask reads from here)
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'model', 'crop_disease_cnn.h5')

# Where training plots and class-name list will be saved
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'model')

# ── Image dimensions (must match predictor.py) ────────────────────────────────
IMG_HEIGHT  = 224
IMG_WIDTH   = 224
IMG_SIZE    = (IMG_HEIGHT, IMG_WIDTH)

# ── Training hyper-parameters ────────────────────────────────────────────────
BATCH_SIZE  = 32     # Number of images processed per weight update
EPOCHS      = 30     # Maximum training epochs (early stopping may cut this short)
FINE_TUNE_EPOCHS = 10  # Additional epochs after unfreezing MobileNetV2 top layers
VAL_SPLIT   = 0.2    # 20% of data reserved for validation
SEED        = 42     # Random seed for reproducibility


# ============================================================
#  STEP 1 — Verify dataset exists
# ============================================================
def verify_dataset():
    if not os.path.isdir(DATASET_ROOT):
        print(f"ERROR: Dataset folder not found at:\n  {DATASET_ROOT}")
        print()
        print("Run download_dataset.py first, then re-check the DATASET_ROOT")
        print("path at the top of this file.")
        sys.exit(1)

    classes = sorted([
        d for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d))
    ])

    if len(classes) == 0:
        print(f"ERROR: No class sub-folders found inside {DATASET_ROOT}")
        sys.exit(1)

    print(f"✅ Dataset found: {len(classes)} classes at {DATASET_ROOT}")
    return classes


# ============================================================
#  STEP 2 — Build tf.data input pipeline with augmentation
# ============================================================
def build_datasets():
    """
    keras.utils.image_dataset_from_directory() walks the dataset folder,
    assigns integer labels based on the alphabetically sorted sub-folder names,
    and splits 80% into training and 20% into validation.

    Data augmentation is applied ONLY to the training set — never validation,
    because we want the validation loss to reflect true performance.
    """
    print("\n── Loading dataset ──────────────────────────────────────────")

    train_ds = keras.utils.image_dataset_from_directory(
        DATASET_ROOT,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',   # one-hot encoded labels needed for softmax
    )

    val_ds = keras.utils.image_dataset_from_directory(
        DATASET_ROOT,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"   Classes  : {num_classes}")
    print(f"   Train    : {len(train_ds)} batches × {BATCH_SIZE} images")
    print(f"   Val      : {len(val_ds)} batches × {BATCH_SIZE} images")

    # ── Save class names alongside the model ─────────────────────────────────
    # predictor.py reads CLASS_NAMES in the same order used during training.
    # If they deviate, all predictions will be wrong.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    class_names_path = os.path.join(OUTPUT_DIR, 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"   Class names saved → {class_names_path}")

    # ── Augmentation layers ───────────────────────────────────────────────────
    # Applied inline as the first layers of the model so they run on the GPU.
    # RandomFlip: mirrors the image horizontally (simulates reversed leaf angle)
    # RandomRotation: rotates up to ±10% (simulates tilted photo)
    # RandomZoom: zooms in/out up to 10% (simulates different camera distances)
    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
    ], name="augmentation")

    # ── Prefetch: prepares next batch while GPU trains on current batch ───────
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names, num_classes


# ============================================================
#  STEP 3 — Build the model using Transfer Learning
# ============================================================
def build_model(num_classes: int) -> keras.Model:
    """
    Uses MobileNetV2 as a feature extractor (base model).

    WHY MOBILENETV2?
      MobileNetV2 was designed for mobile devices — it is accurate AND
      fast with very few parameters (~3.4M). This is ideal for our
      target users (farmers on low-end Android phones). It was pre-
      trained on ImageNet (14M images, 1000 classes) so it already
      understands general visual features like edges, shapes, textures.

    TRANSFER LEARNING STRATEGY (two phases):
      Phase A (frozen base): Train only the new head (classification layers).
                             This is fast and prevents "catastrophic forgetting"
                             — destroying the ImageNet knowledge.
      Phase B (fine-tuning): Unfreeze the top layers of MobileNetV2 and
                             train at a very low learning rate to specialise
                             the features for plant disease recognition.
    """
    print("\n── Building model ───────────────────────────────────────────")

    # ── Preprocessing: normalise pixels from [0,255] → [-1, 1] ──────────────
    # MobileNetV2 was trained with pixels in [-1, 1] range, so we must match.
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # ── Load MobileNetV2 without its top classification layers ───────────────
    # include_top=False removes the final Dense layer trained for ImageNet.
    # We will attach our own head for 38 disease classes.
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'        # download pre-trained weights automatically
    )
    base_model.trainable = False  # Phase A: freeze all base layers
    print(f"   Base model : MobileNetV2 ({len(base_model.layers)} layers, frozen)")

    # ── Build the full model ─────────────────────────────────────────────────
    inputs  = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = preprocess_input(inputs)           # normalise to [-1, 1]
    x = base_model(x, training=False)      # extract features (frozen)
    x = keras.layers.GlobalAveragePooling2D()(x)  # flatten feature map to vector
    x = keras.layers.Dropout(0.2)(x)       # regularisation (prevent overfitting)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name="HarvestSavior_MobileNetV2")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"   Trainable params : {model.count_params():,}")
    return model, base_model


# ============================================================
#  STEP 4 — Train Phase A (frozen base)
# ============================================================
def train_phase_a(model, train_ds, val_ds):
    print("\n── Phase A Training (classification head only) ──────────────")

    callbacks = [
        # Stop early if validation accuracy does not improve for 5 epochs
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Save the best model checkpoint during training
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Reduce learning rate when progress stalls
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        ),
    ]

    history_a = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
    )
    return history_a


# ============================================================
#  STEP 5 — Fine-tune Phase B (unfreeze top layers)
# ============================================================
def train_phase_b(model, base_model, train_ds, val_ds):
    """
    Fine-tuning: unfreeze the top 30 layers of MobileNetV2 and train
    at a 10× smaller learning rate. This specialises the base features
    for plant disease patterns while preserving low-level feature detectors.
    """
    print("\n── Phase B Fine-tuning (unfreeze top 30 base layers) ────────")

    # Unfreeze only the top portion of the base model
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    frozen     = sum(1 for l in base_model.layers if not l.trainable)
    unfrozen   = sum(1 for l in base_model.layers if l.trainable)
    print(f"   Frozen: {frozen} layers  |  Unfrozen: {unfrozen} layers")

    # Recompile with a lower learning rate to avoid destroying learned features
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
    ]

    history_b = model.fit(
        train_ds,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
    )
    return history_b


# ============================================================
#  STEP 6 — Plot and save training curves
# ============================================================
def save_training_plots(history_a, history_b):
    """
    Saves accuracy and loss graphs to model/training_plots.png
    These plots are useful for your project report and viva presentation.

    WHAT TO LOOK FOR:
      - Training accuracy ↑ and validation accuracy ↑ together → good learning
      - Training accuracy ↑ but validation accuracy ↓ → overfitting
      - Both accuracies flat → model not learning (adjust learning rate)
    """
    # Combine Phase A and Phase B history values
    acc     = history_a.history['accuracy']     + history_b.history['accuracy']
    val_acc = history_a.history['val_accuracy'] + history_b.history['val_accuracy']
    loss    = history_a.history['loss']         + history_b.history['loss']
    val_loss= history_a.history['val_loss']     + history_b.history['val_loss']

    epochs_range = range(len(acc))
    fine_tune_start = len(history_a.history['accuracy'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Accuracy plot ─────────────────────────────────────────────────────────
    axes[0].plot(epochs_range, acc,     label='Training Accuracy',   color='#2e7d32')
    axes[0].plot(epochs_range, val_acc, label='Validation Accuracy', color='#81c784', linestyle='--')
    axes[0].axvline(fine_tune_start, color='grey', linestyle=':', label='Fine-tune start')
    axes[0].set_title('Model Accuracy', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Loss plot ─────────────────────────────────────────────────────────────
    axes[1].plot(epochs_range, loss,     label='Training Loss',   color='#c62828')
    axes[1].plot(epochs_range, val_loss, label='Validation Loss', color='#ef9a9a', linestyle='--')
    axes[1].axvline(fine_tune_start, color='grey', linestyle=':', label='Fine-tune start')
    axes[1].set_title('Model Loss', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Harvest Savior CNN — Training History', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, 'training_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Training plots saved → {plot_path}")


# ============================================================
#  STEP 7 — Final evaluation on the validation set
# ============================================================
def evaluate_model(model, val_ds, class_names):
    print("\n── Final Evaluation ─────────────────────────────────────────")
    loss, accuracy = model.evaluate(val_ds, verbose=1)
    print(f"\n   Validation Loss     : {loss:.4f}")
    print(f"   Validation Accuracy : {accuracy * 100:.2f}%")

    # Save a summary report
    report = {
        "timestamp":           datetime.datetime.now().isoformat(),
        "val_loss":            round(loss, 4),
        "val_accuracy":        round(accuracy, 4),
        "val_accuracy_pct":    round(accuracy * 100, 2),
        "num_classes":         len(class_names),
        "class_names":         class_names,
        "img_size":            [IMG_HEIGHT, IMG_WIDTH],
        "base_model":          "MobileNetV2",
        "dataset":             "PlantVillage",
    }
    report_path = os.path.join(OUTPUT_DIR, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"   Report saved        → {report_path}")
    return accuracy


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    start_time = datetime.datetime.now()

    print()
    print("=" * 60)
    print("  Harvest Savior — CNN Training Script (Phase 2)")
    print("=" * 60)

    # Check GPU availability (speeds up training 10-50×)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n🎮 GPU detected: {gpus[0].name}")
        print("   Training will be significantly faster.")
    else:
        print("\n💻 No GPU detected — training on CPU.")
        print("   Expected time: 20-60 minutes depending on your processor.")
        print("   Tip: Run this overnight or use Google Colab for free GPU.")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    class_names = verify_dataset()
    train_ds, val_ds, class_names, num_classes = build_datasets()
    model, base_model = build_model(num_classes)

    print("\n── Training begins ──────────────────────────────────────────")
    print(f"   Epochs (Phase A) : up to {EPOCHS} (early stopping enabled)")
    print(f"   Epochs (Phase B) : up to {FINE_TUNE_EPOCHS}")
    print(f"   Batch size       : {BATCH_SIZE}")
    print(f"   Model will save  → {MODEL_SAVE_PATH}")

    history_a = train_phase_a(model, train_ds, val_ds)
    history_b = train_phase_b(model, base_model, train_ds, val_ds)

    save_training_plots(history_a, history_b)
    accuracy = evaluate_model(model, val_ds, class_names)

    elapsed = datetime.datetime.now() - start_time
    mins    = int(elapsed.total_seconds() // 60)
    secs    = int(elapsed.total_seconds()  % 60)

    print()
    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Final accuracy : {accuracy * 100:.2f}%")
    print(f"  Time taken     : {mins}m {secs}s")
    print(f"  Model saved to : {MODEL_SAVE_PATH}")
    print()
    print("  Next step: restart the Flask server.")
    print("  It will automatically load your trained model and switch")
    print("  from DEMO MODE to REAL MODE.")
    print()
    print("  Flask start command:")
    print("  C:\\hs_venv\\Scripts\\python.exe app.py")
    print("=" * 60)
