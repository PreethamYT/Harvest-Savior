"""
train_stage1.py
---------------
Stage 1: Fast Training (128×128 resolution)
Goal: Get a working model in ~30 minutes with 96-97% accuracy

This creates a base model that Stage 2 will fine-tune to 99%+

Run:
    python train_stage1.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ===================== CONFIG =====================
BASE_DIR = os.path.join(os.path.dirname(__file__), "Dataset")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "harvest_model_stage1.h5")

IMG_SIZE = 128  # Smaller = 4× faster
BATCH_SIZE = 64  # Larger batches for speed
EPOCHS = 15      # Fewer epochs
# Auto-detect number of classes from dataset folders
CLASSES = len([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])

print("🌱 Harvest Savior — Stage 1: Fast Training")
print("="*60)
print(f"Training on {CLASSES} classes | Resolution: {IMG_SIZE}×{IMG_SIZE}")
print("Target: 96-97% accuracy | Time: ~30 minutes")

# ===================== DATA LOADING =====================
print("\n[1/4] Loading Dataset...")

# Lighter augmentation for speed
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15
)

train_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print(f"   • Training samples: {train_generator.samples}")
print(f"   • Validation samples: {val_generator.samples}")
print(f"   • Classes: {list(train_generator.class_indices.keys())}")

# ===================== MODEL ARCHITECTURE =====================
print("\n[2/4] Building CNN Architecture...")

model = tf.keras.Sequential([
    # Block 1: 32 filters
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 2: 64 filters
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 3: 128 filters
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 4: 256 filters
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"   • Total Parameters: {model.count_params():,}")

# ===================== TRAINING =====================
print("\n[3/4] Training Model...")
print("   ⏱️  Estimated time: ~30 minutes\n")

# Callbacks
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# ===================== SAVE MODEL =====================
print("\n[4/4] Saving Model...")
model.save(MODEL_PATH)

# Verify file size
size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"\n{'='*60}")
if size_mb > 2:
    print(f"✅ STAGE 1 COMPLETE! Model saved: harvest_model_stage1.h5 ({size_mb:.2f} MB)")
    print(f"   • Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"   • Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
    print(f"\n📌 Next Step: Run train_stage2.py to fine-tune to 99%+")
else:
    print(f"⚠️  WARNING: Model file is only {size_mb:.2f} MB (expected >2MB)")
print("="*60)
