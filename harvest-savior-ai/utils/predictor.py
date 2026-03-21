"""
predictor.py — CNN Model Loader and Inference Engine
======================================================
This module is the core of the AI microservice.

LOGIC EXPLAINED STEP BY STEP (viva-ready):
──────────────────────────────────────────
1. When Flask starts, the Predictor class is instantiated ONCE.
2. __init__() checks if a trained .h5 model file exists in the model/ folder.
   - If YES  → TensorFlow is imported and the model is loaded from disk.
   - If NO   → Flask runs in DEMO MODE. TensorFlow is NOT imported at all,
               so the server starts instantly without needing TF installed.
               Demo mode returns a realistic dummy prediction so you can
               test the full Spring Boot ↔ Flask pipeline immediately.

3. When predict(image_file) is called for each incoming image:
   a. The image bytes are read and decoded by PIL (Pillow).
   b. The image is RESIZED to 224×224 pixels — the fixed CNN input size.
   c. Pixel values are NORMALIZED from [0-255] to [0.0-1.0] by ÷ 255.
   d. A batch dimension is ADDED: (224, 224, 3) → (1, 224, 224, 3).
   e. model.predict() runs a forward pass through all CNN layers.
   f. np.argmax() selects the class with the highest softmax probability.
   g. Confidence = that probability × 100 (e.g. 0.937 → 93.7%).

WHY LAZY IMPORT?
   TensorFlow is a 300+ MB library. On Windows, installing it requires
   "Long Path" support enabled (a system registry setting). By deferring
   the import to only when a real model file is present, Flask starts
   successfully on any machine, even before TF is fully installed.
"""

import os
import io
import json
import random


# ── Absolute path to the saved model file ────────────────────────────────────
# This file is created by train_model.py (Phase 2).
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'crop_disease_cnn.h5')

# ── Class names JSON (generated alongside the model by train_model.py) ────────
# Reading from this file keeps predictor.py in sync with whatever classes
# and ordering were actually used during training.  If class_names.json does
# not exist we fall back to the hardcoded list below.
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'class_names.json')

# ── Input image dimensions (must match training configuration) ────────────────
# The custom CNN in harvest_model.h5 was trained at 256×256.
IMG_HEIGHT = 256
IMG_WIDTH  = 256

# ── Fallback class labels (used in DEMO MODE or before class_names.json exists) ─
# After training, class_names.json will override this list automatically.
CLASS_NAMES = [
    "Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)__Common_rust_",
    "Corn_(maize)__Northern_Leaf_Blight",
    "Corn_(maize)__healthy",
    "Potato__Early_blight",
    "Potato__Late_blight",
    "Potato__healthy",
    "Tomato__Early_blight",
    "Tomato__Late_blight",
    "Tomato__Leaf_Mold",
    "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites Two-spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__healthy",
]


def _load_class_names():
    """Load class names from the JSON file written by train_model.py.
    Falls back to the hardcoded CLASS_NAMES if the file does not yet exist."""
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r') as f:
            names = json.load(f)
        print(f"[Predictor] Loaded {len(names)} class names from class_names.json")
        return names
    print("[Predictor] class_names.json not found — using hardcoded CLASS_NAMES.")
    return list(CLASS_NAMES)


class Predictor:
    """
    Encapsulates the CNN model loading and inference logic.

    Why a class instead of plain functions?
    → We need to store `self.model` in memory between requests.
      A class instance persists for the lifetime of the Flask process.
    """

    def __init__(self):
        self.model = None
        self.demo_mode = False
        self.class_names = _load_class_names()
        self._load_model()

    def _load_model(self):
        """
        Attempts to load the trained model from disk.
        Falls back to DEMO MODE if the model file does not exist.

        TensorFlow is ONLY imported inside this method (lazy import).
        This means Flask can start successfully even without TF installed,
        as long as no model file is present (demo mode).
        """
        if os.path.exists(MODEL_PATH):
            # ── Real mode: TF is imported only here ─────────────────────────
            print(f"[Predictor] Loading model from: {MODEL_PATH}")
            try:
                import tensorflow as tf
                from tensorflow import keras as _keras
                self._keras = _keras
                self._np = __import__('numpy')
                self.model = _keras.models.load_model(MODEL_PATH)
                print(f"[Predictor] Model loaded. Input shape: {self.model.input_shape}")
                # Reload class names — the JSON may now exist if model just trained
                self.class_names = _load_class_names()
                print(f"[Predictor] Model ready. Classes: {len(self.class_names)}")
            except ImportError:
                print("[Predictor] ⚠  TensorFlow not available. Falling back to DEMO MODE.")
                self.demo_mode = True
        else:
            # ── Demo mode: no TF needed ──────────────────────────────────────
            print("[Predictor] ⚠  No trained model found at:", MODEL_PATH)
            print("[Predictor] ⚠  Running in DEMO MODE — returns a realistic dummy prediction.")
            print("[Predictor] ⚠  Train the CNN (Phase 2) and save to model/ to enable real predictions.")
            self.demo_mode = True

    def predict(self, image_file) -> dict:
        """
        Runs inference on the uploaded image file.

        :param image_file: werkzeug FileStorage object (from Flask request.files)
        :return: dict with keys "disease" (str) and "confidence" (float)

        In DEMO MODE — returns a hard-coded realistic result so you can test
        the full Spring Boot ↔ Flask pipeline without a trained model.

        In REAL MODE — runs the image through the CNN:
          Step 1: Read bytes → PIL Image (ensures correct decoding of jpg/png)
          Step 2: Resize to 224×224 (fixed CNN input size)
          Step 3: Normalise pixel values to [0.0, 1.0] by dividing by 255
          Step 4: Add batch dimension: (224,224,3) → (1,224,224,3)
          Step 5: model.predict() → softmax probability array of shape (1, N_CLASSES)
          Step 6: argmax picks the class with the highest probability
          Step 7: confidence = that probability × 100
        """
        # ── DEMO MODE ─────────────────────────────────────────────────────────
        if self.demo_mode:
            # Return a realistic dummy result — useful for testing the pipeline
            # before the model is trained. Spring Boot will store and display this.
            demo_choices = [
                "Tomato__Early_blight",
                "Tomato__healthy",
                "Potato__Late_blight",
                "Corn_(maize)__Common_rust_",
            ]
            disease    = random.choice(demo_choices)
            confidence = round(random.uniform(65.0, 97.0), 2)
            return {
                "disease":    disease + " [DEMO MODE — train model for real predictions]",
                "confidence": confidence
            }

        # ── REAL MODE ─────────────────────────────────────────────────────────
        from PIL import Image as _PIL_Image
        np = self._np

        # Step 1: Decode the uploaded bytes into a PIL Image
        image_bytes = image_file.read()
        image = _PIL_Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Step 2: Resize to 224 × 224 (must match training configuration)
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))

        # Step 3: Normalise pixel values to [0.0, 1.0] by dividing by 255.
        # WHY /255? The custom CNN was trained using ImageDataGenerator with
        # rescale=1./255, meaning it expects float values in [0, 1].
        # This is different from MobileNetV2 which uses [-1, 1] internally.
        img_array = np.array(image, dtype=np.float32) / 255.0  # shape: (256, 256, 3)

        # Step 4: Add batch dimension — model always expects shape (batch, H, W, C)
        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

        # Step 5: Forward pass through the CNN
        predictions = self.model.predict(img_array, verbose=0)

        # Step 6: Pick the class with the highest softmax probability
        class_index = int(np.argmax(predictions[0]))
        confidence  = float(predictions[0][class_index]) * 100.0

        return {
            "disease":    self.class_names[class_index],
            "confidence": round(confidence, 2)
        }
