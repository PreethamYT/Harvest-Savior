"""
app.py — Flask Microservice Entry Point
========================================
This is the AI/ML server for Harvest Savior.

HOW THIS SERVICE FITS INTO THE ARCHITECTURE (important for viva):
──────────────────────────────────────────────────────────────────
  [Farmer's Browser]
        │  HTTP POST (image)
        ▼
  [Spring Boot :8080]   ← handles the UI, stores results in DB
        │  HTTP POST (image forwarded) via RestTemplate
        ▼
  [Flask :5000  ← THIS FILE]  ← loads CNN model, runs inference
        │  JSON response { "disease": "...", "confidence": 93.7 }
        ▼
  [Spring Boot :8080]   ← saves to DB, renders result page

Flask is a lightweight Python web framework.
One route is all we need: POST /predict
"""

from flask import Flask, request, jsonify
from utils.predictor import Predictor
import traceback

# ── Create the Flask application instance ────────────────────────────────────
app = Flask(__name__)

# ── Load the CNN model once at startup (not on every request) ─────────────────
# Loading a model is expensive (seconds). We load it once and reuse it.
predictor = Predictor()


# ── Health Check Route ───────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    """
    Simple endpoint so Spring Boot (or a developer) can confirm Flask is running.
    Visit http://localhost:5000/health in your browser.
    """
    return jsonify({"status": "ok", "message": "Harvest Savior AI Service is running."})


# ── Prediction Route ─────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives a leaf image from Spring Boot and returns a JSON prediction.

    Expected request:  multipart/form-data with field "image"
    Returns:           JSON → { "disease": "<label>", "confidence": <float> }

    Error handling:
      - If no image field → 400 Bad Request
      - If model fails    → 500 Internal Server Error
    """

    # ── Step 1: Validate that an image was sent ───────────────────────────────
    if 'image' not in request.files:
        return jsonify({"error": "No image field found in the request."}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "Image file has no filename."}), 400

    # ── Step 2: Run the CNN predictor ─────────────────────────────────────────
    try:
        result = predictor.predict(image_file)
        # result = {"disease": "Tomato__Early_blight", "confidence": 93.7}
        return jsonify(result), 200

    except Exception as e:
        # Log the full stack trace to the Flask console for debugging
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ── Run the server ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    """
    debug=True  → auto-reloads when you edit Python files (development only).
    host='0.0.0.0' → listens on all network interfaces, not just localhost.
    port=5000   → must match the flask.base-url in Spring Boot's application.properties.
    """
    print("=" * 55)
    print("  Harvest Savior AI Microservice")
    print("  Listening on http://localhost:5000")
    print("  Health check: http://localhost:5000/health")
    print("=" * 55)
    app.run(debug=True, host='0.0.0.0', port=5000)
