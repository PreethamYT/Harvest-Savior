import os
from flask import Flask, jsonify

# Initialize the Flask application
app = Flask(__name__)

#hello

# Route 1: The Health Check
# This lets us verify if the Python server is running without doing any complex AI yet.
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "system": "Harvest Savior AI Engine",
        "message": "The brain is active and waiting for leaves! 🌿"
    })

# Main entry point
if __name__ == '__main__':
    # We run on port 5000. The Java app will send requests here.
    print("Starting Harvest Savior AI Engine...")
    app.run(host='0.0.0.0', port=5000, debug=True)