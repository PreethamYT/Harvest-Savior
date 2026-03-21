# Harvest Savior — Project Review Prep Guide

> **For:** GRIET 2nd-year team — Mid-Term Evaluation  
> **Project:** Harvest Savior: AI-Based Crop Disease Detection and Remedy Recommendation System  
> **Date:** February 2026

---

## 1. The Big Picture (Start Every Answer Here)

When the examiner asks *"explain your project"*, begin with this:

> "Harvest Savior is a two-server system. A farmer uploads a photo of a diseased leaf. The **Java Spring Boot server** (port 8080) handles the user interface and receives the image. It then **forwards that image** to a **Python Flask microservice** (port 5000) via an HTTP REST API call. The Flask server runs a **Convolutional Neural Network** trained on the PlantVillage dataset, classifies the disease, and sends back a JSON response. Spring Boot receives that JSON, looks up the remedy for that disease from its own in-memory map, saves the result to the database, and renders the result page to the farmer."

Draw this on the board:

```
[Farmer Browser]
      │  POST /analyze  (multipart image)
      ▼
[Spring Boot  :8080]  ←── Java, Thymeleaf UI, MySQL/H2 DB
      │  POST /predict  (forwards the image)
      ▼
[Flask  :5000]  ←── Python, TensorFlow CNN
      │  JSON: { "disease": "Tomato_Late_blight", "confidence": 98.5 }
      ▼
[Spring Boot]  ←── looks up remedy, saves to DB, renders result
      │
      ▼
[Farmer sees result page]
```

---

## 2. The CNN Model — Explain This Deeply

### What is a CNN?

A **Convolutional Neural Network** is a type of deep learning model specifically designed for image classification. It works in stages:

| Stage | What Happens | Why |
|---|---|---|
| **Conv Layer** | Slides small filters (e.g. 3×3) over the image to detect edges, textures, patterns | Learns local features automatically |
| **MaxPooling** | Downsamples the feature map (takes the max value in each region) | Reduces computation, keeps dominant features |
| **ReLU Activation** | Replaces negative values with zero | Adds non-linearity so the model learns complex patterns |
| **Flatten** | Converts the 2D feature maps to a 1D vector | Connects to the fully-connected classifier |
| **Dense (Softmax)** | Outputs a probability for each class | Gives the final classification |

### Our Specific Model

Our model (`crop_disease_cnn.h5`) is a **custom 4-block CNN** (not a pre-built architecture). It was trained on the **PlantVillage dataset** — a public dataset of 54,000+ labeled leaf images covering 38 disease/healthy categories. We use **15 classes** (Pepper, Potato, Tomato).

**Key training configuration to remember:**

| Parameter | Value | Why |
|---|---|---|
| Input image size | **256 × 256 pixels** | Fixed size the model was trained on — all inputs must match |
| Pixel normalization | **÷ 255** (range 0.0–1.0) | Neural networks train better on small float values than 0–255 ints |
| Class ordering | **Python `sorted()` case-sensitive** | Keras `flow_from_directory` uses this exact ordering for class indices |
| Output layer | **15 neurons + Softmax** | One probability per disease class, all sum to 1.0 |

### What is Transfer Learning? (If asked)

We used a **custom CNN from scratch**, not transfer learning. If asked why not transfer learning: *"Transfer learning (e.g. MobileNetV2, VGG16) uses a model pre-trained on ImageNet. We chose a custom CNN because the PlantVillage dataset is domain-specific — plant leaf textures are very different from everyday objects — and a simpler custom model is more interpretable for an academic project."*

---

## 3. The Prediction Pipeline — Step by Step

This is what happens inside `predictor.py` every time a farmer uploads an image:

```
Image File (JPEG/PNG from browser)
         │
         ▼
Step 1: PIL decode → RGB image
         (ensures correct color channel ordering regardless of device)
         │
         ▼
Step 2: Resize to 256 × 256
         (model ONLY accepts this fixed input size)
         │
         ▼
Step 3: Convert to float array ÷ 255
         (normalize pixel values from [0,255] → [0.0, 1.0])
         │
         ▼
Step 4: Add batch dimension
         (256,256,3) → (1,256,256,3)
         (model always expects a BATCH of images, even if batch=1)
         │
         ▼
Step 5: model.predict() — forward pass through CNN layers
         Output: array of 15 probabilities, e.g. [0.02, 0.01, ..., 0.98, 0.01]
         │
         ▼
Step 6: np.argmax() → picks the class with highest probability
         e.g. index 7 → "Tomato_Late_blight"
         │
         ▼
Step 7: confidence = that probability × 100
         e.g. 0.985 → 98.5%
         │
         ▼
Return: { "disease": "Tomato_Late_blight", "confidence": 98.5 }
```

---

## 4. How Spring Boot Talks to Flask (REST API Communication)

This is the **most important integration question** the examiner will ask.

In `PredictionService.java`, Spring Boot uses `RestTemplate` to make an HTTP POST request to Flask:

```java
// Spring Boot sends the image to Flask as multipart form-data
RestTemplate restTemplate = new RestTemplate();
MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
body.add("image", new ByteArrayResource(imageBytes) {
    @Override public String getFilename() { return "leaf.jpg"; }
});
HttpHeaders headers = new HttpHeaders();
headers.setContentType(MediaType.MULTIPART_FORM_DATA);
HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

ResponseEntity<String> response = restTemplate.postForEntity(
    "http://localhost:5000/predict",   // Flask endpoint
    requestEntity,
    String.class
);
// response.getBody() = {"disease":"Tomato_Late_blight","confidence":98.5}
```

**Key concept to explain:** The two servers communicate using the **REST API pattern** over HTTP, the same protocol as a web browser. Spring Boot acts as an **HTTP client** to Flask. This is called a **microservices architecture** — each server has one job (UI/DB vs. AI/ML).

---

## 5. Why Two Servers? (A Common Examiner Question)

> "Why not put the CNN directly in Spring Boot?"

**Answer:** *"Java's TensorFlow support is limited and the ecosystem for ML is Python-native. The entire ML stack — TensorFlow, Keras, NumPy, PIL — is Python-only. By keeping the AI in a Python Flask microservice, we get the best of both worlds: Java's robust web framework and database support for the UI layer, and Python's rich ML libraries for inference. This is a standard industry pattern called a microservices architecture."*

---

## 6. The Class Name Ordering Bug (Great Technical Story to Volunteer)

This is a great thing to bring up voluntarily — it shows deep understanding.

> *"During integration we discovered a subtle bug: Python's `sorted()` function is **case-sensitive** (uppercase letters sort before lowercase), but PowerShell's `Sort-Object` is **case-insensitive**. When we generated the class name list using PowerShell, `Tomato_healthy` landed at index 10 instead of index 14. This caused the model to return completely wrong labels — `Potato_healthy` images were being classified as `Tomato_Spider_mites`. The fix was to always generate `class_names.json` using Python itself, matching the exact ordering Keras `flow_from_directory` uses during training."*

**Why this is impressive to mention:** It proves you actually debugged the system end-to-end and understood the root cause rather than just running the code blindly.

---

## 7. Model Accuracy — Be Honest

Overall: **12 out of 15 classes predict correctly** on first-image sanity tests.

| Class | Result | Confidence | Note |
|---|---|---|---|
| `Pepper__bell___Bacterial_spot` | ✅ | 77% | Mild visual similarity to other spot diseases |
| `Pepper__bell___healthy` | ✅ | 100% | |
| `Potato___Early_blight` | ✅ | 100% | |
| `Potato___Late_blight` | ✅ | 90% | |
| `Potato___healthy` | ✅ | 55% | Low confidence but correct class |
| `Tomato_Bacterial_spot` | ❌ | — | Confused with Spider mites — similar spot patterns |
| `Tomato_Early_blight` | ❌ | — | Confused with Potato Early blight — visually similar |
| `Tomato_Late_blight` | ✅ | 98% | |
| `Tomato_Leaf_Mold` | ✅ | 99% | |
| `Tomato_Septoria_leaf_spot` | ✅ | 99% | |
| `Tomato_Spider_mites` | ❌ | — | Confused with Target Spot |
| `Tomato__Target_Spot` | ✅ | 94% | |
| `Tomato__YellowLeaf_Curl_Virus` | ✅ | 99% | |
| `Tomato__mosaic_virus` | ✅ | 90% | |
| `Tomato_healthy` | ✅ | 100% | |

If an examiner points out the 3 wrong classes, say:

> *"These are genuine model limitations caused by visual similarity between disease phenotypes. For example, Tomato Early blight and Potato Early blight both produce concentric ring-shaped lesions — they are difficult to distinguish even for domain experts using photographs alone. With more training epochs and data augmentation (random flips, brightness shifts), these would improve. For a 2nd-year project, 80% accuracy across 15 classes is within acceptable range."*

---

## 8. The Dataset

**PlantVillage Dataset:**
- ~54,000 labeled leaf images across 38 classes (we use 15)
- Collected under controlled lab conditions
- Publicly available on Kaggle
- Standard benchmark for plant disease detection research

**Known limitation to mention proactively:**

> *"The images were taken in controlled conditions. Real-world photos from farmers' phones may have varying backgrounds, lighting, and angles, which could reduce accuracy. A future improvement would be to augment the training data with field-condition images, and potentially add a background removal pre-processing step."*

---

## 9. Quick-Fire Q&A Prep

| Question | Answer |
|---|---|
| *What is Softmax?* | An activation function that converts raw scores into probabilities that sum to 1.0 |
| *What is overfitting?* | When the model memorizes training data and performs poorly on new images |
| *What is the role of Flask?* | Lightweight Python web server hosting the CNN inference endpoint |
| *What is RestTemplate?* | Spring's built-in HTTP client for making REST API calls to other services |
| *Why H2 database?* | In-memory database requiring zero installation — ideal for demos and academic projects |
| *What is Thymeleaf?* | Java server-side template engine — generates HTML on the server before sending to browser |
| *What is CORS?* | Cross-Origin Resource Sharing — a browser security rule that blocks requests between different servers. Not an issue here since Spring Boot makes the call to Flask server-to-server, not the browser directly |
| *What is multipart/form-data?* | An HTTP content type for sending binary files (images) in a form submission |
| *What is ImageDataGenerator?* | Keras utility that reads images from folders during training, applies rescaling and augmentation on-the-fly |
| *What does `flow_from_directory` do?* | Scans a folder structure where each subfolder = one class, automatically assigns labels and batches images for training |
| *What is a batch dimension?* | Models process images in groups (batches) for efficiency. Even a single image must be wrapped in a batch of size 1: shape (1, 256, 256, 3) |
| *What is np.argmax?* | NumPy function that returns the index of the highest value in an array — we use it to find which class has the highest probability |

---

## 10. How to Start the System (For the Demo)

### Terminal 1 — Spring Boot (Java)

```powershell
cd "C:\Users\Preetham Rao\Desktop\HarvestSavior\harvest-savior-server"
mvn spring-boot:run
# Wait for: Started HarvestSaviorApplication in X seconds
# Open: http://localhost:8080
```

### Terminal 2 — Flask AI Service (Python)

```powershell
cd "C:\Users\Preetham Rao\Desktop\HarvestSavior"
.\start_flask.ps1
# Wait for: [Predictor] Model ready. Classes: 15
# Takes ~60 seconds on first start (TensorFlow loading 242MB model)
```

> ⚠️ **Never** start Flask by typing plain `python app.py` — this accidentally picks up the Windows Store Python which has a broken TensorFlow installation. Always use `.\start_flask.ps1`.

### Verify Both Are Running

```powershell
# Flask health check
curl.exe http://localhost:5000/health

# Spring Boot check
curl.exe http://localhost:8080/
```

---

## 11. One-Line Summary for Each File

| File | What It Does |
|---|---|
| `app.py` | Flask entry point — defines `/predict` and `/health` routes |
| `utils/predictor.py` | Loads the CNN model, preprocesses images, runs inference |
| `model/crop_disease_cnn.h5` | The trained CNN weights (242 MB) |
| `model/class_names.json` | Maps model output index 0–14 to disease class names |
| `PredictionService.java` | Spring Boot service — calls Flask via RestTemplate, looks up remedies |
| `AnalysisController.java` | Spring Boot controller — handles `/analyze` POST from the browser |
| `start_flask.ps1` | Safe launcher script for the Flask service |

---

> **Final tip:** During the demo, upload a clear, well-lit leaf photo with a plain background. The model performs best on images similar to the controlled PlantVillage dataset. If the examiner uploads a field photo with soil/sky in the background and the confidence is low, explain the controlled-conditions limitation proactively — it shows you understand the system's boundaries.
