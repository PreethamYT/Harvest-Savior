# Harvest Savior
### AI Based Crop Disease Detection and Remedy Recommendation System
*GRIET Academic Project — 2nd Year B.Tech*

---

## Project Abstract

Crop diseases significantly reduce agricultural yield and threaten global food security especially for small scale farmers using low end mobile devices. This project falls under the domain of basic Artificial Intelligence and Machine Learning. To address this agricultural challenge we propose Harvest Savior which is an artificial intelligence based crop disease detection and remedy recommendation system. The software architecture utilizes a Java Spring Boot framework to seamlessly handle user interactions alongside a Python Flask microservice to perform deep learning inference. We successfully trained a MobileNetV2 Convolutional Neural Network on the prominent PlantVillage dataset applying advanced data augmentation techniques to improve visual robustness. The system resizes uploaded leaf images to specific pixel dimensions and processes them through the deep neural network to output a mathematical probability distribution representing model confidence. The expected outcome of the proposed work is a highly accurate web application that reliably identifies fifteen different crop diseases and provides actionable mitigation strategies. This immediate feedback loop empowers rural farmers to take corrective action to protect their crops and maximize their overall harvest yield.

---

## Project Highlights

* **High Accuracy:** Our fine-tuned MobileNetV2 architecture achieves **96.29% validation accuracy** using advanced Data Augmentation and Early Stopping optimization.
* **Fast Inference:** Real-time predictions execute in under 2 seconds leveraging a lightweight `.h5` model.
* **Microservice Design:** Strict separation of the Java Spring Boot UI/routing and Python Flask AI inference.

---

## Project Architecture

```
[Farmer's Browser]
       │  upload leaf image (HTTP)
       ▼
[Spring Boot :8080]  — Java — UI + DB
       │  forwards image via REST call
       ▼
[Flask :5000]        — Python — CNN Model
       │  returns JSON {"disease": "...", "confidence": 93.7}
       ▼
[Spring Boot :8080]  — saves to H2 DB, renders result page
```

---

## Project Structure

```
HarvestSavior/
├── harvest-savior-server/          ← Java Spring Boot
│   ├── pom.xml
│   └── src/main/
│       ├── java/com/harvestsavior/
│       │   ├── HarvestSaviorApplication.java   ← Entry point
│       │   ├── config/AppConfig.java            ← RestTemplate Bean
│       │   ├── controller/HomeController.java   ← HTTP routes
│       │   ├── model/Prediction.java            ← JPA Entity (DB table)
│       │   ├── repository/PredictionRepository  ← Data access layer
│       │   └── service/
│       │       ├── FlaskClient.java             ← Calls Flask API
│       │       └── PredictionService.java       ← Business logic + Remedy map
│       └── resources/
│           ├── application.properties
│           └── templates/
│               ├── index.html    ← Upload page
│               ├── result.html   ← Result page
│               └── history.html  ← History page
│
└── harvest-savior-ai/              ← Python Flask
    ├── app.py                      ← Flask entry point (/predict route)
    ├── requirements.txt
    ├── utils/
    │   └── predictor.py            ← CNN loader + inference logic
    └── model/
        └── crop_disease_cnn.h5     ← (generated after training — Phase 2)
```

---

## How to Run

### Prerequisites
| Tool | Required version |
|------|-----------------|
| Java | 17+ |
| Maven | 3.6+ |
| Python | 3.9+ |
| pip | latest |

---

### Step 1 — Start the Python Flask AI Service

```bash
cd harvest-savior-ai
pip install -r requirements.txt
python app.py
```

Flask starts on **http://localhost:5000**  
Verify: open http://localhost:5000/health in your browser.

> **Note:** Without a trained model (`model/crop_disease_cnn.h5`), Flask runs in
> **DEMO MODE** — it still accepts requests but returns untrained predictions.
> Run the training script in Phase 2 to enable real predictions.

---

### Step 2 — Start the Java Spring Boot Server

```bash
cd harvest-savior-server
mvnw.cmd spring-boot:run       # Windows
./mvnw spring-boot:run         # Mac/Linux
```

Spring Boot starts on **http://localhost:8080**  
Open your browser and go to: **http://localhost:8080**

> For Maven to work without a system install, Spring Boot generates a Maven Wrapper
> (`mvnw`/`mvnw.cmd`) inside the project. Run `mvn wrapper:wrapper` once if it
> does not exist yet, or download Maven manually from https://maven.apache.org.

---

### Step 3 — Use the Application

1. Open **http://localhost:8080**
2. Upload a photo of a crop leaf
3. Click **Analyse Leaf**
4. The result page shows the detected disease and the recommended remedy
5. Visit **http://localhost:8080/history** to see all past predictions

---

### H2 Database Console (Development)

While Spring Boot is running, visit:  
**http://localhost:8080/h2-console**

- JDBC URL: `jdbc:h2:mem:harvestsaviordb`
- Username: `sa`
- Password: *(leave blank)*

---

## Phase Roadmap

| Phase | Task | Status |
|-------|------|--------|
| 1 | Project scaffold (Spring Boot + Flask + CNN architecture) | ✅ Done |
| 2 | Train CNN on PlantVillage dataset, save model files | ✅ Done |
| 3 | Migrate from local H2 to MySQL (production DB) | ⏭️ Skipped (H2 optimal for local demo) |
| 4 | Final UI polish, abstract formulation | ✅ Done (Ready for submission) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend / Server | Java 17, Spring Boot 3.2, Thymeleaf |
| AI Microservice | Python 3.9, Flask 3.0 |
| Deep Learning | TensorFlow 2.16+ / Keras 3 (MobileNetV2 CNN) |
| Dataset | PlantVillage (15 disease classes) |
| Database | H2 (dev) → MySQL (prod) |
| Build Tool | Maven |

---

*Harvest Savior — Gokaraju Rangaraju Institute of Engineering and Technology*
