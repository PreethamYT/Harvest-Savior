package com.harvestsavior.service;

import com.harvestsavior.model.Prediction;
import com.harvestsavior.repository.PredictionRepository;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * PredictionService — the business-logic layer.
 *
 * Responsibilities:
 *  1. Receive the uploaded image from the Controller.
 *  2. Call FlaskClient to get the CNN prediction result.
 *  3. Look up the remedy for the detected disease.
 *  4. Save the full result to the database via the Repository.
 *  5. Return the saved Prediction object to the Controller.
 *
 * The SERVICE layer is the correct place for business rules, not the Controller.
 * The Controller only handles HTTP; the Service handles logic.
 */
@Service
public class PredictionService {

    private final FlaskClient flaskClient;
    private final PredictionRepository predictionRepository;

    public PredictionService(FlaskClient flaskClient,
                             PredictionRepository predictionRepository) {
        this.flaskClient = flaskClient;
        this.predictionRepository = predictionRepository;
    }

    /**
     * Orchestrates the full prediction pipeline.
     *
     * @param imageFile — the image uploaded by the farmer
     * @return Prediction entity saved to the database
     */
    public Prediction predict(MultipartFile imageFile) throws IOException {

        // ── Step 1: Call Flask microservice ──────────────────────────────────
        Map<String, Object> flaskResponse = flaskClient.getPrediction(imageFile);

        String diseaseName = (String) flaskResponse.get("disease");
        double confidence  = ((Number) flaskResponse.get("confidence")).doubleValue();

        // ── Step 2: Look up the remedy for this disease ───────────────────────
        String remedy = REMEDY_MAP.getOrDefault(diseaseName,
                "No specific remedy found. Please consult your local agriculture officer.");

        // ── Step 3: Build and persist the entity ─────────────────────────────
        Prediction prediction = new Prediction();
        prediction.setImageFileName(imageFile.getOriginalFilename());
        prediction.setDiseaseName(formatDiseaseName(diseaseName));
        prediction.setConfidence(confidence);
        prediction.setRemedy(remedy);

        return predictionRepository.save(prediction);
    }

    /** Returns all past predictions, newest first (for the history page). */
    public List<Prediction> getHistory() {
        return predictionRepository.findAllByOrderByCreatedAtDesc();
    }

    // ── Helper: convert folder-style label → readable title ─────────────────
    // e.g. "Tomato_Early_blight"            → "Tomato Early Blight"
    //      "Potato___Late_blight"           → "Potato Late Blight"
    //      "Pepper__bell___Bacterial_spot"  → "Pepper Bell Bacterial Spot"
    private String formatDiseaseName(String raw) {
        // Strip the DEMO MODE suffix if present
        String name = raw.contains("[DEMO") ? raw.substring(0, raw.indexOf("[DEMO")).trim() : raw;
        // Replace any run of underscores with a single space, then title-case
        String spaced = name.replaceAll("_+", " ").trim();
        StringBuilder sb = new StringBuilder();
        for (String word : spaced.split(" ")) {
            if (!word.isEmpty()) {
                sb.append(Character.toUpperCase(word.charAt(0)))
                  .append(word.substring(1).toLowerCase())
                  .append(" ");
            }
        }
        return sb.toString().trim();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Remedy Map — maps PlantVillage class labels to remedy text.
    // Source labels follow the PlantVillage dataset naming convention.
    // Expand this map as you train the model on more classes.
    // ─────────────────────────────────────────────────────────────────────────
    private static final Map<String, String> REMEDY_MAP = Map.ofEntries(

        // ── Tomato ────────────────────────────────────────────────────────────
        Map.entry("Tomato_Early_blight",
            "Remove infected leaves immediately. Apply copper-based fungicide every 7 days. " +
            "Avoid overhead watering; water at the base of the plant. Rotate crops next season."),
        Map.entry("Tomato_Late_blight",
            "Destroy infected plants to prevent spread. Apply Mancozeb or Chlorothalonil fungicide. " +
            "Ensure good air circulation between plants. Avoid planting tomatoes in the same spot for 2 years."),
        Map.entry("Tomato_Leaf_Mold",
            "Improve ventilation. Apply sulfur-based fungicide. Remove and destroy severely infected leaves."),
        Map.entry("Tomato_Septoria_leaf_spot",
            "Apply fungicides containing chlorothalonil or copper. Remove infected lower leaves. " +
            "Mulch around the base to prevent soil splash."),
        Map.entry("Tomato_Spider_mites_Two_spotted_spider_mite",
            "Apply neem oil or insecticidal soap. Increase humidity around plants. " +
            "Introduce predatory mites (Phytoseiulus persimilis) as biological control."),
        Map.entry("Tomato__Target_Spot",
            "Apply Azoxystrobin or Difenoconazole fungicide. Remove infected leaves. " +
            "Ensure proper plant spacing for air circulation."),
        Map.entry("Tomato__Tomato_YellowLeaf__Curl_Virus",
            "No chemical cure exists. Remove and destroy infected plants immediately. " +
            "Control whitefly populations using yellow sticky traps and neem oil spray. " +
            "Use virus-resistant tomato varieties for future planting."),
        Map.entry("Tomato__Tomato_mosaic_virus",
            "Remove and destroy infected plants. Disinfect tools with bleach solution. " +
            "Control aphid vectors with insecticidal soap. Plant resistant varieties."),
        Map.entry("Tomato_Bacterial_spot",
            "Apply copper-based bactericide sprays. Remove and destroy infected plant material. " +
            "Avoid working with plants when wet. Use certified disease-free seeds."),
        Map.entry("Tomato_healthy",
            "Your tomato plant is healthy! Continue regular watering, fertilization, and monitoring."),

        // ── Potato ───────────────────────────────────────────────────────────
        Map.entry("Potato___Early_blight",
            "Apply chlorothalonil or mancozeb fungicide. Remove infected leaves. " +
            "Avoid overhead irrigation. Ensure adequate fertilization to strengthen the plant."),
        Map.entry("Potato___Late_blight",
            "Apply metalaxyl-based fungicide immediately. Destroy infected plants to prevent spread. " +
            "Do not store infected tubers. This disease caused the Irish Famine — take it seriously."),
        Map.entry("Potato___healthy",
            "Your potato plant is healthy! Maintain consistent watering and hill up soil around stems."),

        // ── Pepper ───────────────────────────────────────────────────────────
        Map.entry("Pepper__bell___Bacterial_spot",
            "Apply copper-based bactericide every 7–10 days. Remove infected leaves and fruit. " +
            "Avoid overhead irrigation. Use certified disease-free transplants next season."),
        Map.entry("Pepper__bell___healthy",
            "Your pepper plant is healthy! Ensure consistent moisture and full sunlight for best yield.")
    );
}
