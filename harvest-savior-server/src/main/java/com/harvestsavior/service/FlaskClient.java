package com.harvestsavior.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;

/**
 * FlaskClient — the bridge between Spring Boot and the Python Flask microservice.
 *
 * HOW IT WORKS (important for viva):
 * ─────────────────────────────────
 * 1. The farmer's browser sends the leaf image to Spring Boot (port 8080).
 * 2. Spring Boot (this class) forwards the image to the Flask AI server (port 5000)
 *    using an HTTP POST request with "multipart/form-data" encoding.
 * 3. Flask runs the image through the CNN model and returns a JSON response:
 *    { "disease": "Tomato__Early_blight", "confidence": 93.7 }
 * 4. Spring Boot reads that JSON, maps it to a Java Map, and passes it up to PredictionService.
 *
 * RestTemplate is Spring's built-in HTTP client — like a programmatic web browser.
 */
@Service
public class FlaskClient {

    /**
     * The base URL of the Flask microservice.
     * Value is read from application.properties → flask.base-url
     * Default: http://localhost:5000  (runs on the same machine)
     */
    @Value("${flask.base-url:http://localhost:5000}")
    private String flaskBaseUrl;

    private final RestTemplate restTemplate;

    public FlaskClient(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    /**
     * Sends the image file to Flask's /predict endpoint.
     *
     * @param imageFile — the MultipartFile uploaded by the user from the HTML form
     * @return a Map containing "disease" (String) and "confidence" (Number)
     * @throws IOException if the file bytes cannot be read
     */
    @SuppressWarnings("unchecked")
    public Map<String, Object> getPrediction(MultipartFile imageFile) throws IOException {

        // Step 1: Set the Content-Type header to multipart/form-data
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        // Step 2: Wrap the image bytes into a "resource" Spring can send over HTTP
        ByteArrayResource imageResource = new ByteArrayResource(imageFile.getBytes()) {
            @Override
            public String getFilename() {
                // Flask needs a filename to detect the image format (jpg/png)
                return imageFile.getOriginalFilename();
            }
        };

        // Step 3: Build the multipart body — "image" matches the field name Flask expects
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("image", imageResource);

        // Step 4: Wrap headers + body into one HttpEntity object
        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        // Step 5: POST to Flask and receive the JSON response as a plain Java Map
        ResponseEntity<Map> response = restTemplate.exchange(
                flaskBaseUrl + "/predict",
                HttpMethod.POST,
                requestEntity,
                Map.class
        );

        return response.getBody();
    }
}
