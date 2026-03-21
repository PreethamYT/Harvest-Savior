package com.harvestsavior.controller;

import com.harvestsavior.model.Prediction;
import com.harvestsavior.service.PredictionService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

/**
 * HomeController — handles all browser-facing HTTP routes.
 *
 * @Controller (NOT @RestController) means this returns HTML page names,
 * not raw JSON. Spring looks for the matching Thymeleaf template in:
 *   src/main/resources/templates/<name>.html
 *
 * ROUTE MAP:
 *   GET  /           → shows the home/upload page (index.html)
 *   POST /predict    → receives the image, calls Flask, stores result → redirect to result page
 *   GET  /result     → shows the result page (result.html)
 *   GET  /history    → shows past predictions (history.html)
 */
@Controller
public class HomeController {

    private final PredictionService predictionService;

    public HomeController(PredictionService predictionService) {
        this.predictionService = predictionService;
    }

    /** Displays the upload form. */
    @GetMapping("/")
    public String home() {
        return "index";   // → templates/index.html
    }

    /**
     * Receives the uploaded image from the HTML form.
     *
     * "enctype=multipart/form-data" in the HTML form is mandatory to send binary
     * (image) data over HTTP. Spring's MultipartFile automatically captures it.
     *
     * After processing, we redirect to /result so a browser refresh does NOT
     * resubmit the form (Post/Redirect/Get pattern).
     */
    @PostMapping("/predict")
    public String predict(@RequestParam("image") MultipartFile imageFile,
                          RedirectAttributes redirectAttributes) {
        // Validate that the user actually selected a file
        if (imageFile.isEmpty()) {
            redirectAttributes.addFlashAttribute("error", "Please select an image file to upload.");
            return "redirect:/";
        }

        try {
            Prediction result = predictionService.predict(imageFile);
            // Flash attributes survive ONE redirect and then disappear
            redirectAttributes.addFlashAttribute("prediction", result);
            return "redirect:/result";

        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error",
                "Prediction failed: " + e.getMessage() +
                ". Please make sure the Flask AI server is running on port 5000.");
            return "redirect:/";
        }
    }

    /**
     * Displays the prediction result.
     * The 'prediction' object was placed in the model by the redirect above.
     */
    @GetMapping("/result")
    public String result(Model model) {
        // If someone navigates directly to /result without going through /predict,
        // the model will have no prediction → show them a friendly message via the template.
        return "result";  // → templates/result.html
    }

    /** Displays all past predictions from the database. */
    @GetMapping("/history")
    public String history(Model model) {
        model.addAttribute("predictions", predictionService.getHistory());
        return "history";  // → templates/history.html
    }
}
