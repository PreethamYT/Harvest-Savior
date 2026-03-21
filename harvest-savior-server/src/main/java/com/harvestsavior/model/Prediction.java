package com.harvestsavior.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

/**
 * JPA Entity — represents one prediction record stored in the database.
 *
 * @Entity  → Hibernate will map this class to a database table called "prediction".
 * @Id + @GeneratedValue → the primary key is auto-incremented (1, 2, 3, ...).
 *
 * Each time a farmer uploads a leaf image, one row is inserted into this table
 * containing: which disease was detected, the confidence score, and the remedy.
 */
@Entity
@Table(name = "prediction")
public class Prediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /** Original filename of the uploaded image (e.g. "leaf_photo.jpg") */
    @Column(nullable = false)
    private String imageFileName;

    /** Disease label returned by the Flask CNN model (e.g. "Tomato__Early_blight") */
    @Column(nullable = false)
    private String diseaseName;

    /** Confidence percentage from the model's softmax output (e.g. 93.7) */
    @Column(nullable = false)
    private double confidence;

    /** Remedy text looked up from our remedy map in PredictionService */
    @Column(length = 1000)
    private String remedy;

    /** Timestamp of when the prediction was made */
    @Column(nullable = false)
    private LocalDateTime createdAt;

    @PrePersist  // automatically called by JPA just before INSERT
    public void prePersist() {
        this.createdAt = LocalDateTime.now();
    }

    // ── getters & setters ─────────────────────────────────────────────────────

    public Long getId()                          { return id; }
    public String getImageFileName()             { return imageFileName; }
    public void setImageFileName(String v)       { this.imageFileName = v; }
    public String getDiseaseName()               { return diseaseName; }
    public void setDiseaseName(String v)         { this.diseaseName = v; }
    public double getConfidence()                { return confidence; }
    public void setConfidence(double v)          { this.confidence = v; }
    public String getRemedy()                    { return remedy; }
    public void setRemedy(String v)              { this.remedy = v; }
    public LocalDateTime getCreatedAt()          { return createdAt; }
}
