package com.harvestsavior.repository;

import com.harvestsavior.model.Prediction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * Repository — the data-access layer for the Prediction entity.
 *
 * By extending JpaRepository<Prediction, Long>, Spring Data JPA automatically
 * generates the SQL for common operations:
 *   - save(prediction)         → INSERT or UPDATE
 *   - findById(id)             → SELECT WHERE id = ?
 *   - findAll()                → SELECT * FROM prediction
 *   - deleteById(id)           → DELETE WHERE id = ?
 *
 * We do NOT write any SQL ourselves — the framework handles it.
 */
@Repository
public interface PredictionRepository extends JpaRepository<Prediction, Long> {

    /**
     * Spring Data JPA reads the method name and auto-generates:
     * SELECT * FROM prediction ORDER BY created_at DESC
     */
    List<Prediction> findAllByOrderByCreatedAtDesc();
}
