

# MBSQI: Migratory Bird Stopover Quality Index 🦅

**An Uncertainty-Aware Ecosystem Intelligence Framework for Multi-Species Migratory Bird Conservation** 

This repository contains the code and methodology for the **Migratory Bird Species Quality Index (MBSQI)**, an end-to-end spatial machine learning pipeline designed to identify and model ecologically stable, environmentally favorable regions for migratory bird assemblages across the Andaman and Nicobar Islands (ANI).

The project operates as a hierarchical ecosystem intelligence system, transforming raw biodiversity observations into governed migratory evidence, extracting dynamic environmental features, training spatially-validated XGBoost models, and generating uncertainty-aware hotspot intelligence surfaces.

---

## 🚀 Project Highlights

* 
**Massive-Scale ETL:** Processed over 1.47 million multi-source biodiversity records (GBIF, eBird, iNaturalist) with a 99.51% precision-first attrition rate to isolate high-fidelity migratory evidence.


* 
**Advanced Feature Engineering:** Extracted and aligned dynamic vegetation indices (MODIS NDVI) and static terrain/proximity descriptors (SRTM DEM, Coastline distance) into an analysis-ready spatial matrix.


* 
**Spatial Machine Learning:** Utilized an adaptive, heavily regularized XGBoost classifier featuring monotonic ecological constraints to prevent overfitting and enforce biological realism.


* 
**Leakage-Aware Validation:** Implemented Spatial `GroupKFold` cross-validation (using KMeans clustering on coordinates) to ensure models learned true ecological relationships rather than simply memorizing geographic spatial autocorrelation.


* 
**Ecosystem-Scale Synthesis:** Generated final conservation intelligence products, mapping a 1,581 km² universal hotspot area based on high species richness, strong ecological agreement, and low extrapolation burden.



---

## 🛠️ Tech Stack & Architecture

* 
**Programming & Core ML:** `Python 3.9+`, `XGBoost 1.5+`, `Scikit-learn 1.0+` 


* 
**Geospatial Processing:** `QGIS`, `GDAL 3.x`, `Rasterio 1.2+`, `GeoPandas 0.10+` 


* 
**Data Engineering:** `NumPy`, `Pandas` (Parquet integration) 



### Pipeline Architecture

1. 
**Phase 1: Ecological Governance (ETL):** Multi-source schema harmonization, taxonomic normalization, exact duplicate purging, and observer-bias density capping.


2. 
**Phase 2 & 3: Environmental Intelligence:** Satellite data mosaicking, cloud masking via Bitwise QA logic, and dynamic feature extraction (e.g., $NDVI = \frac{\rho_{NIR} - \rho_{Red}}{\rho_{NIR} + \rho_{Red}}$). Generated observer-bias-aware Target-Group Background (TGB) pseudo-absences.


3. 
**Phase 4: Spatially-Aware Predictive Modeling:** Trained species-specific XGBoost models with sigmoid calibration and SHAP-based feature importance extraction.


4. 
**Phase 5 & 6: Ecosystem Synthesis:** Computed continuous GeoTIFF rasters for Mean Suitability, Standard Deviation (Uncertainty), and Extrapolation Novelty to extract universal conservation hotspots.



---

## 📊 Key Results & Metrics

* 
**Predictive Performance:** The overall framework achieved a mean ROC-AUC of 0.729 across 37 governed species. Flagship species (e.g., *Arenaria interpres*) achieved highly discriminative ROC-AUC scores of 0.877.


* 
**Model Calibration:** Achieved a highly reliable Brier Score of 0.1578, confirming that raw model scores translate into well-calibrated, real-world probability estimates.


* 
**Ecological Drivers (SHAP):** The model correctly identified distance to mangrove ecosystems (20.7%), dewpoint temperature (19.3%), and climate productivity (17.5%) as the most dominant predictors for coastal migratory taxa.







## 👥 Authors

* 
**Abhishek Gupta** 


* 
**Institution:** International Institute of Information Technology, Hyderabad (IIIT-H) 


* 
**Faculty Advisor:** Dr. Rama Chandra Prasad Pillutla 



