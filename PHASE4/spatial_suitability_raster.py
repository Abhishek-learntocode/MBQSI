import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import time
import rasterio
import gc
from datetime import datetime
from tqdm.auto import tqdm

# ---------------- CONFIG ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
REF_RASTER = os.path.join(BASE_DIR, 'PHASE2', 'yearly_aggregates', 'NDVI_mean_2024.tif')
INFERENCE_MATRIX_PQ = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.3', 'outputs', 'archipelago_feature_grid.parquet')
MODELS_DIR = os.path.join(BASE_DIR, 'PHASE4', 'models')
CONFIG_PATH = os.path.join(BASE_DIR, 'PHASE4', 'outputs', 'phase4_config.json')
SUMMARY_PATH = os.path.join(BASE_DIR, 'PHASE4', 'validation', 'model_performance_summary.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'PHASE4', 'rasters')
VAL_DIR = os.path.join(BASE_DIR, 'PHASE4', 'validation')
FAILED_LOG = os.path.join(VAL_DIR, "failed_species.log")

for d in [OUTPUT_DIR, VAL_DIR]: os.makedirs(d, exist_ok=True)

# ---------------- PRE-LOAD SHARED DATA ----------------
log_msg = lambda m: tqdm.write(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")

def load_governed_environment():
    log_msg("Loading Production Environment...")
    with open(CONFIG_PATH, 'r') as f: config = json.load(f)
    summary = pd.read_csv(SUMMARY_PATH)
    pass_species = summary[summary['governance'] == 'PASS']['species'].tolist()
    
    log_msg("Caching Dense Global Inference Grid (Parquet)...")
    df_grid = pd.read_parquet(INFERENCE_MATRIX_PQ)
    
    # CRITICAL DUPLICATE GOVERNANCE
    if df_grid.duplicated(subset=['pixel_y', 'pixel_x']).any():
        raise ValueError("CRITICAL: Duplicate deployment pixels detected in inference substrate.")
        
    # Pre-cache Models with Stability Audit
    log_msg("Pre-caching Model Ensembles...")
    model_cache = {}
    for s in pass_species:
        safe_name = s.replace(' ', '_').replace('/', '_')
        models = [joblib.load(os.path.join(MODELS_DIR, f"{safe_name}_fold{f}.joblib")) for f in range(4) if os.path.exists(os.path.join(MODELS_DIR, f"{safe_name}_fold{f}.joblib"))]
        if len(models) >= 3:
            if len(models) < 4:
                log_msg(f"WARNING: {s} has only {len(models)} folds loaded. Stability may be reduced.")
            model_cache[s] = models
        else:
            log_msg(f"SKIP: {s} (Insufficient folds < 3)")
        
    # Pre-calculate Medians (Excluding Roughness)
    log_msg("Calculating Ecological Medians for imputation...")
    FEATURES_TO_USE = [f for f in config['features'] if f != 'static_terrain_roughness']
    existing_features = [f for f in FEATURES_TO_USE if f in df_grid.columns]
    FEATURE_MEDIANS = df_grid[existing_features].median().to_dict()
    FEATURE_MEDIANS['ndvi_valid_fraction'] = 0.5
    
    return config, pass_species, df_grid, model_cache, FEATURE_MEDIANS, FEATURES_TO_USE

CONFIG, PASS_LIST, DF_GRID, MODEL_CACHE, MEDIANS, FEATURES = load_governed_environment()

def prepare_features(chunk):
    # Safe Feature Parity Audit
    HANDLED_DERIVED = ['ndvi_valid_fraction', 'is_lowland', 'is_highland']
    missing_features = [f for f in FEATURES if f not in HANDLED_DERIVED and f not in chunk.columns]
    if missing_features:
        raise ValueError(f"CRITICAL: Missing features: {missing_features}")

    X = pd.DataFrame()
    for f in FEATURES:
        if f == 'ndvi_valid_fraction': X[f] = chunk['ndvi_valid_count'] / 23.0
        elif f == 'is_lowland': X[f] = (chunk['static_elevation'] < 50.0).astype(float)
        elif f == 'is_highland': X[f] = (chunk['static_elevation'] > 150.0).astype(float)
        else: X[f] = chunk[f]
    
    chunk_missing = X.isnull().sum()
    imputed_count = chunk_missing.sum()
    if imputed_count > 0: X = X.fillna(MEDIANS)
        
    return X.astype(np.float32), imputed_count, chunk_missing

def run_spatial_inference(species_name):
    if species_name not in MODEL_CACHE:
        log_msg(f"SKIP: {species_name} (No Ensemble)")
        return

    safe_name = species_name.replace(' ', '_').replace('/', '_')
    start_time = time.time()
    with rasterio.open(REF_RASTER) as src:
        profile = src.profile.copy()
        full_shape = src.shape
        nodata = src.nodata

    mins = pd.Series(CONFIG['feature_mins'])[FEATURES] if 'feature_mins' in CONFIG else None
    maxs = pd.Series(CONFIG['feature_maxs'])[FEATURES] if 'feature_maxs' in CONFIG else None

    suit_arr = np.full(full_shape, nodata, dtype=np.float32)
    std_arr = np.full(full_shape, nodata, dtype=np.float32)
    cv_arr = np.full(full_shape, nodata, dtype=np.float32)
    extrap_arr = np.full(full_shape, nodata, dtype=np.float32)

    models = MODEL_CACHE[species_name]
    total_imputed = 0
    feature_imputed_totals = pd.Series(0.0, index=FEATURES)
    chunk_size = 75000
    for i in tqdm(range(0, len(DF_GRID), chunk_size), desc=f"Mapping {species_name}", leave=False):
        chunk = DF_GRID.iloc[i : i+chunk_size]
        X, chunk_imputed, chunk_missing = prepare_features(chunk)
        total_imputed += chunk_imputed
        feature_imputed_totals += chunk_missing
            
        fold_probs = np.stack([m.predict_proba(X)[:, 1] for m in models])
        mean_prob = np.clip(np.mean(fold_probs, axis=0), 0.001, 0.999)
        std_prob = np.clip(np.std(fold_probs, axis=0), 0.0, 1.0)
        
        # Numerically Clean CV Governance
        cv_prob = np.divide(std_prob, mean_prob, out=np.zeros_like(std_prob), where=mean_prob > 0.05).astype(np.float32)
        cv_prob = np.clip(cv_prob, 0.0, 5.0)
        
        # Extrapolation
        if mins is not None and maxs is not None:
            outside = ((X < mins) | (X > maxs))
            extrap_score = outside.mean(axis=1).astype(np.float32).values
        else:
            extrap_score = np.zeros(len(X), dtype=np.float32)
        
        rows, cols = chunk['pixel_y'].values.astype(int), chunk['pixel_x'].values.astype(int)
        mask = (rows >= 0) & (cols >= 0) & (rows < full_shape[0]) & (cols < full_shape[1])
        suit_arr[rows[mask], cols[mask]] = mean_prob[mask]
        std_arr[rows[mask], cols[mask]] = std_prob[mask]
        cv_arr[rows[mask], cols[mask]] = cv_prob[mask]
        extrap_arr[rows[mask], cols[mask]] = extrap_score[mask]

    # Optimized GeoTIFF Compression (LZW + Predictor=2 for floats)
    profile.update(dtype=rasterio.float32, count=1, compress='lzw', predictor=2, nodata=nodata)
    outputs = {'suitability': suit_arr, 'uncertainty_std': std_arr, 'uncertainty_cv': cv_arr, 'extrapolation': extrap_arr}
    
    total_cells = len(DF_GRID) * len(FEATURES)
    runtime = time.time() - start_time
    
    audit_results = []
    for suffix, arr in outputs.items():
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}_{suffix}.tif")
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(arr, 1)
        
        # Numerically Robust Nodata Audit
        if nodata is None: valid_mask = np.isfinite(arr)
        elif isinstance(nodata, float) and np.isnan(nodata): valid_mask = np.isfinite(arr)
        else: valid_mask = ~np.isclose(arr, nodata)
        valid_data = arr[valid_mask]
        
        audit_results.append({
            'species': species_name, 'layer': suffix,
            'min': np.min(valid_data) if len(valid_data)>0 else 0,
            'max': np.max(valid_data) if len(valid_data)>0 else 0,
            'coverage_pct': (len(valid_data) / (full_shape[0]*full_shape[1])) * 100 if len(valid_data)>0 else 0,
            'imputation_pct': (total_imputed / total_cells) * 100 if suffix == 'suitability' else 0,
            'runtime_sec': runtime if suffix == 'suitability' else 0,
            'pixels_per_sec': (len(valid_data) / runtime if suffix == 'suitability' and runtime > 0 else 0)
        })

    pd.DataFrame(audit_results).to_csv(os.path.join(VAL_DIR, f"{safe_name}_raster_audit.csv"), index=False)
    feature_imputed_totals.to_csv(os.path.join(VAL_DIR, f"{safe_name}_feature_imputation_audit.csv"))
    log_msg(f"PROJECTION SUCCESS: {species_name}")

if __name__ == "__main__":
    FORCE_OVERWRITE = True # Enforce clean re-run
    log_msg(f"Commencing Full Scale-up for {len(PASS_LIST)} species...")
    for s in tqdm(PASS_LIST, desc="Full Species Deployment"):
        try: 
            safe_name = s.replace(' ', '_').replace('/', '_')
            if not FORCE_OVERWRITE and os.path.exists(os.path.join(OUTPUT_DIR, f"{safe_name}_suitability.tif")): 
                continue
            run_spatial_inference(s)
            gc.collect() 
        except Exception as e: 
            log_msg(f"ERROR: {s} -> {e}")
            with open(FAILED_LOG, 'a') as f: f.write(f"{datetime.now().isoformat()} | {s} | {str(e)}\n")
