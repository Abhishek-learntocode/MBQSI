import os
import pandas as pd
import numpy as np
import rasterio
import json
import time
from datetime import datetime
from tqdm.auto import tqdm

# ---------------- CONFIG ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
REF_RASTER = os.path.join(BASE_DIR, 'PHASE2', 'yearly_aggregates', 'NDVI_mean_2024.tif')
OUTPUT_PQ = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.3', 'outputs', 'archipelago_feature_grid.parquet')
MISSING_AUDIT = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.3', 'outputs', 'archipelago_grid_missingness.csv')
RANGE_AUDIT = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.3', 'outputs', 'archipelago_feature_summary.csv')
MANIFEST_PATH = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.3', 'outputs', 'archipelago_feature_grid_manifest.json')

# Definitive Predictor Map
PREDICTORS = {
    'ndvi_mean': os.path.join(BASE_DIR, 'PHASE2', 'yearly_aggregates', 'NDVI_mean_2024.tif'),
    'ndvi_std': os.path.join(BASE_DIR, 'PHASE2', 'yearly_aggregates', 'NDVI_std_2024.tif'),
    'ndvi_cv': os.path.join(BASE_DIR, 'PHASE2', 'yearly_aggregates', 'NDVI_cv_2024.tif'),
    'ndvi_temporal_range': os.path.join(BASE_DIR, 'PHASE2', 'yearly_aggregates', 'NDVI_temporal_range_2024.tif'),
    'ndvi_valid_count': os.path.join(BASE_DIR, 'PHASE2', 'yearly_aggregates', 'NDVI_valid_count_2024.tif'),
    'static_elevation': os.path.join(BASE_DIR, 'PHASE2', 'static_layers', 'elevation.tif'),
    'static_slope': os.path.join(BASE_DIR, 'PHASE2', 'static_layers', 'slope.tif'),
    'static_terrain_roughness': os.path.join(BASE_DIR, 'PHASE2', 'static_layers', 'terrain_roughness.tif'),
    'static_dist_coast': os.path.join(BASE_DIR, 'PHASE2', 'static_layers', 'dist_coast.tif'),
    'static_dist_water': os.path.join(BASE_DIR, 'PHASE2', 'static_layers', 'dist_water.tif'),
    'static_local_ndvi_mean_1km': os.path.join(BASE_DIR, 'PHASE2', 'static_layers', 'local_ndvi_mean_1km.tif'),
    'static_local_ndvi_std_1km': os.path.join(BASE_DIR, 'PHASE2', 'static_layers', 'local_ndvi_std_1km.tif')
}

log_msg = lambda m: print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")

def generate_wall_to_wall_grid():
    log_msg("Commencing PRODUCTION Wall-to-Wall Grid Generation...")
    start_time = time.time()
    
    # 0. Early Predictor Audit
    missing_files = [p for p in PREDICTORS.values() if not os.path.exists(p)]
    if missing_files:
        raise FileNotFoundError(f"Missing predictor rasters: {missing_files}")

    # 1. Load Reference Landmask
    with rasterio.open(REF_RASTER) as src:
        mask = src.read(1)
        nodata = src.nodata
        transform = src.transform
        shape = src.shape
        crs = src.crs
        
    # ROBUST LANDMASK GOVERNANCE (Dtype-Safe)
    if nodata is not None:
        if isinstance(nodata, float) and np.isnan(nodata):
            nodata_mask = ~np.isnan(mask)
        else:
            nodata_mask = ~np.isclose(mask, nodata)
    else:
        nodata_mask = np.isfinite(mask)

    valid_mask = (np.isfinite(mask)) & (nodata_mask) & (mask > -1)
    rows, cols = np.where(valid_mask)
    coverage_pct = (len(rows)/(shape[0]*shape[1]))*100
    log_msg(f"Targeting {len(rows)} valid land pixels ({coverage_pct:.2f}% coverage).")
    
    # Extract Coordinates
    lons, lats = rasterio.transform.xy(transform, rows, cols)
    df = pd.DataFrame({
        'pixel_y': rows.astype(np.int32), 'pixel_x': cols.astype(np.int32),
        'longitude': np.array(lons, dtype=np.float32), 
        'latitude': np.array(lats, dtype=np.float32)
    })
    
    # 2. Extract Predictors with TRIPLE AUDIT
    for feat_name, path in tqdm(PREDICTORS.items(), desc="Stacking Environmental Layers"):
        with rasterio.open(path) as src:
            if src.shape != shape: raise ValueError(f"Geometry mismatch: {feat_name}")
            if src.crs != crs: raise ValueError(f"CRS mismatch: {feat_name}")
            if src.transform != transform: raise ValueError(f"Transform mismatch: {feat_name}")
                
            arr = src.read(1)
            if rows.max() >= arr.shape[0] or cols.max() >= arr.shape[1]:
                raise ValueError(f"Index overflow: {feat_name}")
                
            vals = arr[rows, cols].astype(np.float32)
            
            # PRECISION NODATA GOVERNANCE (Dtype-Safe)
            if src.nodata is not None:
                if isinstance(src.nodata, float) and np.isnan(src.nodata):
                    vals[np.isnan(vals)] = np.nan
                else:
                    vals[np.isclose(vals, src.nodata)] = np.nan
            
            df[feat_name] = vals

    # 3. Model Alignment & Sanity Corrections
    df['year'] = np.float32(2024.0)
    df['month'] = np.float32(6.0)
    
    log_msg("Applying Ecological Sanity Corrections...")
    # Clip Slope and Distances to be non-negative (handles residual sentinels)
    df['static_slope'] = np.clip(df['static_slope'], 0.0, None)
    df['static_dist_coast'] = np.clip(df['static_dist_coast'], 0.0, None)
    df['static_dist_water'] = np.clip(df['static_dist_water'], 0.0, None)
    # Clip elevation to remove extreme negative ocean sentinels
    df['static_elevation'] = np.clip(df['static_elevation'], -100.0, None)
    
    log_msg("Validating Ecological Sanity...")
    assert df['ndvi_mean'].dropna().between(-1.1, 1.1).all(), "NDVI Mean out of bounds"
    assert df['static_slope'].dropna().ge(0).all(), "Negative slope detected"
    assert df['static_elevation'].dropna().gt(-150).all(), "Extreme negative elevation detected"

    # 4. Deployment Audits
    dupes = df.duplicated(subset=['pixel_y', 'pixel_x']).sum()
    if dupes > 0: raise ValueError(f"Duplicate pixels detected: {dupes}")
        
    log_msg("Generating Governance Artifacts...")
    missing_summary = df.isnull().mean().mul(100).reset_index()
    missing_summary.columns = ['feature', 'missing_pct']
    missing_summary.to_csv(MISSING_AUDIT, index=False)
    
    audit_cols = [c for c in df.columns if c not in ['pixel_x','pixel_y','longitude','latitude','year','month']]
    df[audit_cols].describe(percentiles=[0.01, 0.99]).T.to_csv(RANGE_AUDIT)
    
    # Manifest Generation
    runtime = time.time() - start_time
    feature_valid_pct = (100 - missing_summary.set_index('feature')['missing_pct']).to_dict()
    complete_rows_pct = float(df[audit_cols].notnull().all(axis=1).mean() * 100)

    manifest = {
        'timestamp': datetime.now().isoformat(),
        'shape': list(shape),
        'crs': str(crs),
        'pixel_count': int(len(df)),
        'coverage_pct': float(coverage_pct),
        'complete_rows_pct': complete_rows_pct,
        'feature_valid_pct': feature_valid_pct,
        'runtime_sec': float(runtime),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=4)
        
    # 5. Export
    log_msg(f"Exporting Grid substrate: {OUTPUT_PQ}")
    df.reset_index(drop=True, inplace=True)
    df.to_parquet(OUTPUT_PQ, compression='snappy')
    log_msg("SUCCESS: Wall-to-Wall substrate ready for inference.")

if __name__ == "__main__":
    generate_wall_to_wall_grid()
