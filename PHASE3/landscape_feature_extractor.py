import os
import pandas as pd
import numpy as np
from osgeo import gdal, osr
from datetime import datetime
import json
from scipy.stats import ks_2samp

# ---------------- CONFIG ----------------
gdal.UseExceptions()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DYNAMIC_MATRIX = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.2', 'outputs', 'dynamic_feature_matrix.csv')
STATIC_ROOT = os.path.join(BASE_DIR, 'PHASE2', 'static_layers') # Assuming Phase 2 static outputs
GRID_METADATA = os.path.join(BASE_DIR, 'PHASE2', 'metadata', 'reference_grid.json')

OUTPUT_DIR = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.3', 'outputs')
VAL_DIR = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.3', 'validation')

# Static Features to Extract
TERRAIN_FEATURES = ['elevation', 'slope', 'terrain_roughness']
PROXIMITY_FEATURES = ['dist_coast', 'dist_water']
CONTEXT_FEATURES = ['local_ndvi_mean_1km', 'local_ndvi_std_1km']
ALL_STATIC = TERRAIN_FEATURES + PROXIMITY_FEATURES + CONTEXT_FEATURES

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

def log_message(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def get_pixel_coords(lon, lat, gt):
    px = int((lon - gt[0]) / gt[1])
    py = int((gt[3] - lat) / abs(gt[5]))
    return px, py

def get_terrain_confidence(elevation, dist_coast):
    """Categorize terrain reliability."""
    if elevation < 0 or dist_coast < 0.1: return 'LOW' # Possible ocean/coast bleed
    if elevation > 500: return 'MEDIUM' # High altitude steep slope risk
    return 'HIGH'

def extract_landscape():
    start_time = datetime.now()
    log_message("Starting Phase 3.3: Landscape & Context Feature Extraction")
    
    # 1. Load Reference Grid & Matrix
    with open(GRID_METADATA, 'r') as f:
        grid = json.load(f)
    gt_ref, wkt_ref, ref_shape = tuple(grid['geo']), grid['proj'], (grid['ys'], grid['xs'])
    ref_srs = osr.SpatialReference(wkt=wkt_ref)
    
    df = pd.read_csv(DYNAMIC_MATRIX)
    log_message(f"Input records: {len(df)}")

    # 2. Open Static Handles (Lazy Access)
    handles = {}
    missing_layers = []
    for feat in ALL_STATIC:
        # Note: In Phase 2 we assume these were generated as single-band TIFs
        f_path = os.path.join(STATIC_ROOT, f"{feat}.tif")
        if os.path.exists(f_path):
            ds = gdal.Open(f_path)
            # Alignment check (Simplified for static)
            if ds.RasterXSize == ref_shape[1] and ds.RasterYSize == ref_shape[0]:
                handles[feat] = ds
            else:
                log_message(f"Alignment Error: {feat}.tif does not match reference grid.")
                missing_layers.append(feat)
        else:
            missing_layers.append(feat)

    if missing_layers:
        log_message(f"Warning: Missing layers {missing_layers}. These will be NaN.")

    # 3. Extraction Loop
    results = []
    runtime_errors = []
    failure_counts = {'NODATA': 0, 'OUT_OF_BOUNDS': 0, 'INVALID_RANGE': 0}

    for _, row in df.iterrows():
        px, py = get_pixel_coords(row['longitude'], row['latitude'], gt_ref)
        
        if not (0 <= px < ref_shape[1] and 0 <= py < ref_shape[0]):
            failure_counts['OUT_OF_BOUNDS'] += 1
            continue
            
        ext_row = row.to_dict()
        valid_count = 0
        
        for feat in ALL_STATIC:
            if feat not in handles:
                ext_row[f'static_{feat}'] = np.nan
                continue
                
            try:
                ds = handles[feat]
                band = ds.GetRasterBand(1)
                nodata = band.GetNoDataValue()
                # 1x1 Lazy Read
                val_arr = band.ReadAsArray(px, py, 1, 1)
                val = float(val_arr[0,0])
                
                if val == nodata:
                    ext_row[f'static_{feat}'] = np.nan
                    failure_counts['NODATA'] += 1
                else:
                    ext_row[f'static_{feat}'] = val
                    valid_count += 1
            except Exception as e:
                runtime_errors.append({'id': row['original_row_id'], 'feat': feat, 'error': str(e)})
                ext_row[f'static_{feat}'] = np.nan

        # Derived Contextual Features
        ext_row['coastal_zone_flag'] = 1 if ext_row.get('static_dist_coast', 99) < 5 else 0
        ext_row['terrain_confidence_flag'] = get_terrain_confidence(
            ext_row.get('static_elevation', 0), 
            ext_row.get('static_dist_coast', 1)
        )
        
        results.append(ext_row)

    # 4. Save Intermediate Landscape Matrix
    land_df = pd.DataFrame(results)
    land_df.to_csv(os.path.join(OUTPUT_DIR, 'landscape_feature_matrix.csv'), index=False)
    
    # 5. Operational Audit
    pd.DataFrame(runtime_errors).to_csv(os.path.join(VAL_DIR, 'landscape_runtime_errors.csv'), index=False)
    
    log_message(f"Landscape Extraction Complete. Intermediate matrix saved: {len(land_df)} records.")

if __name__ == "__main__":
    extract_landscape()
