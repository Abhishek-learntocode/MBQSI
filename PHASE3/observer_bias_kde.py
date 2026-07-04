import os
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
from osgeo import gdal
import json
import time
from datetime import datetime

# ---------------- CONFIG ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OCCURRENCE_PATH = os.path.join(BASE_DIR, 'PHASE1', 'EXTRACTED_ANI_BIRDS', 'ANI_ONLY_occurrence.csv')
THINNED_CSV = os.path.join(BASE_DIR, 'PHASE1', 'CLEANED_DATA', 'MIGRATORY_BIRDS_THINNED_FINAL.csv')
GRID_PATH = os.path.join(BASE_DIR, 'PHASE2', 'metadata', 'reference_grid.json')
# Using NDVI valid_count as canonical landmask (bypassing incomplete GPKG)
LANDMASK_PATH = os.path.join(BASE_DIR, 'PHASE2', 'yearly_aggregates', 'NDVI_valid_count_2024.tif')

OUTPUT_DIR = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.1', 'outputs')
VAL_DIR = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.1', 'validation')

BANDWIDTH_KM = 10.0  # 10km search radius for accessibility hotspots
NODATA_VAL = -9999
NICOBAR_THRESHOLD = 10.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

def log_message(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def run_kde():
    log_message("Starting Phase 3.1b: Observer Bias KDE Generation (Raster-Masked)")
    
    # 1. Load Reference Grid & Landmask
    log_message(f"Loading reference grid and NDVI landmask...")
    with open(GRID_PATH, 'r') as f:
        grid = json.load(f)
    
    mask_ds = gdal.Open(LANDMASK_PATH)
    mask = mask_ds.GetRasterBand(1).ReadAsArray()
    mask_gt = mask_ds.GetGeoTransform()
    mask_nodata = mask_ds.GetRasterBand(1).GetNoDataValue()
    mask_ds = None
    
    # Define valid land: pixels where valid_count is not NoData and > 0
    land_mask = (mask != mask_nodata) & (mask > 0)
    
    xs, ys = grid['xs'], grid['ys']
    gt = grid['geo']
    projection = grid['proj']
    
    # 2. Load and Filter Records
    log_message(f"Loading occurrence data from {OCCURRENCE_PATH}...")
    
    # We must exclude the target thinned migratory species to avoid circularity
    target_species = pd.read_csv(THINNED_CSV)['species_name'].unique()
    log_message(f"Loaded {len(target_species)} target species to exclude from effort surface.")
    
    # Efficient loading: only columns we need
    # Schema check: eBird uses 'decimalLatitude', 'decimalLongitude', 'scientificName'
    # Our thinned CSV uses 'latitude', 'longitude', 'species_name'
    # The raw ANI_ONLY_occurrence.csv uses 'latitude', 'longitude', 'species_name' (post-Phase1)
    
    # Schema: GBIF raw uses 'decimalLatitude', 'decimalLongitude', 'scientificName'
    try:
        df = pd.read_csv(OCCURRENCE_PATH, 
                         usecols=['decimalLatitude', 'decimalLongitude', 'scientificName', 'year'],
                         dtype={'decimalLatitude': float, 'decimalLongitude': float, 'scientificName': str, 'year': 'float32'})
        df = df.rename(columns={
            'decimalLatitude': 'latitude',
            'decimalLongitude': 'longitude',
            'scientificName': 'species_name'
        })
    except Exception as e:
        log_message(f"CSV Loading warning: {e}. Attempting fallback...")
        df = pd.read_csv(OCCURRENCE_PATH, usecols=['decimalLatitude', 'decimalLongitude', 'scientificName', 'year'])
        df = df.rename(columns={
            'decimalLatitude': 'latitude',
            'decimalLongitude': 'longitude',
            'scientificName': 'species_name'
        })
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    log_message(f"Loaded {len(df)} total records.")
    
    # Remove target records
    df_effort = df[~df['species_name'].isin(target_species)].copy()
    n_target_removed = len(df) - len(df_effort)
    
    # Remove NaNs
    df_effort = df_effort.dropna(subset=['latitude', 'longitude'])
    n_invalid = len(df) - n_target_removed - len(df_effort)
    
    log_message(f"Filtered to {len(df_effort)} valid effort records.")
    log_message(f"  - Target records removed: {n_target_removed}")
    log_message(f"  - Invalid/corrupt coordinates removed: {n_invalid}")

    # 3. Spatial Alignment
    log_message("Aligning effort records to reference grid...")
    
    # Map lat/lon to pixel indices
    # col = (lon - x_min) / res
    # row = (lat_max - lat) / res
    px_x = ((df_effort['longitude'] - gt[0]) / gt[1]).astype(int)
    px_y = ((df_effort['latitude'] - gt[3]) / gt[5]).astype(int)
    
    # Boundary check
    valid_px = (px_x >= 0) & (px_x < xs) & (px_y >= 0) & (px_y < ys)
    outside_pct = (1 - valid_px.mean()) * 100
    log_message(f"Records outside reference grid: {outside_pct:.4f}%")
    
    df_effort = df_effort[valid_px]
    px_x = px_x[valid_px]
    px_y = px_y[valid_px]
    
    log_message(f"Final records used for KDE: {len(df_effort)}")

    # 4. Fast KDE (Histogram + Gaussian Filter)
    log_message(f"Binning occurrences into {xs}x{ys} grid...")
    hist, _, _ = np.histogram2d(px_y, px_x, bins=[ys, xs], range=[[0, ys], [0, xs]])
    
    log_message(f"Applying Gaussian smoothing (BW={BANDWIDTH_KM}km)...")
    pixel_res_deg = gt[1]
    # Approx 111km per degree
    sigma_pixels = (BANDWIDTH_KM / 111.0) / pixel_res_deg
    
    kde_smooth = gaussian_filter(hist, sigma=sigma_pixels)
    
    # 6. Normalization & Masking
    log_message("Applying NDVI landmask and normalizing...")
    # Mask non-land pixels first to prevent them from influencing stats
    # land_mask is already defined at start of function
    kde_smooth[~land_mask] = 0
    
    # Run a second small smoothing pass to refine edges without crossing water
    kde_smooth = gaussian_filter(kde_smooth, sigma=1.0)
    kde_smooth[~land_mask] = 0 # Mask again to be absolute
    
    # 99th Percentile Normalization for stability
    land_weights = kde_smooth[land_mask]
    if len(land_weights) > 0:
        z_ref = np.percentile(land_weights, 99)
        kde_norm = np.clip(kde_smooth / z_ref, 0, 1)
    else:
        kde_norm = kde_smooth
    Z_final = np.full_like(kde_norm, NODATA_VAL)
    Z_final[land_mask] = kde_norm[land_mask]
    
    # 6. Auditing & Validation
    log_message("Generating Scientific Audit...")
    
    # Create latitudinal mask for stratification
    lats_grid = gt[3] + np.arange(ys)[:, np.newaxis] * gt[5]
    
    # Create latitudinal mask for stratification
    is_nicobar = lats_grid <= NICOBAR_THRESHOLD
    is_andaman = lats_grid > NICOBAR_THRESHOLD
    effort_mask = (Z_final > 0) & (Z_final != NODATA_VAL)
    
    audit_metrics = {
        'metric': [
            'n_total_records', 'n_target_removed', 'n_invalid_coords', 'n_final_effort_records',
            'raw_max_percentile', 'entropy_shannon', 'hotspot_density_pct_top1', 'bandwidth_km',
            'sigma_pixels', 'raster_resolution_deg', 'nonzero_pixel_pct', 'land_coverage_pct',
            'andaman_nonzero_pixels', 'nicobar_nonzero_pixels', 'nicobar_effort_sum'
        ],
        'value': [
            len(df), n_target_removed, n_invalid, len(df_effort),
            float(z_ref), float(-np.sum(Z_final[effort_mask] * np.log(Z_final[effort_mask] + 1e-12))),
            float(((Z_final > 0.9) & effort_mask).sum() / effort_mask.sum() * 100), BANDWIDTH_KM,
            float(sigma_pixels), float(pixel_res_deg), 
            float(effort_mask.sum() / (xs * ys) * 100), float(land_mask.mean() * 100),
            float((effort_mask & is_andaman).sum()),
            float((effort_mask & is_nicobar).sum()),
            float(Z_final[effort_mask & is_nicobar].sum())
        ]
    }
    pd.DataFrame(audit_metrics).to_csv(os.path.join(VAL_DIR, 'observer_kde_validation.csv'), index=False)
    
    # 7. Export Raster
    out_path = os.path.join(OUTPUT_DIR, 'observer_effort_kde.tif')
    log_message(f"Exporting KDE surface to {out_path}...")
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(out_path, xs, ys, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(projection)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(Z_final)
    out_band.SetNoDataValue(NODATA_VAL)
    out_ds.FlushCache()
    out_ds = None
    
    log_message("KDE Generation Complete.")

if __name__ == "__main__":
    start_time = time.time()
    try:
        run_kde()
        log_message(f"Runtime: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        log_message(f"CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
