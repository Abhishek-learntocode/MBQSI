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
PRESENCE_CSV = os.path.join(BASE_DIR, 'PHASE1', 'CLEANED_DATA', 'MIGRATORY_BIRDS_THINNED_FINAL.csv')
BACKGROUND_CSV = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.1', 'outputs', 'background_pool.csv')
NDVI_ROOT = os.path.join(BASE_DIR, 'PHASE2', 'yearly_aggregates')
PHASE2_VAL = os.path.join(BASE_DIR, 'PHASE2', 'validation', 'csv', 'aggregate_validation.csv')
GRID_METADATA = os.path.join(BASE_DIR, 'PHASE2', 'metadata', 'reference_grid.json')

OUTPUT_DIR = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.2', 'outputs')
VAL_DIR = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.2', 'validation')

# Parameters
MIN_RECORDS_PER_SPECIES = 25
ECO_FEATURES = ['mean', 'std', 'cv', 'temporal_range']
CONF_FEATURES = ['valid_count']
ALL_FEATURES = ECO_FEATURES + CONF_FEATURES
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

def log_message(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def get_pixel_coords(lon, lat, gt):
    px = int((lon - gt[0]) / gt[1])
    py = int((gt[3] - lat) / abs(gt[5]))
    return px, py

def get_spatial_fold(lat):
    if lat >= 13.0: return 'North_Andaman'
    if lat >= 11.5: return 'Middle_Andaman'
    if lat >= 10.0: return 'South_Andaman'
    return 'Nicobar'

def validate_range(feat, val):
    if feat == 'mean': return 0 <= val <= 1
    if feat == 'std': return 0 <= val <= 1
    if feat == 'cv': return 0 <= val <= 5 
    if feat == 'temporal_range': return 0 <= val <= 1
    if feat == 'valid_count': return 0 <= val <= 23 # MODIS 16-day composites limit
    return True

def extract_features():
    start_time = datetime.now()
    log_message("Starting Phase 3.2: PRODUCTION-GOVERNANCE Dynamic Feature Extraction")
    
    # 1. Load Reference & Quality Gating
    with open(GRID_METADATA, 'r') as f:
        grid = json.load(f)
    gt_ref, wkt_ref, ref_shape = tuple(grid['geo']), grid['proj'], (grid['ys'], grid['xs'])
    
    ref_srs = osr.SpatialReference()
    ref_srs.ImportFromWkt(wkt_ref)
    
    # Quality Gate: Phase 2 aggregate health
    try:
        p2_val = pd.read_csv(PHASE2_VAL)
        # Assuming archipelago size ~150k pixels
        valid_years = p2_val[p2_val['valid_pixels'] > 75000]['year'].unique()
        log_message(f"Phase 2 Quality Gate: {len(valid_years)} years passed quality threshold.")
    except:
        log_message("Warning: Phase 2 validation not found. Proceeding with caution.")
        valid_years = np.arange(2005, 2026)

    # 2. Load Data & IDs
    presence_df = pd.read_csv(PRESENCE_CSV)
    presence_df['original_row_id'] = presence_df.index
    
    valid_species = presence_df['species_name'].value_counts()
    valid_species = valid_species[valid_species >= MIN_RECORDS_PER_SPECIES].index.tolist()
    
    p_filtered = presence_df[presence_df['species_name'].isin(valid_species)].copy()
    p_filtered['target'], p_filtered['observer_weight'] = 1, 1.0
    
    b_df = pd.read_csv(BACKGROUND_CSV)
    b_df['original_row_id'] = b_df.index
    b_df['species_name'], b_df['target'] = 'BACKGROUND', 0
    b_df = b_df.rename(columns={'weight': 'observer_weight'})
    
    full_df = pd.concat([
        p_filtered[['species_name', 'longitude', 'latitude', 'year', 'month', 'target', 'observer_weight', 'original_row_id']],
        b_df[['species_name', 'longitude', 'latitude', 'year', 'month', 'target', 'observer_weight', 'original_row_id']]
    ]).reset_index(drop=True)

    n_input_p = len(p_filtered)
    n_input_b = len(b_df)

    # 3. Extraction
    results = []
    failure_audit = []
    temporal_health = []
    feat_inventory = {f: [] for f in ALL_FEATURES}
    runtime_errors = []

    for year in sorted(full_df['year'].unique()):
        if year not in valid_years: continue
        
        y_pts = full_df[full_df['year'] == year].copy()
        handles = {}
        for feat in ALL_FEATURES:
            f_path = os.path.join(NDVI_ROOT, f"NDVI_{feat}_{int(year)}.tif")
            if os.path.exists(f_path):
                try:
                    ds = gdal.Open(f_path)
                    if np.allclose(ds.GetGeoTransform(), gt_ref, atol=1e-9) and \
                       osr.SpatialReference(wkt=ds.GetProjection()).IsSame(ref_srs):
                        handles[feat] = ds
                except: pass
        
        if len(handles) != len(ALL_FEATURES):
            log_message(f"Skipping {int(year)} - Missing Features")
            continue

        log_message(f"Processing Year {int(year)}: {len(y_pts)} points")
        y_retained = 0
        for _, pt in y_pts.iterrows():
            px, py = get_pixel_coords(pt['longitude'], pt['latitude'], gt_ref)
            
            # Explicit Boundary Safety
            if not (0 <= px < ref_shape[1] and 0 <= py < ref_shape[0]):
                failure_audit.append({'target': pt['target'], 'reason': 'OUT_OF_BOUNDS'})
                continue
                
            row = pt.to_dict()
            row.update({'pixel_x': px, 'pixel_y': py, 'spatial_fold': get_spatial_fold(pt['latitude'])})
            
            valid_eco = 0
            pt_reasons = []
            for feat in ALL_FEATURES:
                try:
                    ds = handles[feat]
                    band = ds.GetRasterBand(1)
                    val = float(band.ReadAsArray(px, py, 1, 1)[0,0])
                    
                    if val == band.GetNoDataValue(): pt_reasons.append('NODATA')
                    elif not np.isfinite(val): pt_reasons.append('NON_FINITE')
                    elif not validate_range(feat, val): pt_reasons.append('INVALID_RANGE')
                    else:
                        row[f'ndvi_{feat}'] = val
                        if feat in ECO_FEATURES: 
                            valid_eco += 1
                            feat_inventory[feat].append(val)
                except Exception as e:
                    pt_reasons.append('EXTRACTION_ERROR')
                    runtime_errors.append({'year': year, 'feature': feat, 'pixel': (px,py), 'error': str(e)})

            if valid_eco >= 3:
                results.append(row)
                y_retained += 1
            else:
                failure_audit.append({'target': pt['target'], 'reason': '|'.join(sorted(set(pt_reasons)))})
        
        temporal_health.append({'year': year, 'input': len(y_pts), 'retained': y_retained, 'retained_pct': (y_retained/len(y_pts))*100 if len(y_pts)>0 else 0})
        for h in handles.values(): h = None

    # 4. De-duplication & Final Matrix
    final_df = pd.DataFrame(results)
    if not final_df.empty:
        # Preserve multi-species sightings but collapse pixel hotspots
        final_df = final_df.sort_values('observer_weight', ascending=False)
        final_df = final_df.groupby(['pixel_x', 'pixel_y', 'year', 'target', 'species_name']).first().reset_index()
        
        max_v = final_df['ndvi_valid_count'].max()
        final_df['confidence_flag'] = pd.cut(final_df['ndvi_valid_count'], bins=[-1, 0.5*max_v, 0.8*max_v, max_v+1], labels=['LOW', 'MEDIUM', 'HIGH'])

    # 5. Save Results
    final_df.to_csv(os.path.join(OUTPUT_DIR, 'dynamic_feature_matrix.csv'), index=False)
    try: final_df.to_parquet(os.path.join(OUTPUT_DIR, 'dynamic_feature_matrix.parquet'), index=False)
    except: pass

    # 6. Audits (Extended)
    pd.DataFrame(temporal_health).to_csv(os.path.join(VAL_DIR, 'temporal_extraction_health.csv'), index=False)
    pd.DataFrame(failure_audit).to_csv(os.path.join(VAL_DIR, 'dynamic_failure_breakdown.csv'), index=False)
    pd.DataFrame(runtime_errors).to_csv(os.path.join(VAL_DIR, 'extraction_runtime_errors.csv'), index=False)
    
    # Class Retention Audit
    n_final_p = len(final_df[final_df['target'] == 1])
    n_final_b = len(final_df[final_df['target'] == 0])
    retention_audit = [
        {'class': 'Presence', 'before': n_input_p, 'after': n_final_p, 'retained_pct': (n_final_p/n_input_p)*100},
        {'class': 'Background', 'before': n_input_b, 'after': n_final_b, 'retained_pct': (n_final_b/n_input_b)*100}
    ]
    pd.DataFrame(retention_audit).to_csv(os.path.join(VAL_DIR, 'class_retention_audit.csv'), index=False)

    # Ecological Distribution Audit
    eco_summary = []
    for feat in [f'ndvi_{f}' for f in ECO_FEATURES]:
        p = final_df[final_df['target'] == 1][feat].dropna()
        b = final_df[final_df['target'] == 0][feat].dropna()
        if len(p) > 10 and len(b) > 10:
            ks, _ = ks_2samp(p, b)
            d = (p.mean() - b.mean()) / max(np.sqrt((p.std()**2 + b.std()**2)/2), 1e-8)
            eco_summary.append({
                'feature': feat, 'ks_stat': ks, 'cohens_d': d,
                'p_mean': p.mean(), 'b_mean': b.mean(), 'p_std': p.std(), 'b_std': b.std()
            })
    pd.DataFrame(eco_summary).to_csv(os.path.join(VAL_DIR, 'ecological_distribution_summary.csv'), index=False)
    
    inv_data = []
    for f, vals in feat_inventory.items():
        if vals:
            inv_data.append({'feature': f, 'mean': np.mean(vals), 'std': np.std(vals), 'p01': np.percentile(vals, 1), 'p99': np.percentile(vals, 99)})
    pd.DataFrame(inv_data).to_csv(os.path.join(VAL_DIR, 'feature_inventory_report.csv'), index=False)

    # 7. Inventory Narrative
    inventory = f"""PHASE 3.2 PRODUCTION LEDGER
==========================
Timestamp: {datetime.now().isoformat()}
Total Records: {len(final_df)}
Species Passing Gate: {final_df[final_df['target']==1]['species_name'].nunique()}
Class Balance: {final_df['target'].value_counts().to_dict()}
Spatial Folds: {final_df['spatial_fold'].value_counts().to_dict()}
Average Confidence: {final_df['ndvi_valid_count'].mean():.2f}
Runtime: {(datetime.now()-start_time).total_seconds():.1f}s
"""
    with open(os.path.join(OUTPUT_DIR, 'phase3_2_inventory_report.txt'), 'w') as f: f.write(inventory)
    
    # Save Config Manifest
    with open(os.path.join(OUTPUT_DIR, 'phase3_2_config.json'), 'w') as f:
        json.dump({'min_records': MIN_RECORDS_PER_SPECIES, 'features': ECO_FEATURES, 'folds': ['North', 'Middle', 'South', 'Nicobar'], 'seed': RANDOM_SEED}, f, indent=4)

    log_message("Phase 3.2 Production Extraction Complete (with Full Audits).")

if __name__ == "__main__":
    extract_features()
