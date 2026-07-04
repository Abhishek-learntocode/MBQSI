import os
import pandas as pd
import numpy as np
import geopandas as gpd
from osgeo import gdal
import json
from scipy.spatial import cKDTree
import time
from datetime import datetime
from shapely.geometry import Point
from scipy.ndimage import label, distance_transform_edt
from scipy.stats import ks_2samp

# ---------------- CONFIG ----------------
GLOBAL_RANDOM_SEED = 42
np.random.seed(GLOBAL_RANDOM_SEED)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
KDE_PATH = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.1', 'outputs', 'observer_effort_kde.tif')
PRESENCE_CSV = os.path.join(BASE_DIR, 'PHASE1', 'CLEANED_DATA', 'MIGRATORY_BIRDS_THINNED_FINAL.csv')
GRID_PATH = os.path.join(BASE_DIR, 'PHASE2', 'metadata', 'reference_grid.json')
# NDVI-based landmask for spatial domain consistency
LANDMASK_PATH = os.path.join(BASE_DIR, 'PHASE2', 'yearly_aggregates', 'NDVI_valid_count_2024.tif')

OUTPUT_DIR = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.1', 'outputs')
VAL_DIR = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.1', 'validation')

# Sampling Parameters
N_BACKGROUND_GOAL = 30000 
PRESENCE_BUFFER_M = 1000 
MIN_THINNING_DISTANCE_M = 500 # Reduced to 500m to capture fine-grained coastal transitions
STRATA_QUOTA = {'Andaman': 0.7, 'Nicobar': 0.3}
PROJ_CRS = "EPSG:32646" 
NICOBAR_LAT_THRESHOLD = 10.0 

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

def log_message(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def generate_background():
    log_message("Starting Phase 3.1c: Target-Group Background (TGB) Sampling (Research-Grade)")
    
    # 1. Load Data
    log_message(f"Loading KDE surface and Presence data...")
    ds = gdal.Open(KDE_PATH)
    kde_band = ds.GetRasterBand(1)
    kde_array = kde_band.ReadAsArray()
    nodata = kde_band.GetNoDataValue()
    gt = ds.GetGeoTransform()
    ds = None
    
    mask_ds = gdal.Open(LANDMASK_PATH)
    mask_band = mask_ds.GetRasterBand(1)
    mask_array = mask_band.ReadAsArray()
    mask_nodata = mask_band.GetNoDataValue()
    mask_ds = None
    
    presence_df = pd.read_csv(PRESENCE_CSV)
    
    # 2. Candidate Extraction & Connected Components Stratification
    log_message("Extracting non-zero weight candidates and identifying island clusters...")
    land_mask = (mask_array != mask_nodata) & (mask_array > 0)
    
    # Stratification via Connected Components
    labeled_mask, n_clusters = label(land_mask)
    y_idx, x_idx = np.where((kde_array != nodata) & (kde_array > 0) & land_mask)
    
    lons = gt[0] + (x_idx + 0.5) * gt[1] + (y_idx + 0.5) * gt[2]
    lats = gt[3] + (x_idx + 0.5) * gt[4] + (y_idx + 0.5) * gt[5]
    weights = kde_array[y_idx, x_idx]
    cluster_ids = labeled_mask[y_idx, x_idx]
    
    candidates = pd.DataFrame({
        'longitude': lons,
        'latitude': lats,
        'weight': weights,
        'cluster_id': cluster_ids,
        'pixel_x': x_idx,
        'pixel_y': y_idx
    })
    
    # Assign Stratum by Cluster Centroid
    cluster_lats = candidates.groupby('cluster_id')['latitude'].mean()
    cluster_strata = {cid: ('Andaman' if clat > NICOBAR_LAT_THRESHOLD else 'Nicobar') 
                      for cid, clat in cluster_lats.items()}
    candidates['stratum'] = candidates['cluster_id'].map(cluster_strata)
    
    n_initial = len(candidates)
    log_message(f"Initial candidates: {n_initial}")

    # 3. Metric-Space Presence Exclusion
    log_message(f"Applying Presence-Exclusion Buffer ({PRESENCE_BUFFER_M}m)...")
    
    presence_gdf = gpd.GeoDataFrame(
        presence_df, geometry=gpd.points_from_xy(presence_df.longitude, presence_df.latitude), crs="EPSG:4326"
    ).to_crs(PROJ_CRS)
    
    candidate_gdf_proj = gpd.GeoDataFrame(
        candidates, geometry=gpd.points_from_xy(candidates.longitude, candidates.latitude), crs="EPSG:4326"
    ).to_crs(PROJ_CRS)
    
    presence_coords = np.column_stack((presence_gdf.geometry.x, presence_gdf.geometry.y))
    presence_tree = cKDTree(presence_coords)
    candidate_coords_proj = np.column_stack((candidate_gdf_proj.geometry.x, candidate_gdf_proj.geometry.y))
    
    near_presence_idx = presence_tree.query_ball_point(candidate_coords_proj, PRESENCE_BUFFER_M)
    candidates = candidates[[len(idx) == 0 for idx in near_presence_idx]].copy()
    candidate_coords_proj = candidate_coords_proj[[len(idx) == 0 for idx in near_presence_idx]]
    
    n_after_exclusion = len(candidates)
    log_message(f"Candidates after exclusion: {n_after_exclusion}")

    # 4. Proportional Sampling (Stratified + Temperature Flattening)
    log_message("Executing Stratified Sampling with Temperature Flattening...")
    sampled_dfs = []
    for stratum, proportion in STRATA_QUOTA.items():
        stratum_df = candidates[candidates['stratum'] == stratum]
        stratum_goal = int(N_BACKGROUND_GOAL * proportion)
        if len(stratum_df) == 0: continue
        
        n_to_sample = min(len(stratum_df), stratum_goal)
        w = np.sqrt(stratum_df['weight'].values)
        w = np.maximum(w, 1e-12)
        
        idx_pool = np.arange(len(stratum_df))
        sampled_indices = np.random.choice(idx_pool, size=n_to_sample, replace=False, p=w / w.sum())
        sampled_dfs.append(stratum_df.iloc[sampled_indices])
        
    background_pool = pd.concat(sampled_dfs)
    log_message(f"Initial sample size: {len(background_pool)}")

    # 5. Neutral Spatial Thinning
    log_message(f"Applying {MIN_THINNING_DISTANCE_M}m Neutral Thinning...")
    background_pool = background_pool.sample(frac=1, random_state=42)
    pool_gdf_proj = gpd.GeoDataFrame(
        background_pool, geometry=gpd.points_from_xy(background_pool.longitude, background_pool.latitude), crs="EPSG:4326"
    ).to_crs(PROJ_CRS)
    pool_coords_proj = np.column_stack((pool_gdf_proj.geometry.x, pool_gdf_proj.geometry.y))
    
    tree = cKDTree(pool_coords_proj)
    removed_indices = set()
    final_indices = []
    indices = background_pool.index.tolist()
    for i, idx in enumerate(indices):
        if idx in removed_indices: continue
        final_indices.append(idx)
        neighbors = tree.query_ball_point(pool_coords_proj[i], MIN_THINNING_DISTANCE_M)
        for n_idx in neighbors: 
            if indices[n_idx] != idx:
                removed_indices.add(indices[n_idx])
            
    background_thinned = background_pool.loc[final_indices].copy()
    background_thinned = background_thinned.drop_duplicates(subset=['pixel_x', 'pixel_y'])
    n_final = len(background_thinned)
    log_message(f"Final pool size after thinning: {n_final}")

    # 6. NEAREST-PRESENCE SEASONAL BORROWING
    log_message("Applying Seasonal-Aware Temporal Borrowing (Inheriting Year & Month)...")
    final_gdf_proj = gpd.GeoDataFrame(
        background_thinned, geometry=gpd.points_from_xy(background_thinned.longitude, background_thinned.latitude), crs="EPSG:4326"
    ).to_crs(PROJ_CRS)
    final_coords_proj = np.column_stack((final_gdf_proj.geometry.x, final_gdf_proj.geometry.y))
    
    # Find 25 nearest presences
    _, near_p_indices = presence_tree.query(final_coords_proj, k=25)
    
    assigned_years = []
    assigned_months = []
    presence_years = presence_df['year'].values
    presence_months = presence_df['month'].values
    for neighbors in near_p_indices:
        # Pick one real observation neighbor and inherit its full timestamp
        selected_neighbor = np.random.choice(neighbors)
        assigned_years.append(presence_years[selected_neighbor])
        assigned_months.append(presence_months[selected_neighbor])
    
    background_thinned['year'] = assigned_years
    background_thinned['month'] = assigned_months
    background_thinned['target'] = 0

    # 7. Auditing & Validation
    log_message("Finalizing Audits and Validation...")
    
    # Temporal Audit
    p_year_counts = presence_df['year'].value_counts(normalize=True).sort_index()
    b_year_counts = background_thinned['year'].value_counts(normalize=True).sort_index()
    temporal_audit = pd.DataFrame({'presence_prop': p_year_counts, 'background_prop': b_year_counts}).fillna(0)
    temporal_audit.to_csv(os.path.join(VAL_DIR, 'background_temporal_audit.csv'))

    coast_dist = distance_transform_edt(land_mask)
    pixel_size_km = abs(gt[1]) * 111.32
    
    def get_coast_dists(df):
        dists = []
        for _, r in df.iterrows():
            px, py = int((r['longitude'] - gt[0]) / gt[1]), int((r['latitude'] - gt[3]) / gt[5])
            if 0 <= px < land_mask.shape[1] and 0 <= py < land_mask.shape[0]:
                dists.append(coast_dist[py, px] * pixel_size_km)
        return dists

    p_coast = get_coast_dists(presence_df)
    b_coast = get_coast_dists(background_thinned)
    ks_coast, _ = ks_2samp(p_coast, b_coast)
    
    presence_weights = []
    for _, row in presence_df.iterrows():
        px, py = int((row['longitude'] - gt[0]) / gt[1]), int((row['latitude'] - gt[3]) / gt[5])
        if 0 <= px < kde_array.shape[1] and 0 <= py < kde_array.shape[0]:
            val = kde_array[py, px]
            if val != nodata: presence_weights.append(val)
    
    ks_weight, _ = ks_2samp(presence_weights, background_thinned['weight'])
    
    overlap_metrics = [
        {'metric': 'kde_weight_ks_stat', 'value': ks_weight},
        {'metric': 'latitude_ks_stat', 'value': ks_2samp(presence_df['latitude'], background_thinned['latitude'])[0]},
        {'metric': 'coast_dist_ks_stat', 'value': ks_coast}
    ]
    pd.DataFrame(overlap_metrics).to_csv(os.path.join(VAL_DIR, 'presence_background_overlap.csv'), index=False)
    
    # 6d. Spatiotemporal Leakage Audit
    # Find distance to nearest presence and temporal overlap
    dists, near_p_idx = presence_tree.query(final_coords_proj, k=1)
    
    leakage_audit = pd.DataFrame({
        'dist_to_presence_m': dists,
        'same_year': [background_thinned.iloc[i]['year'] == presence_df.iloc[near_p_idx[i]]['year'] for i in range(len(dists))],
        'same_month': [background_thinned.iloc[i]['month'] == presence_df.iloc[near_p_idx[i]]['month'] for i in range(len(dists))]
    })
    
    leakage_summary = pd.DataFrame({
        'metric': ['min_dist_m', 'same_year_overlap_pct', 'same_month_overlap_pct'],
        'value': [dists.min(), leakage_audit['same_year'].mean() * 100, leakage_audit['same_month'].mean() * 100]
    })
    leakage_summary.to_csv(os.path.join(VAL_DIR, 'spatiotemporal_leakage_audit.csv'), index=False)
    
    # Final export
    background_thinned[['longitude', 'latitude', 'year', 'month', 'target', 'stratum', 'weight']].to_csv(
        os.path.join(OUTPUT_DIR, 'background_pool.csv'), index=False
    )
    log_message(f"Phase 3.1c Complete. Final records: {n_final}")

if __name__ == "__main__":
    start_time = time.time()
    try:
        generate_background()
        log_message(f"Runtime: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        log_message(f"CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
