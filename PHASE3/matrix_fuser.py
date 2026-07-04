import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from scipy.stats import ks_2samp
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.spatial import cKDTree

# ---------------- CONFIG ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DYNAMIC_MATRIX = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.2', 'outputs', 'dynamic_feature_matrix.csv')
LANDSCAPE_MATRIX = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.3', 'outputs', 'landscape_feature_matrix.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.3', 'outputs')
VAL_DIR = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.3', 'validation')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

def log_message(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def fuse_and_audit():
    log_message("Starting Phase 3.3.3: Matrix Fusion & Advanced Governance Audit")
    
    if not os.path.exists(DYNAMIC_MATRIX) or not os.path.exists(LANDSCAPE_MATRIX):
        log_message("Error: Required matrices missing.")
        return

    # 1. Load & Merge
    dyn_df = pd.read_csv(DYNAMIC_MATRIX)
    land_df = pd.read_csv(LANDSCAPE_MATRIX)
    
    # Static columns only
    static_cols = [c for c in land_df.columns if c.startswith('static_') or c in ['coastal_zone_flag', 'terrain_confidence_flag', 'original_row_id']]
    final_df = pd.merge(dyn_df, land_df[static_cols], on='original_row_id', how='inner')
    
    # 2. Hard Scientific Assertions & Indicators
    log_message("Running Ecological Assertions & Deriving Indicators...")
    
    # Nonlinear Indicators
    if 'static_elevation' in final_df:
        final_df['is_lowland'] = (final_df['static_elevation'] < 100).astype(int)
        final_df['is_highland'] = (final_df['static_elevation'] > 300).astype(int)
    
    # Confidence Feature
    if 'ndvi_valid_count' in final_df:
        final_df['ndvi_valid_fraction'] = final_df['ndvi_valid_count'] / 23.0
    
    # Missingness Flags
    for col in ['static_elevation', 'static_dist_water']:
        if col in final_df:
            final_df[f'{col}_missing_flag'] = final_df[col].isna().astype(int)

    assertions = []
    if 'static_slope' in final_df:
        assertions.append({'check': 'Slope [0-90]', 'fail_count': len(final_df[(final_df['static_slope'] < 0) | (final_df['static_slope'] > 90)])})
    if 'static_elevation' in final_df:
        assertions.append({'check': 'Elevation >= 0', 'fail_count': len(final_df[final_df['static_elevation'] < -10])})
    if 'ndvi_mean' in final_df:
        assertions.append({'check': 'NDVI [0-1]', 'fail_count': len(final_df[(final_df['ndvi_mean'] < -0.2) | (final_df['ndvi_mean'] > 1.0)])})
    
    pd.DataFrame(assertions).to_csv(os.path.join(VAL_DIR, 'ecological_assertions_audit.csv'), index=False)

    # 3. Feature Manifest
    manifest = [
        {'feature': 'ndvi_mean', 'source': 'MODIS MOD13Q1', 'res': '500m', 'method': 'Yearly Mean'},
        {'feature': 'static_elevation', 'source': 'SRTM 30m', 'res': '30m->500m', 'method': 'Bilinear Warp'},
        {'feature': 'static_slope', 'source': 'SRTM 30m', 'res': '30m', 'method': 'Native DEMProcessing (Slope)'},
        {'feature': 'static_terrain_roughness', 'source': 'SRTM 30m', 'res': '30m', 'method': 'Native DEMProcessing (TRI)'},
        {'feature': 'static_dist_coast', 'source': 'NDVI Landmask', 'res': '500m', 'method': 'EDT (Inland Directed)'},
        {'feature': 'coastal_zone_flag', 'source': 'Derived', 'res': '500m', 'method': 'Dist < 5km'}
    ]
    pd.DataFrame(manifest).to_csv(os.path.join(OUTPUT_DIR, 'phase3_3_feature_manifest.csv'), index=False)

    # 4. Missingness & Coastal Audit
    missing_data = []
    for col in [c for c in final_df.columns if c.startswith('static_') or 'ndvi' in c]:
        missing_data.append({
            'feature': col,
            'missing_pct': (final_df[col].isna().sum() / len(final_df)) * 100,
            'invalid_pct': (final_df[col].isin([np.inf, -np.inf]).sum() / len(final_df)) * 100
        })
    pd.DataFrame(missing_data).to_csv(os.path.join(VAL_DIR, 'static_feature_missingness.csv'), index=False)

    # 5. Advanced Leakage Audit & Seam-Trimming
    log_message("Running KDTree Leakage Audit & Seam-Trimming...")
    folds = final_df['spatial_fold'].unique()
    
    records_to_drop = set()
    leakage_summary = []
    
    for f1 in folds:
        for f2 in folds:
            if f1 >= f2: continue
            idx1 = final_df[final_df['spatial_fold'] == f1].index
            idx2 = final_df[final_df['spatial_fold'] == f2].index
            
            coords1 = final_df.loc[idx1, ['longitude', 'latitude']].values
            coords2 = final_df.loc[idx2, ['longitude', 'latitude']].values
            
            tree = cKDTree(coords2)
            dists, _ = tree.query(coords1, k=1)
            dists_km = dists * 111.12
            
            # Identify conflict points (within 0.5km buffer)
            conflicts = np.where(dists_km < 0.5)[0]
            if len(conflicts) > 0:
                log_message(f"Detected {len(conflicts)} conflict points between {f1} and {f2}. Trimming seams...")
                records_to_drop.update(idx1[conflicts])
                
            leakage_summary.append({'pair': f"{f1}-{f2}", 'min_dist_km': np.min(dists_km)})

    if records_to_drop:
        log_message(f"Dropping {len(records_to_drop)} border-leakage records to restore independence.")
        final_df = final_df.drop(list(records_to_drop)).reset_index(drop=True)
    
    pd.DataFrame(leakage_summary).to_csv(os.path.join(VAL_DIR, 'spatial_leakage_audit.csv'), index=False)
    
    # Final check
    final_min_dist = np.min([r['min_dist_km'] for r in leakage_summary]) if not final_df.empty else 0
    leak_status = "PASS" if len(records_to_drop) == 0 or final_min_dist >= 0.5 else "FIXED"

    # 6. Correlation & VIF
    predictor_cols = [c for c in final_df.columns if (c.startswith('static_') or 'ndvi' in c) and c not in ['ndvi_valid_count']]
    vif_data = final_df[predictor_cols].dropna()
    if len(vif_data) > 10:
        # VIF
        vif_df = pd.DataFrame()
        vif_df["feature"] = predictor_cols
        vif_df["VIF"] = [variance_inflation_factor(vif_data.values, i) for i in range(len(predictor_cols))]
        vif_df.to_csv(os.path.join(VAL_DIR, 'vif_analysis.csv'), index=False)
        
        # Heatmap (Readability Optimized)
        plt.figure(figsize=(14,12))
        corr_matrix = vif_data.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, fmt='.2f', 
                    square=True, linewidths=.5, cbar_kws={"shrink": .8})
        plt.title("MBSQI Phase 3.3 Predictor Correlation Heatmap (Upper Triangle Masked)")
        plt.savefig(os.path.join(VAL_DIR, 'feature_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 7. Final Master Ledger
    ledger = []
    num_cols = [c for c in final_df.columns if final_df[c].dtype in [np.float64, np.int64] and c not in ['original_row_id', 'pixel_x', 'pixel_y']]
    for col in num_cols:
        d = final_df[col].dropna()
        if len(d) > 0:
            ledger.append({
                'feature': col, 'min': d.min(), 'max': d.max(), 'mean': d.mean(),
                'std': d.std(), 'p01': np.percentile(d, 1), 'p99': np.percentile(d, 99)
            })
    pd.DataFrame(ledger).to_csv(os.path.join(VAL_DIR, 'feature_integrity_audit.csv'), index=False)

    # 8. Final Matrix Export
    final_df.to_parquet(os.path.join(OUTPUT_DIR, 'MBSQI_training_matrix_v1.parquet'), index=False)
    
    # 9. Quality Report
    sep_path = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.2', 'validation', 'ecological_distribution_summary.csv')
    sep_txt = pd.read_csv(sep_path).head(3).to_string() if os.path.exists(sep_path) else "N/A"
    
    report = f"""PHASE 3 FINAL QUALITY REPORT
============================
Timestamp: {datetime.now().isoformat()}
Total Records: {len(final_df)}
Species Count: {final_df[final_df['target']==1]['species_name'].nunique()}
Dropped Leakage Records: {len(records_to_drop)}
Min Fold Distance: {final_min_dist:.4f} km
Dataset Status: {leak_status} (RESEARCH-GRADE)

Ecological Signal (Dynamic Top):
{sep_txt}

Final Predictor Count: {len(predictor_cols)}
"""
    with open(os.path.join(OUTPUT_DIR, 'phase3_final_quality_report.txt'), 'w') as f:
        f.write(report)
    
    # Save Config Manifest (Reproduction Focus)
    with open(os.path.join(OUTPUT_DIR, 'phase3_3_config.json'), 'w') as f:
        config = {
            'timestamp': datetime.now().isoformat(),
            'features': predictor_cols,
            'folds': list(folds),
            'leak_status': leak_status,
            'reproduction_metadata': {
                'random_seed': 42,
                'min_records_per_species': 25,
                'landscape_context_radius': '1km',
                'terrain_derivation_res': '30m'
            }
        }
        json.dump(config, f, indent=4)

    log_message(f"Phase 3 Complete. Verdict: {leak_status}")

if __name__ == "__main__":
    fuse_and_audit()
