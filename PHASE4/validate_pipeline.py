import os
import pandas as pd
import numpy as np
import json
import joblib
import platform
import sys
import xgboost as xgb
import rasterio
import time
from datetime import datetime
from tqdm.auto import tqdm

# ---------------- CONFIG ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OBS_PATH = os.path.join(BASE_DIR, 'PHASE1', 'CLEANED_DATA', 'MIGRATORY_BIRDS_THINNED_FINAL.csv')
LEAKAGE_PATH = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.3', 'validation', 'spatial_leakage_audit.csv')

PHASE3_3_DIR = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.3', 'outputs')
PHASE4_DIR = os.path.join(BASE_DIR, 'PHASE4')
RASTER_DIR = os.path.join(PHASE4_DIR, 'rasters')
MODELS_DIR = os.path.join(PHASE4_DIR, 'models')
OUTPUT_DIR = os.path.join(PHASE4_DIR, 'validation', 'dossier')
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_PATH = os.path.join(OUTPUT_DIR, 'final_pipeline_validation_report.md')

def log_msg(m): tqdm.write(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")

class PipelineAuditor:
    def __init__(self):
        self.results = {}
        self.warnings = []
        self.failures = []
        log_msg("Initializing Scientific Auditor (Final Production Lockdown)...")

    def get_status(self, keywords):
        """Derives PASS/FAIL status from central lists with case-insensitivity."""
        for fail in self.failures:
            if any(kw.lower() in fail.lower() for kw in keywords): return "FAIL"
        for warn in self.warnings:
            if any(kw.lower() in warn.lower() for kw in keywords): return "WARN"
        return "PASS"

    def get_valid_mask(self, arr, nodata):
        if nodata is None: return np.isfinite(arr)
        elif isinstance(nodata, float) and np.isnan(nodata): return np.isfinite(arr)
        else: return ~np.isclose(arr, nodata)

    def audit_section_1_governance(self):
        log_msg("Section 1: Data Governance & Spatial Leakage...")
        if os.path.exists(OBS_PATH):
            df = pd.read_csv(OBS_PATH)
            self.results['raw_obs_total'] = len(df)
            outliers = df[(df['latitude'] < 6) | (df['latitude'] > 15) | (df['longitude'] < 91) | (df['longitude'] > 95)]
            self.results['coord_outliers'] = len(outliers)
            if len(outliers) > 0: self.failures.append(f"Coordinate outliers detected: {len(outliers)}")
        else:
            self.results['raw_obs_total'] = "FILE_MISSING"
            self.warnings.append(f"Observations file missing: {os.path.basename(OBS_PATH)}")
        
        if os.path.exists(LEAKAGE_PATH):
            ldf = pd.read_csv(LEAKAGE_PATH)
            if len(ldf.columns) == 0:
                self.failures.append("Spatial leakage audit malformed.")
                return
            dist_col = 'min_dist_km' if 'min_dist_km' in ldf.columns else ldf.columns[0]
            min_dist_m = ldf[dist_col].min() * 1000.0
            self.results['leakage_min_dist'] = min_dist_m
            if min_dist_m < 100: self.failures.append(f"Spatial leakage detected: {min_dist_m:.1f}m separation")
        else:
            self.results['leakage_min_dist'] = 0.0
            self.failures.append("Spatial leakage audit missing.")

    def audit_section_3_features(self):
        log_msg("Section 3: Environmental Feature Rigor...")
        grid_path = os.path.join(PHASE3_3_DIR, 'archipelago_feature_grid.parquet')
        if os.path.exists(grid_path):
            df = pd.read_parquet(grid_path)
            self.results['grid_pixels'] = len(df)
            ndvi_min, ndvi_max = df['ndvi_mean'].min(), df['ndvi_mean'].max()
            if ndvi_min < -1.1 or ndvi_max > 1.1: self.warnings.append(f"Ecological NDVI range suspicious: {ndvi_min:.2f} to {ndvi_max:.2f}")

    def audit_section_4_models(self):
        log_msg("Section 4: Model Stability & Fold Scrutiny...")
        summary_path = os.path.join(PHASE4_DIR, 'validation', 'model_performance_summary.csv')
        if not os.path.exists(summary_path): 
            self.failures.append("Model performance summary missing.")
            return
        
        df = pd.read_csv(summary_path)
        pass_species = df[df['governance'] == 'PASS']['species'].tolist()
        self.results['pass_count'] = len(pass_species)
        self.results['avg_auc'] = df[df['governance'] == 'PASS']['auc_mean'].mean()
        
        if self.results['pass_count'] < 30: self.failures.append(f"Insufficient pass species count: {self.results['pass_count']}")
        if self.results['avg_auc'] < 0.70: self.failures.append(f"Low mean ensemble performance: {self.results['avg_auc']:.4f}")

        missing_folds = 0
        corrupt_folds = 0
        for s in tqdm(pass_species, desc="Scanning Folds", leave=False):
            safe_name = s.replace(' ', '_').replace('/', '_')
            for f in range(4):
                p = os.path.join(MODELS_DIR, f"{safe_name}_fold{f}.joblib")
                if not os.path.exists(p): missing_folds += 1
                else:
                    try: joblib.load(p)
                    except: corrupt_folds += 1
        self.results['missing_folds'] = missing_folds
        self.results['corrupt_folds'] = corrupt_folds
        if (missing_folds + corrupt_folds) > 0: self.failures.append(f"Detected {missing_folds} missing and {corrupt_folds} corrupt model fold artifacts.")

    def audit_section_5_deployment(self):
        log_msg("Section 5: Continuous Deployment Quality...")
        tifs = [f for f in os.listdir(RASTER_DIR) if f.endswith('_suitability.tif')]
        self.results['total_suites'] = len(tifs)
        
        non_finite = 0
        bad_range = 0
        mean_extrap = []
        coverage_stats = []
        
        for f in tqdm(tifs, desc="Auditing Rasters", leave=False):
            with rasterio.open(os.path.join(RASTER_DIR, f)) as src:
                arr = src.read(1)
                mask = self.get_valid_mask(arr, src.nodata)
                valid = arr[mask]
                if len(valid) > 0:
                    if not np.all(np.isfinite(valid)): non_finite += 1
                    if np.min(valid) < -0.01 or np.max(valid) > 1.01: bad_range += 1
                    coverage_stats.append((np.sum(mask) / arr.size) * 100)
            
            ex_p = os.path.join(RASTER_DIR, f.replace('_suitability.tif', '_extrapolation.tif'))
            if os.path.exists(ex_p):
                with rasterio.open(ex_p) as src:
                    ex = src.read(1)
                    v_ex = ex[self.get_valid_mask(ex, src.nodata)]
                    if len(v_ex) > 0: mean_extrap.append(np.nanmean(v_ex))

        self.results['non_finite_rasters'] = non_finite
        self.results['bad_range_rasters'] = bad_range
        self.results['avg_extrap_burden'] = np.mean(mean_extrap) if mean_extrap else 0
        self.results['min_raster_coverage'] = np.min(coverage_stats) if coverage_stats else 0
        
        if non_finite > 0: self.failures.append(f"Instability: {non_finite} non-finite rasters.")
        if bad_range > 0: self.failures.append(f"Invalid probability range in {bad_range} rasters.")
        if self.results['min_raster_coverage'] < 3.0: self.failures.append(f"Low spatial coverage: {self.results['min_raster_coverage']:.2f}%")

    def audit_section_7_reproducibility(self):
        self.results['env_python'] = sys.version.split()[0]
        self.results['env_xgboost'] = xgb.__version__
        self.results['env_os'] = platform.system()
        self.results['env_time'] = datetime.now().isoformat()

    def generate_report(self):
        log_msg(f"Building Definitive Dossier: {REPORT_PATH}")
        with open(REPORT_PATH, 'w') as f:
            f.write("# MBSQI Pipeline: Final Scientific Validation Dossier\n\n")
            f.write(f"| Category | Status | Metric |\n")
            f.write(f"| :--- | :--- | :--- |\n")
            f.write(f"| **Governance** | {self.get_status(['Observations', 'Coordinate'])} | {self.results.get('raw_obs_total')} Obs |\n")
            f.write(f"| **Spatial Leakage** | {self.get_status(['leakage'])} | {self.results.get('leakage_min_dist', 0):.1f}m Separation |\n")
            f.write(f"| **Model Coverage** | {self.get_status(['Insufficient', 'Model performance', 'artifact'])} | {self.results.get('pass_count', 0)} Species |\n")
            f.write(f"| **Ensemble Accuracy** | {self.get_status(['Low mean'])} | {self.results.get('avg_auc', 0):.4f} Mean AUC |\n")
            f.write(f"| **Raster Integrity** | {self.get_status(['Instability', 'Invalid probability', 'coverage'])} | {self.results.get('total_suites', 0)} Suites |\n\n")
            
            f.write("## 1. Audit Summary\n")
            f.write(f"- **Total Failures**: {len(self.failures)}\n")
            f.write(f"- **Total Warnings**: {len(self.warnings)}\n\n")

            if self.failures:
                f.write("### CRITICAL FAILURES\n")
                for fail in self.failures: f.write(f"- [FAIL] {fail}\n")
                f.write("\n")

            if self.warnings:
                f.write("### SCIENTIFIC WARNINGS\n")
                for warn in self.warnings: f.write(f"- [WARN] {warn}\n")
                f.write("\n")
                
            f.write("## 7. Environment Reproducibility\n")
            f.write(f"- **OS**: {self.results.get('env_os')}\n")
            f.write(f"- **Python**: {self.results.get('env_python')}\n")
            f.write(f"- **XGBoost**: {self.results.get('env_xgboost')}\n")
            f.write(f"- **Timestamp**: {self.results.get('env_time')}\n\n")

            f.write("## 8. Final Scientific Verdict\n")
            if not self.failures and float(self.results.get('avg_auc', 0)) > 0.7:
                f.write("> [!TIP]\n> **STATUS: PUBLICATION READY.** The pipeline demonstrates zero spatial leakage, stable numerical continuity, and high mean ensemble performance.\n")
            else:
                f.write("> [!CAUTION]\n> **STATUS: SCIENTIFICALLY UNSTABLE.** Inspect critical failures before reporting results.\n")

    def run_full_audit(self):
        start = time.time()
        self.audit_section_1_governance()
        self.audit_section_3_features()
        self.audit_section_4_models()
        self.audit_section_5_deployment()
        self.audit_section_7_reproducibility()
        self.generate_report()
        log_msg(f"Full Dossier Generated in {time.time()-start:.1f}s.")

if __name__ == "__main__":
    PipelineAuditor().run_full_audit()
