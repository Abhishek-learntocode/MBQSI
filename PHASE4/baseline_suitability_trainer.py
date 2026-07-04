import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import time
from datetime import datetime
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix

# ---------------- CONFIG ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MATRIX_PATH = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.3', 'outputs', 'MBSQI_training_matrix_v1.parquet')
OUTPUT_DIR = os.path.join(BASE_DIR, 'PHASE4', 'outputs')
MODELS_DIR = os.path.join(BASE_DIR, 'PHASE4', 'models')
VAL_DIR = os.path.join(BASE_DIR, 'PHASE4', 'validation')

for d in [OUTPUT_DIR, MODELS_DIR, VAL_DIR]: os.makedirs(d, exist_ok=True)

# GPU Check
GPU_AVAILABLE = False
try:
    xgb.XGBClassifier(tree_method='hist', device='cuda').fit(np.zeros((2,2)), np.array([0,1]))
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False

FEATURES = [
    'ndvi_mean', 'ndvi_std', 'ndvi_cv', 'ndvi_temporal_range',
    'static_elevation', 'static_slope', 'static_dist_coast', 'static_dist_water',
    'static_local_ndvi_mean_1km', 'static_local_ndvi_std_1km',
    'ndvi_valid_fraction', 'is_lowland', 'is_highland'
]

def log_message(msg):
    tqdm.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def optimize_tss_threshold(y_true, y_prob):
    thresholds = np.arange(0.01, 1.0, 0.01)
    tss_scores = []
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        tss_scores.append(sens + spec - 1)
    return thresholds[np.argmax(tss_scores)], np.max(tss_scores)

def train_governed_model(species_name, df, is_null_model=False):
    X = df[FEATURES].copy()
    y = df['target'].copy()
    n_pres = len(df[df['target'] == 1])
    p = {'max_depth': 4, 'n_est': 500, 'lr': 0.05} if n_pres > 500 else \
        ({'max_depth': 3, 'n_est': 300, 'lr': 0.03} if n_pres > 100 else \
         {'max_depth': 2, 'n_est': 150, 'lr': 0.02})

    gkf = GroupKFold(n_splits=4)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, df['spatial_fold'])):
        f_start = time.time()
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        if is_null_model:
            rng = np.random.default_rng(42 + fold)
            y_tr = rng.permutation(y_tr)

        # Scientific Governance: Disable Early Stopping for Null models or Sparse Folds
        use_early_stopping = (not is_null_model) and (np.sum(y_tr == 1) > 25)
        
        # Base Model (Vanilla Constructor)
        model = xgb.XGBClassifier(
            max_depth=p['max_depth'], n_estimators=p['n_est'], learning_rate=p['lr'],
            subsample=0.7, colsample_bytree=0.7, random_state=42,
            tree_method='hist', device='cuda' if GPU_AVAILABLE else 'cpu', n_jobs=4
        )

        if use_early_stopping:
            try:
                X_vtr, X_vte, y_vtr, y_vte = train_test_split(X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=42)
                # Pass Early Stopping directly to FIT, not constructor
                model.fit(X_vtr, y_vtr, eval_set=[(X_vte, y_vte)], early_stopping_rounds=30, eval_metric='logloss', verbose=False)
            except:
                use_early_stopping = False # Fallback

        if not use_early_stopping:
            model.fit(X_tr, y_tr, verbose=False)
        
        calib = CalibratedClassifierCV(model, cv=3, method='sigmoid', ensemble=False)
        calib.fit(X_tr, y_tr)
        
        probs = calib.predict_proba(X_te)[:, 1]
        best_th, best_tss = optimize_tss_threshold(y_te, probs)
        
        prob_true, prob_pred = calibration_curve(y_te, probs, n_bins=10)
        ece = np.mean(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0
        
        fold_results.append({
            'fold': fold, 'auc': roc_auc_score(y_te, probs), 'pr_auc': average_precision_score(y_te, probs),
            'tss': best_tss, 'th': best_th, 'brier': brier_score_loss(y_te, probs), 'ece': ece,
            'time': time.time() - f_start
        })
        
        if not is_null_model:
            safe_name = species_name.replace(' ', '_').replace('/', '_')
            joblib.dump(calib, os.path.join(MODELS_DIR, f"{safe_name}_fold{fold}.joblib"))

    return pd.DataFrame(fold_results)

def run_pipeline():
    log_message(f"XGBoost Build Info: {xgb.build_info()}")
    log_message(f"Starting Phase 4 Modeling (GPU: {GPU_AVAILABLE})")
    df_all = pd.read_parquet(MATRIX_PATH)
    valid_species = df_all[df_all['target'] == 1]['species_name'].value_counts()[lambda x: x >= 25].index
    
    # FULL PRODUCTION MODE: All species with >= 25 records
    valid_species = list(valid_species)
    
    report_path = os.path.join(VAL_DIR, 'model_performance_summary.csv')
    fold_path = os.path.join(VAL_DIR, 'fold_performance_metrics.csv')
    log_path = os.path.join(VAL_DIR, 'failed_species.log')

    if os.path.exists(log_path): os.remove(log_path)

    all_summaries = []
    all_fold_metrics = []

    for idx, species in enumerate(tqdm(valid_species, desc="Archipelago Modeling")):
        try:
            start_time = time.time()
            s_df = df_all[(df_all['species_name'] == species) | (df_all['target'] == 0)]
            tqdm.write(f"\n[{idx+1}/{len(valid_species)}] Processing: {species}")
            
            real_df = train_governed_model(species, s_df, is_null_model=False)
            num_cols = ['auc', 'pr_auc', 'tss', 'th', 'brier', 'ece', 'time']
            stats = real_df[num_cols].agg(['mean', 'std']).T
            
            null_aucs = [train_governed_model(species, s_df, True)['auc'].mean() for _ in range(3)]
            null_auc_mean = np.mean(null_aucs)
            
            summary = {
                'species': species, 'auc_mean': stats.loc['auc', 'mean'], 'auc_std': stats.loc['auc', 'std'],
                'pr_auc_mean': stats.loc['pr_auc', 'mean'], 'tss_mean': stats.loc['tss', 'mean'],
                'brier_mean': stats.loc['brier', 'mean'], 'null_auc_mean': null_auc_mean,
                'runtime_sec': time.time() - start_time,
                'stability': 'PASS' if stats.loc['auc', 'std'] < 0.1 else 'WARN',
                'governance': 'PASS' if stats.loc['auc', 'mean'] > null_auc_mean + 0.1 else 'FAIL'
            }
            all_summaries.append(summary)
            real_df['species'] = species
            all_fold_metrics.append(real_df)
            
        except Exception as e:
            log_message(f"FAILED: {species} -> {e}")
            with open(log_path, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - {species}: {str(e)}\n")

    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(report_path, index=False)
        pd.concat(all_fold_metrics).to_csv(fold_path, index=False)
        log_message(f"Reports saved to {VAL_DIR}")

    log_message("Phase 4 Pipeline Finalized.")

if __name__ == "__main__":
    run_pipeline()
