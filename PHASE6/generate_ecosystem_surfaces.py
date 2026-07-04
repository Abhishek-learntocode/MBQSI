import os
import json
import pandas as pd
import rasterio
import numpy as np
from datetime import datetime
import hashlib

# ---------------- CONFIG ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PHASE4_RASTERS = os.path.join(BASE_DIR, 'PHASE4', 'rasters')
SUMMARY_PATH = os.path.join(BASE_DIR, 'PHASE4', 'validation', 'model_performance_summary.csv')
PHASE6_DIR = os.path.join(BASE_DIR, 'PHASE6')
MANIFEST_PATH = os.path.join(PHASE6_DIR, 'manifests', 'ecosystem_manifest.json')

def generate_standardization_manifest():
    """Phase 1: Production-grade gatekeeper for raster standardization and ecosystem manifest generation."""
    print("[PROCESS] Phase 1: Production Raster Standardization & Manifest Generation")
    
    # 1. Load Governance Summary
    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError(f"CRITICAL: Performance summary missing at {SUMMARY_PATH}")
        
    df = pd.read_csv(SUMMARY_PATH)
    # Strip whitespace to prevent matching failures
    df['species'] = df['species'].str.strip()
    df['governance'] = df['governance'].str.strip()
    
    # LOAD THRESHOLDS FROM FOLD METRICS
    FOLD_METRICS_PATH = os.path.join(os.path.dirname(SUMMARY_PATH), 'fold_performance_metrics.csv')
    if os.path.exists(FOLD_METRICS_PATH):
        fdf = pd.read_csv(FOLD_METRICS_PATH)
        fdf['species'] = fdf['species'].str.strip()
        # Calculate mean threshold per species
        thresholds = fdf.groupby('species')['th'].mean().to_dict()
        print(f"[DEBUG] Sample thresholds keys: {list(thresholds.keys())[:5]}")
        df['threshold'] = df['species'].map(thresholds)
    
    manifest = {
        "project": "MBSQI Ecosystem Intelligence",
        "generated_at": datetime.now().isoformat(),
        "version": "1.0.0",
        "threshold_method": "Species-Specific Optimization (Youden J / F1)",
        "governance_summary": df['governance'].value_counts().to_dict(),
        "reference_grid": None,
        "layers": [],
        "excluded_species": [],
        "validation_summary": {"validated_species": 0, "excluded_species": 0, "layer_triads_verified": 0}
    }

    # Only include species that meet scientific rigor
    pass_species = df[df['governance'].isin(['PASS', 'WARN'])].copy()
    print(f"[DEBUG] Processing {len(pass_species)} species: {pass_species['species'].tolist()}")
    
    if pass_species.empty:
        raise ValueError("ZERO species passed governance check. Halting.")

    reference_spec = None
    
    # 2. Iterate and Verify Layer Triad Alignment
    debug_log = []
    for _, row in pass_species.iterrows():
        species = row['species']
        safe_name = species.replace(' ', '_').replace('/', '_')
        debug_log.append(f"\n[SPECIES] {species} -> Safe: {safe_name}")
        
        # Threshold Integrity Check
        th_val = row.get('threshold')
        debug_log.append(f"  [CHECK] Threshold: {th_val}")
        if pd.isnull(th_val):
            manifest["excluded_species"].append({"species": species, "reason": "Missing optimal threshold"})
            debug_log.append("  [FAIL] Missing threshold")
            continue

        # Layer Triad Existence Check
        species_layers = {}
        missing_layers = []
        layer_suffixes = {
            'suitability': '_suitability.tif',
            'uncertainty': '_uncertainty_std.tif',
            'extrapolation': '_extrapolation.tif'
        }
        
        for l_type, l_suffix in layer_suffixes.items():
            l_path = os.path.abspath(os.path.join(PHASE4_RASTERS, f"{safe_name}{l_suffix}"))
            exists = os.path.exists(l_path)
            debug_log.append(f"  [CHECK] {l_type}: {l_path} | Exists: {exists}")
            if exists:
                species_layers[l_type] = l_path
            else:
                missing_layers.append(l_type)

        if missing_layers:
            manifest["excluded_species"].append({"species": species, "reason": f"Missing layer triad: {missing_layers}"})
            debug_log.append(f"  [FAIL] Missing layers: {missing_layers}")
            continue

        # 3. Spatial Alignment & Integrity Verification (Robust Triad Check)
        try:
            print(f"  [PROCESS] Validating {species}...", flush=True)
            current_species_data = {"species": species, "safe_name": safe_name, "valid_coverage": {}}
            
            for layer_name, layer_path in species_layers.items():
                with rasterio.open(layer_path) as src:
                    meta = src.meta
                    
                    # Establish/Verify Reference Grid
                    if reference_spec is None:
                        reference_spec = {
                            "width": meta["width"],
                            "height": meta["height"],
                            "crs": str(meta["crs"]),
                            "transform": [float(x) for x in src.transform],
                            "bounds": list(src.bounds),
                            "nodata": float(meta["nodata"]) if meta["nodata"] is not None else -9999.0,
                            "dtype": str(np.dtype(meta["dtype"]))
                        }
                        manifest["reference_grid"] = reference_spec
                        print(f"[INFO] Reference Baseline Established: {meta['width']}x{meta['height']} | {meta['crs']}")

                    # STRICT CROSS-LAYER & CROSS-SPECIES VALIDATION
                    mismatches = []
                    if meta['width'] != reference_spec['width']: mismatches.append(f"Width: {meta['width']} vs {reference_spec['width']}")
                    if meta['height'] != reference_spec['height']: mismatches.append(f"Height: {meta['height']} vs {reference_spec['height']}")
                    if str(meta['crs']) != reference_spec['crs']: mismatches.append(f"CRS: {meta['crs']} vs {reference_spec['crs']}")
                    
                    # Numerical Tolerance for Transform
                    current_transform = [float(x) for x in src.transform]
                    if not np.allclose(current_transform, reference_spec['transform'], atol=1e-10):
                        mismatches.append("Transform mismatch (significant)")
                        
                    if float(meta.get('nodata', -9999)) != reference_spec['nodata']: mismatches.append(f"NoData: {meta.get('nodata')} vs {reference_spec['nodata']}")
                    
                    if mismatches:
                        raise ValueError(f"Spatial mismatch in {layer_name}: {', '.join(mismatches)}")

                    # Valid Pixel Audit (Eco-Support Metadata)
                    arr = src.read(1, masked=True)
                    current_species_data["valid_coverage"][layer_name] = int(np.sum(~arr.mask))

            # Record Success
            current_species_data["paths"] = {k: os.path.relpath(v, BASE_DIR) for k, v in species_layers.items()}
            current_species_data["threshold"] = float(th_val)
            current_species_data["auc"] = float(row['auc_mean'])
            manifest['layers'].append(current_species_data)
            manifest["validation_summary"]["validated_species"] += 1
            manifest["validation_summary"]["layer_triads_verified"] += 1

        except Exception as e:
            err_msg = str(e)
            print(f"  [FAIL] {species}: {err_msg}")
            manifest["excluded_species"].append({"species": species, "reason": f"Corruption or Alignment Error: {err_msg}"})
            debug_log.append(f"  [FAIL] Spatial/Integrity Error: {err_msg}")
            manifest["validation_summary"]["excluded_species"] += 1

    # 4. Final Operational Governance
    if reference_spec is None:
        # Write log before failing
        with open(os.path.join(PHASE6_DIR, 'manifests', 'debug_log.txt'), 'w') as f:
            f.write("\n".join(debug_log))
        raise ValueError("CRITICAL: No valid raster triads found in ecosystem stack.")

    # Memory Governance
    pixel_count = reference_spec['width'] * reference_spec['height']
    raw_cube_gb = (3 * len(manifest['layers']) * pixel_count * 4) / (1024**3)
    manifest["memory_governance"] = {
        "raw_cube_ram_gb": round(raw_cube_gb, 3),
        "estimated_peak_ram_gb": round(raw_cube_gb * 3, 3), # 3x overhead for masking/normalization
        "chunk_strategy": "recommended" if raw_cube_gb * 3 > 4.0 else "full_load"
    }

    # Integrity Hash (SHA-256 for provenance)
    id_string = "".join([l['species'] for l in manifest['layers']]) + str(manifest['generated_at'])
    manifest["integrity_hash"] = hashlib.sha256(id_string.encode()).hexdigest()

    # 5. Save Manifest and Log
    with open(os.path.join(PHASE6_DIR, 'manifests', 'debug_log.txt'), 'w') as f:
        f.write("\n".join(debug_log))
        
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=4)
    
    print(f"[SUCCESS] Phase 1 Complete. {manifest['validation_summary']['validated_species']} species verified.")
    print(f"[INFO] Estimated PEAK RAM: {manifest['memory_governance']['estimated_peak_ram_gb']} GB")
    return manifest

if __name__ == "__main__":
    generate_standardization_manifest()
