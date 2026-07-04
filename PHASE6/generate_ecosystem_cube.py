import os
import json
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
from affine import Affine

# --- CONFIGURATION ---
BASE_DIR = r"C:\Users\Abhishek\IIITH\IITH\INDEPENDENT_STUDY\PROJECT"
PHASE6_DIR = os.path.join(BASE_DIR, "PHASE6")
MANIFEST_PATH = os.path.join(PHASE6_DIR, "manifests", "ecosystem_manifest.json")
OUTPUT_DIR = os.path.join(PHASE6_DIR, "ecosystem_indices")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize(arr, mask):
    valid = arr[mask]
    if valid.size == 0: return np.zeros_like(arr)
    v_min, v_max = np.nanmin(valid), np.nanmax(valid)
    out = np.zeros_like(arr)
    out[mask] = (arr[mask] - v_min) / (v_max - v_min + 1e-8)
    return out

def generate_ecosystem_cube():
    print("[PROCESS] Phase 6: Final Scientific Synthesis (Weighted & Agreement Metrics)")
    
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    
    layers = manifest['layers']
    ref_grid = manifest['reference_grid']
    height, width = ref_grid['height'], ref_grid['width']
    
    # Accumulators
    suitability_sum = np.zeros((height, width), dtype=np.float32)
    uncertainty_sum = np.zeros((height, width), dtype=np.float32)
    extrapolation_sum = np.zeros((height, width), dtype=np.float32)
    binary_presence_stack = [] # For species agreement
    species_support = np.zeros((height, width), dtype=np.int16) 
    
    for species_meta in tqdm(layers, desc="Aggregating Scientific Stack"):
        threshold = species_meta['threshold']
        with rasterio.open(os.path.join(BASE_DIR, species_meta['paths']['suitability'])) as src_s, \
             rasterio.open(os.path.join(BASE_DIR, species_meta['paths']['uncertainty'])) as src_u, \
             rasterio.open(os.path.join(BASE_DIR, species_meta['paths']['extrapolation'])) as src_e:
            
            s, u, e = src_s.read(1), src_u.read(1), src_e.read(1)
            valid_mask = np.isfinite(s) & np.isfinite(u) & np.isfinite(e) & (s != src_s.nodata)
            
            species_support += valid_mask.astype(np.int16)
            suitability_sum += np.where(valid_mask, s, 0)
            uncertainty_sum += np.where(valid_mask, u, 0)
            extrapolation_sum += np.where(valid_mask, e, 0)
            binary_presence_stack.append((np.where(valid_mask, s, 0) >= threshold).astype(np.float32))

    eco_mask = (species_support >= 1)
    
    # 1. PRIMARY METRICS
    mean_suit = np.zeros((height, width), dtype=np.float32)
    mean_unc = np.zeros((height, width), dtype=np.float32)
    mean_extra = np.zeros((height, width), dtype=np.float32)
    np.divide(suitability_sum, species_support, out=mean_suit, where=species_support > 0)
    np.divide(uncertainty_sum, species_support, out=mean_unc, where=species_support > 0)
    np.divide(extrapolation_sum, species_support, out=mean_extra, where=species_support > 0)
    
    # 2. POWER METRICS (NEW)
    print("[PROCESS] Calculating Species Agreement & Weighted Richness...")
    presence_stack = np.stack(binary_presence_stack)
    # Binary Richness (Sum of thresholded presences)
    binary_richness = np.sum(presence_stack, axis=0)
    # Species Agreement (1 - StdDev of binary presence)
    agreement = np.zeros((height, width), dtype=np.float32)
    agreement[eco_mask] = 1.0 - np.std(presence_stack[:, eco_mask], axis=0)
    
    # 3. PRIORITY INDEX
    raw_stability = np.zeros((height, width), dtype=np.float32)
    np.divide(mean_suit, (mean_unc + 0.01), out=raw_stability, where=eco_mask)
    stab_p99 = np.percentile(raw_stability[eco_mask], 99)
    eco_stability = np.clip(raw_stability, 0, stab_p99)
    
    n_rich = normalize(binary_richness, eco_mask)
    n_stab = normalize(eco_stability, eco_mask)
    n_extra = normalize(mean_extra, eco_mask)
    priority_score = np.clip(n_rich * n_stab * (1.0 - n_extra), 0, 1)
    
    # 4. HOTSPOT LOGIC
    p_rich_95 = np.percentile(binary_richness[eco_mask], 95)
    p_stab_75 = np.percentile(eco_stability[eco_mask], 75)
    p_extra_50 = np.percentile(mean_extra[eco_mask], 50)
    hotspots = (eco_mask & (binary_richness >= p_rich_95) & (eco_stability >= p_stab_75) & (mean_extra <= p_extra_50)).astype(np.uint8)

    # 5. EXPORTS
    common_meta = {'driver': 'GTiff', 'height': height, 'width': width, 'count': 1, 'crs': ref_grid['crs'], 'transform': Affine(*ref_grid['transform']), 'compress': 'lzw', 'BIGTIFF': 'YES'}

    outputs = {
        "ECO_Weighted_Richness.tif": suitability_sum,
        "ECO_Binary_Richness.tif": binary_richness.astype(np.float32),
        "ECO_Species_Agreement.tif": agreement,
        "ECO_Scientific_Stability.tif": eco_stability,
        "ECO_Conservation_Priority.tif": priority_score,
        "ECO_Universal_Hotspots.tif": hotspots
    }
    
    for filename, arr in outputs.items():
        dtype = 'float32' if 'Hotspots' not in filename else 'uint8'
        nodata = -9999.0 if dtype == 'float32' else 255
        export_arr = arr.copy()
        if dtype == 'float32': export_arr[~eco_mask] = nodata
        else: export_arr[~eco_mask] = 255
        with rasterio.open(os.path.join(OUTPUT_DIR, filename), 'w', dtype=dtype, nodata=nodata, **common_meta) as dst:
            dst.write(export_arr, 1)

    print(f"[SUCCESS] Synthesis complete. Weighted Richness and Species Agreement added.")

if __name__ == "__main__":
    generate_ecosystem_cube()
