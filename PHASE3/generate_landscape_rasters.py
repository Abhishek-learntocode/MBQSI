import os
import json
import numpy as np
from osgeo import gdal, osr, ogr
from scipy.ndimage import distance_transform_edt
from datetime import datetime

# ---------------- CONFIG ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
GRID_METADATA = os.path.join(BASE_DIR, 'PHASE2', 'metadata', 'reference_grid.json')
LANDMASK_PATH = os.path.join(BASE_DIR, 'PHASE2', 'yearly_aggregates', 'NDVI_valid_count_2024.tif')

# Source Data Paths (Found in Phase 4 script)
DEM_SRC = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.3', 'ELEVATION', 'Elevation_Andaman.tif')
WATER_SRC = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.3', 'Global Surface Water', 'occurrence_90E_20Nv1_4_2021.tif')

OUTPUT_DIR = os.path.join(BASE_DIR, 'PHASE2', 'static_layers')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log_message(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def create_raster(data, path, gt, proj, nodata=-9999):
    driver = gdal.GetDriverByName('GTiff')
    # Add LZW Compression and Tiling
    ds = driver.Create(path, data.shape[1], data.shape[0], 1, gdal.GDT_Float32, 
                       options=['COMPRESS=LZW', 'TILED=YES', 'PREDICTOR=2'])
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    band = ds.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(nodata)
    ds.FlushCache()
    return ds

def generate_rasters():
    log_message("Starting Phase 3.3.1: Landscape Raster Factory")
    
    # 1. Load Reference Grid
    with open(GRID_METADATA, 'r') as f:
        grid = json.load(f)
    gt_ref, proj_ref = tuple(grid['geo']), grid['proj']
    shape_ref = (grid['ys'], grid['xs'])
    
    # 2. Load Landmask
    mask_ds = gdal.Open(LANDMASK_PATH)
    mask = mask_ds.GetRasterBand(1).ReadAsArray()
    mask_nodata = mask_ds.GetRasterBand(1).GetNoDataValue()
    land_mask = (mask != mask_nodata) & (mask > 0)
    
    # --- A. Terrain (Elevation, Slope, Roughness) ---
    if os.path.exists(DEM_SRC):
        log_message("Processing Native Terrain Derivatives (30m)...")
        
        # 1. Native Slope
        slope_30m = os.path.join(OUTPUT_DIR, 'temp_slope_30m.tif')
        gdal.DEMProcessing(slope_30m, DEM_SRC, 'slope')
        
        # 2. Native Roughness (TRI/Roughness)
        rough_30m = os.path.join(OUTPUT_DIR, 'temp_rough_30m.tif')
        gdal.DEMProcessing(rough_30m, DEM_SRC, 'TRI')
        
        # 3. Warp to MODIS Grid
        for src, name in [(DEM_SRC, 'elevation'), (slope_30m, 'slope'), (rough_30m, 'terrain_roughness')]:
            gdal.Warp(os.path.join(OUTPUT_DIR, f'{name}.tif'), src, format='GTiff', 
                       dstSRS=proj_ref, xRes=gt_ref[1], yRes=abs(gt_ref[5]),
                       outputBounds=(gt_ref[0], gt_ref[3] + gt_ref[5]*shape_ref[0], 
                                     gt_ref[0] + gt_ref[1]*shape_ref[1], gt_ref[3]),
                       resampleAlg='bilinear' if name == 'elevation' else 'average')
        
        # Cleanup
        if os.path.exists(slope_30m): os.remove(slope_30m)
        if os.path.exists(rough_30m): os.remove(rough_30m)
    
    # --- B. Proximity (Distance to Coast) ---
    log_message("Computing Distance to Coast (Inland-Directed)...")
    # Correct Fix: distance_transform_edt computes distance to nearest zero.
    # Seed land_mask (Land=True/1, Ocean=False/0).
    # Result: Ocean pixels=0, Land pixels=distance to nearest ocean.
    dist_px = distance_transform_edt(land_mask)
    dist_km = (dist_px * gt_ref[1] * 111.12)
    create_raster(dist_km, os.path.join(OUTPUT_DIR, 'dist_coast.tif'), gt_ref, proj_ref)
    
    # --- C. Proximity (Distance to Inland Water) ---
    if os.path.exists(WATER_SRC):
        log_message("Processing Inland Hydrological Distance...")
        water_ds = gdal.Warp('', WATER_SRC, format='VRT', dstSRS=proj_ref, 
                             outputBounds=(gt_ref[0], gt_ref[3] + gt_ref[5]*shape_ref[0], 
                                           gt_ref[0] + gt_ref[1]*shape_ref[1], gt_ref[3]),
                             width=shape_ref[1], height=shape_ref[0], resampleAlg='max')
        water = water_ds.GetRasterBand(1).ReadAsArray()
        # Correct Fix: Mask water with landmask to ensure distance to INLAND water only
        water_mask = (water > 50) & land_mask
        dist_w_px = distance_transform_edt(~water_mask)
        dist_w_km = (dist_w_px * gt_ref[1] * 111.12).astype(np.float32)
        # Apply Landmask to result to clean ocean/nodata areas
        dist_w_km[~land_mask] = -9999
        create_raster(dist_w_km, os.path.join(OUTPUT_DIR, 'dist_water.tif'), gt_ref, proj_ref)

    # --- D. Local Context (NaN-Aware 1km NDVI) ---
    log_message("Computing Landscape Context (NaN-Aware 1km NDVI)...")
    ndvi_path = os.path.join(BASE_DIR, 'PHASE2', 'yearly_aggregates', 'NDVI_mean_2024.tif')
    if os.path.exists(ndvi_path):
        ndvi_ds = gdal.Open(ndvi_path)
        ndvi = ndvi_ds.GetRasterBand(1).ReadAsArray().astype(float)
        ndvi[ndvi == ndvi_ds.GetRasterBand(1).GetNoDataValue()] = np.nan
        
        from scipy.ndimage import generic_filter
        local_mean = generic_filter(ndvi, np.nanmean, size=3)
        local_std = generic_filter(ndvi, np.nanstd, size=3)
        
        # Cleanup artifacts
        local_mean[~np.isfinite(local_mean)] = -9999
        local_std[~np.isfinite(local_std)] = -9999
        
        create_raster(local_mean, os.path.join(OUTPUT_DIR, 'local_ndvi_mean_1km.tif'), gt_ref, proj_ref)
        create_raster(local_std, os.path.join(OUTPUT_DIR, 'local_ndvi_std_1km.tif'), gt_ref, proj_ref)

    log_message(f"Raster Factory Complete. Static layers saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_rasters()
