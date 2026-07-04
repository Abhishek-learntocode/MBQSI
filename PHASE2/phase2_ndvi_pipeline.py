import os
import glob
import time
import logging
import psutil
import shutil
import numpy as np
import pandas as pd
import json
import traceback
import concurrent.futures
from datetime import datetime
from osgeo import gdal, osr
import matplotlib.pyplot as plt
import calendar
import warnings
from tqdm import tqdm

gdal.SetCacheMax(512 * 1024 * 1024)

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")
warnings.filterwarnings("ignore", category=RuntimeWarning)

gdal.UseExceptions()

# ---------------- CONFIG ----------------
STAGE_A_LIMIT = None
DELETE_DAILY_MOSAICS_AFTER_AGG = True
CHUNK_SIZE = 10
RAM_SAFE_LIMIT_PERCENT = 90.0

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SOURCE_HDF_FOLDER = r'C:\Users\Abhishek\IIITH\IITH\INDEPENDENT_STUDY\DATASET\VEGITATION_AND_FOOD\NASA_YEAR_WISE'
PHASE2_DIR = os.path.join(BASE_DIR, 'PHASE2')
ANI_BOUNDARY_PATH = os.path.join(BASE_DIR, 'PHASE3', 'PHASE3.1', 'ANI_boundary.gpkg')

DIRS = {
    'temp': os.path.join(PHASE2_DIR, 'temp'),
    'mosaics': os.path.join(PHASE2_DIR, 'yearly_mosaics'),
    'aggregates': os.path.join(PHASE2_DIR, 'yearly_aggregates'),
    'validation': os.path.join(PHASE2_DIR, 'validation'),
    'val_logs': os.path.join(PHASE2_DIR, 'validation', 'logs'),
    'val_csv': os.path.join(PHASE2_DIR, 'validation', 'csv'),
    'val_txt': os.path.join(PHASE2_DIR, 'validation', 'txt'),
    'val_plots': os.path.join(PHASE2_DIR, 'validation', 'plots'),
    'val_previews': os.path.join(PHASE2_DIR, 'validation', 'plots', 'raster_previews'),
    'metadata': os.path.join(PHASE2_DIR, 'metadata'),
    'scripts': os.path.join(PHASE2_DIR, 'scripts')
}

NDVI_SCALE = 0.0001
NODATA = -9999
QA_MASK = 0b11

class ValidationFailError(Exception):
    pass

def get_subdataset(hdf, keyword):
    try:
        ds = gdal.Open(hdf)
        if ds is None: return None
        for name, desc in ds.GetSubDatasets():
            if keyword in desc:
                return name
    except:
        pass
    return None

def process_date_group(date_str, hdf_files, dirs_dict, ani_boundary, nodata_val, scale_val, qa_mask_val):
    """
    Top-level function for ProcessPoolExecutor.
    Processes all HDF tiles for a given date, mosaics them, and computes valid ratios.
    """
    meta = {
        'date_str': date_str,
        'year': int(date_str[1:5]),
        'month': pd.to_datetime(date_str[1:], format='%Y%j').month,
        'tiles_expected': len(hdf_files),
        'tiles_processed': 0,
        'valid_pixel_ratio': 0,
        'ndvi_mean': 0.0,
        'ndvi_std': 0.0,
        'nodata_pct': 100.0,
        'status': 'FAILED',
        'severity': 'FAIL',
        'error_category': 'NONE',
        'message': 'Unknown error',
        'traceback': ''
    }
    
    temp_files = []
    mosaic_path = os.path.join(dirs_dict['mosaics'], f"ANI_mosaic_{date_str}.tif")
    
    try:
        # Step 1: Extract and QA mask each tile individually
        for hdf in hdf_files:
            fname = os.path.basename(hdf)
            ndvi_sds = get_subdataset(hdf, 'NDVI')
            qa_sds = get_subdataset(hdf, 'VI Quality')
            
            if not ndvi_sds or not qa_sds:
                raise ValidationFailError(f"Missing Subdatasets in {fname}")
                
            ndvi_ds = gdal.Open(ndvi_sds)
            qa_ds = gdal.Open(qa_sds)
            
            ndvi_arr = ndvi_ds.ReadAsArray().astype(np.float32)
            qa_arr = qa_ds.ReadAsArray()
            
            quality = qa_arr & qa_mask_val
            ndvi_arr *= scale_val
            
            ndvi_arr[quality == 3] = nodata_val
            ndvi_arr[ndvi_arr < -0.2] = nodata_val
            
            valid_test = ndvi_arr[ndvi_arr != nodata_val]
            if valid_test.size > 0:
                if np.max(valid_test) > 1.0 or np.min(valid_test) < -1.0:
                    raise ValidationFailError(f"Numeric corruption in {fname} [-1,1] violated")
            del valid_test
            del qa_arr
            del quality

            temp_tif = os.path.join(dirs_dict['temp'], f"temp_{fname}.tif")
            temp_files.append(temp_tif)
            
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(temp_tif, ndvi_ds.RasterXSize, ndvi_ds.RasterYSize, 1, gdal.GDT_Float32)
            out_ds.SetProjection(ndvi_ds.GetProjection())
            out_ds.SetGeoTransform(ndvi_ds.GetGeoTransform())
            band = out_ds.GetRasterBand(1)
            band.WriteArray(ndvi_arr)
            band.SetNoDataValue(nodata_val)
            band.FlushCache()
            out_ds.FlushCache()
            
            out_ds = None
            band = None
            ndvi_ds = None
            qa_ds = None
            meta['tiles_processed'] += 1

        if len(temp_files) == 0:
            raise ValidationFailError("No valid tiles extracted for mosaicing.")

        # Step 2: Mosaic & Crop
        gdal.Warp(
            mosaic_path, temp_files, format='GTiff',
            dstSRS='EPSG:4326', cutlineDSName=ani_boundary, cropToCutline=True,
            dstNodata=nodata_val, resampleAlg='near', multithread=True,
            creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'PREDICTOR=2']
        )
        
        # Step 3: Validate Mosaic
        mos_ds = gdal.Open(mosaic_path)
        if mos_ds is None:
            raise ValidationFailError("Warp completed but resulting mosaic is unreadable.")
            
        arr = mos_ds.GetRasterBand(1).ReadAsArray()
        mos_ds = None
        
        valid_pixels = arr[arr != nodata_val]
        total_pixels = arr.size
        post_nodata_pct = 100.0 - ((valid_pixels.size / total_pixels) * 100)
        valid_ratio = valid_pixels.size / total_pixels
        
        meta['nodata_pct'] = float(post_nodata_pct)
        meta['valid_pixel_ratio'] = float(valid_ratio)
        
        if post_nodata_pct >= 99.9 and valid_pixels.size < 500:
            meta['status'] = 'EXPECTED_EMPTY'
            meta['severity'] = 'INFO'
            meta['message'] = 'Geographic exclusion: mostly ocean/no ANI overlap'
            # Delete empty mosaic to save space
            try: os.remove(mosaic_path)
            except: pass
        else:
            if valid_pixels.size == 0:
                raise ValidationFailError("Zero valid pixels after clipping (unexpected).")
                
            meta['ndvi_mean'] = float(np.mean(valid_pixels))
            meta['ndvi_std'] = float(np.std(valid_pixels))
            
            if valid_pixels.size < 10000:
                meta['status'] = 'SKIPPED_VERY_LOW'
                meta['severity'] = 'INFO'
                meta['message'] = 'Skipped due to excessive clouds on land'
            elif valid_pixels.size < 50000:
                meta['status'] = 'LOW_COVERAGE_USED'
                meta['severity'] = 'WARNING'
                meta['message'] = 'Low land coverage'
            elif valid_pixels.size < 100000:
                meta['status'] = 'MEDIUM_COVERAGE'
                meta['severity'] = 'INFO'
                meta['message'] = 'Acceptable coverage'
            else:
                meta['status'] = 'HIGH_COVERAGE'
                meta['severity'] = 'INFO'
                meta['message'] = 'High coverage'

    except ValidationFailError as e:
        meta['severity'] = 'FAIL'
        meta['error_category'] = 'GIS_FAILURE'
        meta['message'] = str(e)
        meta['traceback'] = traceback.format_exc()
        try: os.remove(mosaic_path)
        except: pass
    except Exception as e:
        meta['severity'] = 'FAIL'
        err_str = str(e).lower()
        if 'tiff' in err_str or 'fillemptytiles' in err_str:
            meta['error_category'] = 'WARP_FAILURE'
        elif 'memory' in err_str:
            meta['error_category'] = 'RESOURCE_FAILURE'
        else:
            meta['error_category'] = 'READ_FAILURE'
        meta['message'] = str(e)
        meta['traceback'] = traceback.format_exc()
        try: os.remove(mosaic_path)
        except: pass
    finally:
        for t in temp_files:
            try: os.remove(t)
            except: pass
            
    return meta

class Phase2Pipeline:
    def __init__(self):
        # Clean state
        for key in ['mosaics', 'aggregates', 'validation', 'metadata', 'temp']:
            if os.path.exists(DIRS[key]):
                shutil.rmtree(DIRS[key], ignore_errors=True)
                
        for path in DIRS.values():
            os.makedirs(path, exist_ok=True)
            
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            filename=os.path.join(DIRS['val_logs'], 'runtime_log.txt'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.console = logging.StreamHandler()
        self.console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(self.console)

        # Init CSVs
        pd.DataFrame(columns=['date_str','year','month','tiles_expected','tiles_processed','valid_pixel_ratio','ndvi_mean','ndvi_std','nodata_pct','status','severity','message']).to_csv(os.path.join(DIRS['val_csv'], 'daily_mosaic_validation.csv'), index=False)
        pd.DataFrame(columns=['date_str','error_category','message','traceback']).to_csv(os.path.join(DIRS['val_csv'], 'failed_mosaics.csv'), index=False)
        pd.DataFrame(columns=['year','feature','min','max','mean','std','nodata_pct','valid_pixel_count']).to_csv(os.path.join(DIRS['val_csv'], 'aggregate_validation.csv'), index=False)
        pd.DataFrame(columns=['year','feature','min','max','mean','std','nodata_pct','valid_pixel_count']).to_csv(os.path.join(DIRS['val_csv'], 'seasonal_validation.csv'), index=False)
        pd.DataFrame(columns=['year','monsoon_dates','dry_dates','winter_dates','monsoon_ratio','dry_ratio','winter_ratio']).to_csv(os.path.join(DIRS['val_csv'], 'seasonal_completeness.csv'), index=False)
        pd.DataFrame(columns=['year','completeness_ratio','total_valid_mosaics','mean_ndvi','mean_cv','mean_peak_seasonality','weak_year_flag']).to_csv(os.path.join(DIRS['val_csv'], 'yearly_ecological_summary.csv'), index=False)

    def log(self, message, level=logging.INFO):
        logging.log(level, message)
        
    def check_ram(self):
        if psutil.virtual_memory().percent > RAM_SAFE_LIMIT_PERCENT:
            self.log(f"RAM Usage critical. Pausing...", logging.WARNING)
            time.sleep(10)

    def generate_yearly_aggregates(self):
        self.log("[*] Generating Temporal Aggregates from Valid Mosaics...")
        
        df = pd.read_csv(os.path.join(DIRS['val_csv'], 'daily_mosaic_validation.csv'))
        usable = df[df['status'].isin(['HIGH_COVERAGE', 'MEDIUM_COVERAGE', 'LOW_COVERAGE_USED', 'SKIPPED_VERY_LOW'])]
        
        ref_grid = None
        ref_path = os.path.join(DIRS['metadata'], 'reference_grid.json')
        
        for year, group in usable.groupby('year'):
            mosaics = []
            for _, row in group.iterrows():
                path = os.path.join(DIRS['mosaics'], f"ANI_mosaic_{row['date_str']}.tif")
                if os.path.exists(path):
                    mosaics.append(path)
                    
            if not mosaics:
                self.log(f"No valid mosaics found for year {year}", logging.WARNING)
                continue
                
            # strict alignment check
            valid_mosaics = []
            for m in mosaics:
                ds = gdal.Open(m)
                if not ds: continue
                proj = ds.GetProjection()
                geo = ds.GetGeoTransform()
                xs = ds.RasterXSize
                ys = ds.RasterYSize
                ds = None
                
                srs = osr.SpatialReference(wkt=proj)
                epsg = srs.GetAttrValue('AUTHORITY', 1)
                
                if ref_grid is None:
                    ref_grid = {'proj': proj, 'epsg': epsg, 'geo': geo, 'xs': xs, 'ys': ys}
                    with open(ref_path, 'w') as f: json.dump(ref_grid, f)
                else:
                    if xs != ref_grid['xs'] or ys != ref_grid['ys']:
                        self.log(f"Shape mismatch in {m}! Expected {ref_grid['xs']}x{ref_grid['ys']}. Failing fast.", logging.ERROR)
                        continue
                    if epsg != ref_grid['epsg']:
                        self.log(f"Projection mismatch in {m}! EPSG {epsg} != {ref_grid['epsg']}. Failing fast.", logging.ERROR)
                        continue
                    if not np.allclose(geo, ref_grid['geo'], atol=1e-9):
                        self.log(f"GeoTransform mismatch in {m}! Failing fast.", logging.ERROR)
                        continue
                valid_mosaics.append(m)
                
            if not valid_mosaics: continue
            
            if len(valid_mosaics) < 3:
                self.log(f"CRITICAL: Year {year} has insufficient mosaics ({len(valid_mosaics)}). Fails fast.", logging.ERROR)
                continue
            
            EXPECTED_COMPOSITES = 23
            completeness_ratio = len(valid_mosaics) / EXPECTED_COMPOSITES
            if completeness_ratio < 0.5:
                self.log(f"WARNING: Year {year} is a WEAK_YEAR with completeness {completeness_ratio:.2f} ({len(valid_mosaics)}/{EXPECTED_COMPOSITES}). Ecological stability may be compromised.", logging.WARNING)
                
            arrays = []
            arrays_monsoon = []
            arrays_dry = []
            arrays_winter = []
            import datetime
            
            num_dry = 0
            num_monsoon = 0
            num_winter = 0
            for m in valid_mosaics:
                ds = gdal.Open(m)
                arr = ds.GetRasterBand(1).ReadAsArray()
                arr[arr == NODATA] = np.nan
                arrays.append(arr)
                
                # Parse date to month for seasonal segregation
                basename = os.path.basename(m)
                date_str = basename.split('_')[2].split('.')[0]
                try:
                    year_val = int(date_str[1:5])
                    doy_val = int(date_str[5:8])
                    month = datetime.datetime.strptime(f"{year_val}-{doy_val}", "%Y-%j").month
                    if month in [2, 3, 4, 5]:
                        arrays_dry.append(arr)
                        num_dry += 1
                    elif month in [6, 7, 8, 9]:
                        arrays_monsoon.append(arr)
                        num_monsoon += 1
                    else:
                        arrays_winter.append(arr)
                        num_winter += 1
                except Exception:
                    pass
                    
                ds = None
                
            total_dates = len(valid_mosaics)
            if total_dates == 0: continue
            
            pd.DataFrame([{
                'year': year,
                'monsoon_dates': num_monsoon,
                'dry_dates': num_dry,
                'winter_dates': num_winter,
                'monsoon_ratio': num_monsoon / total_dates,
                'dry_ratio': num_dry / total_dates,
                'winter_ratio': num_winter / total_dates
            }]).to_csv(os.path.join(DIRS['val_csv'], 'seasonal_completeness.csv'), mode='a', header=False, index=False)
                
            stack = np.stack(arrays, axis=0)
            del arrays
            
            with np.errstate(all='ignore'):
                agg_mean = np.nanmean(stack, axis=0)
                agg_std = np.nanstd(stack, axis=0)
                agg_min = np.nanmin(stack, axis=0)
                agg_max = np.nanmax(stack, axis=0)
                agg_range = agg_max - agg_min
                valid_count = np.sum(~np.isnan(stack), axis=0).astype(np.float32)
                
                # Seasonal calculations
                def safe_seasonal_stats(arr_list):
                    if arr_list:
                        s = np.stack(arr_list, axis=0)
                        mean_arr = np.nanmean(s, axis=0)
                        std_arr = np.nanstd(s, axis=0)
                        del s
                        return mean_arr, std_arr
                    else:
                        return np.full_like(agg_mean, np.nan), np.full_like(agg_mean, np.nan)
                        
                monsoon_mean, monsoon_std = safe_seasonal_stats(arrays_monsoon)
                dry_mean, dry_std = safe_seasonal_stats(arrays_dry)
                winter_mean, _ = safe_seasonal_stats(arrays_winter)
                
                monsoon_minus_dry = monsoon_mean - dry_mean
                seasonal_means = np.stack([monsoon_mean, dry_mean, winter_mean], axis=0)
                with np.errstate(all='ignore'):
                    peak_seasonality = np.nanmax(seasonal_means, axis=0) - np.nanmin(seasonal_means, axis=0)
                
                del seasonal_means
                seasonal_arrays = [monsoon_mean, dry_mean, winter_mean, monsoon_std, dry_std, monsoon_minus_dry, peak_seasonality]
                
                # Finite value cleaning & Negative NDVI exclusion
                invalid_mask = (~np.isfinite(agg_mean)) | (agg_mean < 0)
                agg_mean[invalid_mask] = np.nan
                for s_arr in seasonal_arrays:
                    s_arr[invalid_mask] = np.nan
                
                # Ecological mask for valid count
                valid_count[invalid_mask] = np.nan
                
                # Stable CV masking
                cv = np.full_like(agg_mean, np.nan, dtype=np.float32)
                valid_cv_mask = (~np.isnan(agg_mean)) & (~np.isnan(agg_std)) & (agg_mean > 0.05)
                cv[valid_cv_mask] = agg_std[valid_cv_mask] / agg_mean[valid_cv_mask]
                cv[cv > 3] = np.nan
                
                # Minimum valid observation threshold
                mask = valid_count < 3
                
                # Weak year full suppression
                if completeness_ratio < 0.5:
                    mask = np.ones_like(valid_count, dtype=bool)
                    
                agg_mean[mask] = np.nan
                agg_std[mask] = np.nan
                agg_min[mask] = np.nan
                agg_max[mask] = np.nan
                agg_range[mask] = np.nan
                cv[mask] = np.nan
                valid_count[mask] = np.nan
                for s_arr in seasonal_arrays:
                    s_arr[mask] = np.nan

            # Yearly ecological summary (Outside numerical block for cleaner semantics)
            eco_sum = {
                'year': year,
                'completeness_ratio': completeness_ratio,
                'total_valid_mosaics': len(valid_mosaics),
                'mean_ndvi': float(np.nanmean(agg_mean)),
                'mean_cv': float(np.nanmean(cv)),
                'mean_peak_seasonality': float(np.nanmean(peak_seasonality)),
                'weak_year_flag': 1 if completeness_ratio < 0.5 else 0
            }
            pd.DataFrame([eco_sum]).to_csv(os.path.join(DIRS['val_csv'], 'yearly_ecological_summary.csv'), mode='a', header=False, index=False)

            del stack
            del arrays_monsoon
            del arrays_dry
            del arrays_winter
                
            features = {
                'NDVI_mean': agg_mean,
                'NDVI_std': agg_std,
                'NDVI_min': agg_min,
                'NDVI_max': agg_max,
                'NDVI_temporal_range': agg_range,
                'NDVI_valid_count': valid_count,
                'NDVI_cv': cv,
                'NDVI_monsoon_mean': monsoon_mean,
                'NDVI_dry_mean': dry_mean,
                'NDVI_winter_mean': winter_mean,
                'NDVI_monsoon_std': monsoon_std,
                'NDVI_dry_std': dry_std,
                'NDVI_monsoon_minus_dry': monsoon_minus_dry,
                'NDVI_peak_seasonality': peak_seasonality
            }
            
            for name, arr in features.items():
                arr[~np.isfinite(arr)] = np.nan
                
                # Extract scientific validation metrics before masking NaN
                valid_px = arr[~np.isnan(arr)]
                strict_ndvi_features = [
                    'NDVI_mean', 'NDVI_std', 'NDVI_min', 'NDVI_max',
                    'NDVI_monsoon_mean', 'NDVI_dry_mean', 'NDVI_winter_mean',
                    'NDVI_monsoon_std', 'NDVI_dry_std'
                ]
                if valid_px.size > 0 and name in strict_ndvi_features:
                    if np.min(valid_px) < -1.0 or np.max(valid_px) > 1.0:
                        raise ValueError(f"{name} ecological NDVI bounds violated: min={np.min(valid_px)}, max={np.max(valid_px)}")
                        
                agg_stats = {
                    'year': year,
                    'feature': name,
                    'min': float(np.min(valid_px)) if valid_px.size > 0 else np.nan,
                    'max': float(np.max(valid_px)) if valid_px.size > 0 else np.nan,
                    'mean': float(np.mean(valid_px)) if valid_px.size > 0 else np.nan,
                    'std': float(np.std(valid_px)) if valid_px.size > 0 else np.nan,
                    'nodata_pct': 100.0 - ((valid_px.size / arr.size) * 100) if arr.size > 0 else 100.0,
                    'valid_pixel_count': valid_px.size
                }
                
                csv_file = 'seasonal_validation.csv' if any(s in name for s in ['monsoon', 'dry', 'winter', 'seasonality']) else 'aggregate_validation.csv'
                pd.DataFrame([agg_stats]).to_csv(os.path.join(DIRS['val_csv'], csv_file), mode='a', header=False, index=False)
                
                arr[np.isnan(arr)] = NODATA
                out_path = os.path.join(DIRS['aggregates'], f"{name}_{int(year)}.tif")
                
                dtype = gdal.GDT_Int16 if name == 'NDVI_valid_count' else gdal.GDT_Float32
                if name == 'NDVI_valid_count':
                    arr = arr.astype(np.int16)
                    
                driver = gdal.GetDriverByName('GTiff')
                out_ds = driver.Create(out_path, ref_grid['xs'], ref_grid['ys'], 1, dtype, options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'PREDICTOR=2'])
                out_ds.SetGeoTransform(ref_grid['geo'])
                out_ds.SetProjection(ref_grid['proj'])
                band = out_ds.GetRasterBand(1)
                band.WriteArray(arr)
                band.SetNoDataValue(NODATA)
                band.FlushCache()
                band = None
                out_ds = None
                
        
        # Lightweight raster integrity check
        integrity_log = "RASTER INTEGRITY REVIEW\n" + "="*30 + "\n"
        agg_files = glob.glob(os.path.join(DIRS['aggregates'], '*.tif'))
        for f in agg_files:
            ds = gdal.Open(f)
            if not ds:
                integrity_log += f"{os.path.basename(f)}: FAILED (Unreadable)\n"
                continue
            band = ds.GetRasterBand(1)
            nodata = band.GetNoDataValue()
            if nodata is None:
                integrity_log += f"{os.path.basename(f)}: FAILED (Missing NoData)\n"
            else:
                integrity_log += f"{os.path.basename(f)}: PASS\n"
            ds = None
        with open(os.path.join(DIRS['val_txt'], 'raster_integrity_review.txt'), 'w') as f:
            f.write(integrity_log)
            
        # File count validation
        expected_tiffs = len(usable['year'].unique()) * 14
        with open(os.path.join(DIRS['val_txt'], 'output_inventory_report.txt'), 'w') as f:
            f.write(f"Total TIFFs Generated: {len(agg_files)}\nExpected: ~{expected_tiffs}\n")
            
        if DELETE_DAILY_MOSAICS_AFTER_AGG:
            self.log("[*] Cleaning up daily mosaics to save space...")
            shutil.rmtree(DIRS['mosaics'], ignore_errors=True)
            os.makedirs(DIRS['mosaics'], exist_ok=True)

    def generate_diagnostics(self):
        df = pd.read_csv(os.path.join(DIRS['val_csv'], 'daily_mosaic_validation.csv'))
        
        analysis = f"FAILURE ROOT-CAUSE ANALYSIS\n{'='*30}\n"
        fails = pd.read_csv(os.path.join(DIRS['val_csv'], 'failed_mosaics.csv'))
        analysis += f"True Pipeline Crashes: {len(fails)}\n"
        if len(fails) > 0:
            analysis += fails['error_category'].value_counts().to_string() + "\n"
        with open(os.path.join(DIRS['val_txt'], 'failure_rootcause_analysis.txt'), 'w') as f:
            f.write(analysis)
            
        review = f"ECOLOGICAL NDVI SANITY REVIEW\n{'='*30}\n"
        review += "Checking mathematically bounded features in valid mosaics:\n"
        review += f"Total Valid Mosaics: {len(df[df['severity'] != 'FAIL'])}\n"
        review += f"Global Mean NDVI (Island Landmass): {df[df['severity'] != 'FAIL']['ndvi_mean'].mean():.3f}\n"
        with open(os.path.join(DIRS['val_txt'], 'ecological_ndvi_review.txt'), 'w') as f:
            f.write(review)
            
        content = f"""STAGE B FULL PRODUCTION DASHBOARD
===================================
Total Dates Encountered: {len(df)}
Expected Empty (Ocean/No Overlap): {len(df[df['status'] == 'EXPECTED_EMPTY'])}
True Pipeline Crashes: {len(fails)}

MOSAIC QUALITY DISTRIBUTION:
{df['status'].value_counts().to_string()}

COMPUTATIONAL SAFETY:
Temp directory fully cleaned. Alignment enforced via reference_grid.json.
See aggregate_validation.csv and seasonal_validation.csv for final mathematical audit of features.

FEATURES EXTRACTED:
NDVI_mean, NDVI_std, NDVI_min, NDVI_max, NDVI_temporal_range, NDVI_cv, NDVI_valid_count
NDVI_monsoon_mean, NDVI_dry_mean, NDVI_winter_mean, NDVI_monsoon_std, NDVI_dry_std
NDVI_monsoon_minus_dry, NDVI_peak_seasonality
"""
        with open(os.path.join(DIRS['val_txt'], 'stageB_master_summary.txt'), 'w') as f:
            f.write(content)

    def generate_previews(self):
        self.log("[*] Generating Aggregate Previews...")
        agg_files = glob.glob(os.path.join(DIRS['aggregates'], '*.tif'))
        
        # Only preview a subset to save time
        preview_targets = ['NDVI_mean', 'NDVI_cv', 'NDVI_monsoon_mean', 'NDVI_valid_count']
        years = sorted(list(set([f.split('_')[-1].replace('.tif', '') for f in agg_files])))
        preview_years = years[:3]
        
        for f in agg_files:
            basename = os.path.basename(f)
            year = basename.split('_')[-1].replace('.tif', '')
            if year not in preview_years:
                continue
            if not any(target in basename for target in preview_targets):
                continue
                
            ds = gdal.Open(f)
            arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
            arr[arr == NODATA] = np.nan
            plt.figure(figsize=(8,8))
            plt.imshow(arr, cmap='RdYlGn' if 'mean' in f else 'plasma')
            plt.title(f"{os.path.basename(f)}")
            plt.colorbar()
            plt.savefig(os.path.join(DIRS['val_previews'], f"preview_{os.path.basename(f).replace('.tif','.png')}"))
            plt.close()
            del arr
            ds = None

    def run_stage_b(self):
        start_time = time.time()
        self.log("="*50)
        self.log("PHASE 2 NDVI PIPELINE - FULL PRODUCTION STAGE B")
        self.log("="*50)
        
        all_files = glob.glob(os.path.join(SOURCE_HDF_FOLDER, '**', '*.hdf'), recursive=True)
        if STAGE_A_LIMIT is None:
            files = all_files
        else:
            files = all_files[:STAGE_A_LIMIT]
        
        date_groups = {}
        for f in files:
            d = os.path.basename(f).split('.')[1]
            if d not in date_groups: date_groups[d] = []
            date_groups[d].append(f)
            
        self.log(f"Grouped {len(files)} files into {len(date_groups)} distinct dates.")
        
        # ProcessPoolExecutor for thread-safe GDAL isolation
        max_workers = min(6, max(2, os.cpu_count() // 2))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_date_group, d, tiles, DIRS, ANI_BOUNDARY_PATH, NODATA, NDVI_SCALE, QA_MASK): d for d, tiles in date_groups.items()}
            
            with tqdm(total=len(futures), desc="Mosaicing Dates") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    self.check_ram()
                    try:
                        res = future.result()
                        row = {k: res[k] for k in ['date_str','year','month','tiles_expected','tiles_processed','valid_pixel_ratio','ndvi_mean','ndvi_std','nodata_pct','status','severity','message']}
                        pd.DataFrame([row]).to_csv(os.path.join(DIRS['val_csv'], 'daily_mosaic_validation.csv'), mode='a', header=False, index=False)
                        if res['severity'] == 'FAIL':
                            pd.DataFrame([{'date_str': res['date_str'], 'error_category': res['error_category'], 'message': res['message'], 'traceback': res['traceback']}]).to_csv(os.path.join(DIRS['val_csv'], 'failed_mosaics.csv'), mode='a', header=False, index=False)
                    except Exception as e:
                        self.log(f"Unexpected Process crash: {e}", logging.ERROR)
                    pbar.update(1)
                    
        self.generate_yearly_aggregates()
        self.generate_diagnostics()
        self.generate_previews()
        
        runtime = time.time() - start_time
        manifest = {
            'runtime_seconds': runtime,
            'total_files': len(files),
            'unique_dates': len(date_groups),
            'gdal_version': gdal.VersionInfo(),
            'features_extracted': [
                'NDVI_mean', 'NDVI_std', 'NDVI_min', 'NDVI_max',
                'NDVI_temporal_range', 'NDVI_valid_count', 'NDVI_cv',
                'NDVI_monsoon_mean', 'NDVI_dry_mean', 'NDVI_winter_mean',
                'NDVI_monsoon_std', 'NDVI_dry_std',
                'NDVI_monsoon_minus_dry', 'NDVI_peak_seasonality'
            ]
        }
        with open(os.path.join(DIRS['metadata'], 'phase2_manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=4)
            
        self.log(f"STAGE B COMPLETE in {runtime:.2f} seconds.")
        self.log("Review PHASE2/validation/txt/stageB_master_summary.txt for Final Go/No-Go decision.")

if __name__ == "__main__":
    pipeline = Phase2Pipeline()
    pipeline.run_stage_b()
