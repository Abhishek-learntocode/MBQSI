import os
import glob
import pandas as pd
import numpy as np
import time
import gc
import re
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from scipy.stats import entropy

class MBSQIPhase1Pipeline:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.df = pd.DataFrame()
        self.attrition_log = []
        self.manifest = {
            "execution_timestamp": datetime.now().isoformat(),
            "input_directory": input_dir,
            "output_directory": output_dir,
            "thresholds_used": {
                "max_missing_pct": 50.0,
                "year_range": [2005, 2025],
                "ani_bounds": {"lat_min": 6.0, "lat_max": 14.0, "lon_min": 92.0, "lon_max": 94.5},
                "min_records_migratory": 50,
                "max_summer_ratio": 0.05,
                "max_grid_density": 4,
                "min_species_retention_pct": 90.0
            },
            "record_counts": {},
            "species_counts": {}
        }
        os.makedirs(self.output_dir, exist_ok=True)

    def log_print(self, message):
        try:
            print(message)
        except UnicodeEncodeError:
            print(message.encode('ascii','replace').decode('ascii'))

    def track_attrition(self, step_name):
        num_records = len(self.df)
        num_species = self.df['species_name'].nunique() if 'species_name' in self.df.columns else 0
        self.attrition_log.append({
            "step": step_name,
            "records": num_records,
            "species": num_species
        })
        self.manifest["record_counts"][step_name] = num_records
        self.manifest["species_counts"][step_name] = num_species

    def step1_ingestion_and_harmonization(self):
        self.log_print("[*] STEP 1: Ingestion & Schema Harmonization...")
        search_pattern = os.path.join(self.input_dir, '**', '*.csv')
        all_files = glob.glob(search_pattern, recursive=True)
        
        schema_audit_path = os.path.join(self.output_dir, 'schema_audit.txt')
        master_columns = set()
        df_list = []
        
        schema_mapping = {
            'decimallatitude': 'latitude', 'locationlat': 'latitude', 'lat': 'latitude', 'start_latitude': 'latitude',
            'decimallongitude': 'longitude', 'locationlon': 'longitude', 'lon': 'longitude', 'start_longitude': 'longitude',
            'eventdate': 'observation_date', 'observed_on': 'observation_date', 'time_observed_at': 'observation_date', 
            'date_time': 'observation_date', 'date': 'observation_date', 'timestamp': 'observation_date',
            'scientificname': 'species_name', 'scientific_name': 'species_name', 'species': 'species_name', 
            'common_name': 'species_name', 'taxon': 'species_name', 'bird_name': 'species_name',
            'gbifid': 'occurrence_id', 'occurrenceid': 'occurrence_id', 'id': 'occurrence_id', 
            'catalog': 'occurrence_id', 'recordnumber': 'occurrence_id',
            'coordinateuncertaintyinmeters': 'uncertainty_m', 'positional_accuracy': 'uncertainty_m', 'accuracy': 'uncertainty_m',
            'elevation': 'elevation_m', 'altitude': 'elevation_m', 'verbatimelevation': 'elevation_m', 'depth': 'elevation_m',
            'recordedby': 'observer_name', 'user_login': 'observer_name', 'observer': 'observer_name'
        }
        
        with open(schema_audit_path, 'w', encoding='utf-8') as f:
            f.write("MBSQI RAW DATA SCHEMA AUDIT\n")
            for file_path in all_files:
                filename = os.path.basename(file_path)
                try:
                    temp_df = pd.read_csv(file_path, low_memory=False)
                    if temp_df.empty: continue
                    
                    raw_cols = list(temp_df.columns)
                    lower_cols = [str(c).lower().strip() for c in raw_cols]
                    master_columns.update(lower_cols)
                    
                    seen = set()
                    dupes = set(x for x in lower_cols if x in seen or seen.add(x))
                    
                    f.write(f"FILE: {filename} | COLS: {len(raw_cols)}\n")
                    if dupes: f.write(f"  WARNING DUPLICATES: {dupes}\n")
                    
                    temp_df.columns = lower_cols
                    temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()].copy()
                    temp_df.rename(columns=schema_mapping, inplace=True)
                    temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()].copy()
                    temp_df = temp_df.dropna(axis=1, how='all')
                    
                    float_cols = temp_df.select_dtypes(include=['float64']).columns
                    temp_df[float_cols] = temp_df[float_cols].apply(pd.to_numeric, downcast='float')
                    int_cols = temp_df.select_dtypes(include=['int64', 'Int64']).columns
                    temp_df[int_cols] = temp_df[int_cols].apply(pd.to_numeric, downcast='integer')
                    
                    temp_df['source_file'] = filename
                    df_list.append(temp_df)
                    del temp_df
                    gc.collect()
                except Exception as e:
                    f.write(f"FILE: {filename} ERROR: {e}\n")
        
        if not df_list: raise ValueError("No data found to ingest.")
        self.df = pd.concat(df_list, axis=0, ignore_index=True)
        del df_list
        gc.collect()
        self.track_attrition("1_Ingestion")
        self.log_print(f"    -> {len(self.df)} records ingested.")

    def step2_feature_density_and_observer_bias(self):
        self.log_print("[*] STEP 2: Feature Density & Observer Bias Optimization...")
        
        # Missing values
        total_rows = len(self.df)
        missing_pct = (self.df.isna().sum() / total_rows) * 100
        
        missing_report = pd.DataFrame({'column_name': missing_pct.index, 'missing_percent': missing_pct.values})
        protected_cols = ['latitude', 'longitude', 'species_name', 'observation_date', 'year', 'month', 'occurrence_id', 'source_file', 'observer_name']
        missing_report['retained_or_removed'] = missing_report.apply(
            lambda row: 'retained' if row['missing_percent'] <= 50.0 or row['column_name'] in protected_cols else 'removed', axis=1
        )
        missing_report.to_csv(os.path.join(self.output_dir, 'missing_value_report.csv'), index=False)
        
        cols_to_keep = missing_report[missing_report['retained_or_removed'] == 'retained']['column_name'].tolist()
        self.df = self.df[cols_to_keep]

        # Observer bias
        if 'observer_name' in self.df.columns:
            obs_counts = self.df['observer_name'].value_counts()
            top_1_pct = max(1, int(len(obs_counts) * 0.01))
            top_5_pct = max(1, int(len(obs_counts) * 0.05))
            top_10_pct = max(1, int(len(obs_counts) * 0.10))
            
            obs_stats = {
                "total_observers": len(obs_counts),
                "top_1_pct_contribution": float(obs_counts.head(top_1_pct).sum() / total_rows),
                "top_5_pct_contribution": float(obs_counts.head(top_5_pct).sum() / total_rows),
                "top_10_pct_contribution": float(obs_counts.head(top_10_pct).sum() / total_rows)
            }
            obs_df = obs_counts.reset_index()
            obs_df.columns = ['observer_name', 'records']
            obs_df.to_csv(os.path.join(self.output_dir, 'observer_bias_report.csv'), index=False)
            
            with open(os.path.join(self.output_dir, 'observer_stats.json'), 'w') as f:
                json.dump(obs_stats, f, indent=4)
        
        self.track_attrition("2_FeatureDensity")
        self.log_print(f"    -> {len(self.df.columns)} columns retained.")

    def step3_temporal_standardization(self):
        self.log_print("[*] STEP 3: Temporal Standardization & Bias Validation...")
        if 'observation_date' in self.df.columns:
            self.df['observation_date'] = pd.to_datetime(self.df['observation_date'], errors='coerce')
            if 'year' not in self.df.columns: self.df['year'] = np.nan
            if 'month' not in self.df.columns: self.df['month'] = np.nan
            self.df['year'] = self.df['year'].fillna(self.df['observation_date'].dt.year)
            self.df['month'] = self.df['month'].fillna(self.df['observation_date'].dt.month)
        
        self.df = self.df.dropna(subset=['year', 'month'])
        
        min_year_before = int(self.df['year'].min())
        max_year_before = int(self.df['year'].max())
        
        self.df = self.df[(self.df['year'] >= 2005) & (self.df['year'] <= 2025)]
        
        with open(os.path.join(self.output_dir, 'temporal_validation_report.csv'), 'w') as f:
            f.write(f"min_year_before,{min_year_before}\n")
            f.write(f"max_year_before,{max_year_before}\n")
            f.write(f"records_after,{len(self.df)}\n")

        # Temporal bias validation
        monthly_dist = self.df.groupby('month').agg(records=('month', 'count'), species=('species_name', 'nunique')).reset_index()
        monthly_dist.to_csv(os.path.join(self.output_dir, 'monthly_sampling_distribution.csv'), index=False)
        
        self.track_attrition("3_Temporal")
        self.log_print(f"    -> {len(self.df)} records bounded to 2005-2025.")

    def step4_spatial_validation(self):
        self.log_print("[*] STEP 4: Spatial Validation & Coordinate Precision...")
        self.df = self.df.dropna(subset=['latitude', 'longitude'])
        self.df['latitude'] = pd.to_numeric(self.df['latitude'], errors='coerce')
        self.df['longitude'] = pd.to_numeric(self.df['longitude'], errors='coerce')
        self.df = self.df.dropna(subset=['latitude', 'longitude'])
        
        # Explicit (0,0) drop
        self.df = self.df[~((self.df['latitude'] == 0) & (self.df['longitude'] == 0))]
        
        lat_min, lat_max, lon_min, lon_max = 6.0, 14.0, 92.0, 94.5
        self.df = self.df[
            (self.df['latitude'] >= lat_min) & (self.df['latitude'] <= lat_max) &
            (self.df['longitude'] >= lon_min) & (self.df['longitude'] <= lon_max)
        ]

        # Precision validation
        self.df['lat_precision'] = self.df['latitude'].astype(str).apply(lambda x: len(x.split('.')[1]) if '.' in x else 0)
        self.df['lon_precision'] = self.df['longitude'].astype(str).apply(lambda x: len(x.split('.')[1]) if '.' in x else 0)
        
        precision_df = self.df.groupby(['lat_precision', 'lon_precision']).size().reset_index(name='count')
        precision_df.to_csv(os.path.join(self.output_dir, 'spatial_validation_report.csv'), index=False)
        self.df = self.df.drop(columns=['lat_precision', 'lon_precision'])
        
        self.track_attrition("4_Spatial")
        self.log_print(f"    -> {len(self.df)} records in ANI.")

    def step4_5_duplicate_validation(self):
        self.log_print("[*] STEP 4.5: Exact Duplicate Record Validation...")
        duplicate_count = self.df.duplicated(subset=['species_name', 'latitude', 'longitude', 'observation_date']).sum()
        self.df = self.df.drop_duplicates(subset=['species_name', 'latitude', 'longitude', 'observation_date'])
        self.track_attrition("4.5_Duplicates")
        self.log_print(f"    -> Removed {duplicate_count} exact duplicates. {len(self.df)} records remain.")

    def step5_taxonomic_standardization(self):
        self.log_print("[*] STEP 5: Strict Taxonomic Standardization...")
        self.df = self.df.dropna(subset=['species_name'])
        
        original_names = self.df['species_name'].copy()
        
        def clean_binomial(name):
            name = str(name).lower()
            # Remove uncertain markers and hybrids explicitly
            for marker in [' sp.', ' sp ', ' cf.', ' cf ', ' aff.', ' aff ', ' x ']:
                if marker in name: return "INVALID_TAXON"
            
            name = re.sub(r'\(.*?\)', '', name)
            name = re.sub(r'[^a-z\s\-]', '', name)
            words = name.split()
            if len(words) < 2: return "INVALID_TAXON"
            # Correct strict binomial casing (Genus species)
            cleaned = words[0].capitalize() + " " + words[1].lower()
            return cleaned

        self.df['cleaned_name'] = self.df['species_name'].apply(clean_binomial)
        
        # Build tax_df BEFORE dropping invalid rows to keep index aligned
        tax_df = pd.DataFrame({'original_name': original_names, 'cleaned_name': self.df['cleaned_name']})
        tax_df['modified_flag'] = tax_df['original_name'].str.lower() != tax_df['cleaned_name'].str.lower()
        tax_df.drop_duplicates().to_csv(os.path.join(self.output_dir, 'taxonomic_cleaning_report.csv'), index=False)

        self.df['species_name'] = self.df['cleaned_name']
        self.df = self.df[self.df['species_name'] != "INVALID_TAXON"]
        self.df = self.df.drop(columns=['cleaned_name'])
        
        self.track_attrition("5_Taxonomy")
        self.log_print(f"    -> {self.df['species_name'].nunique()} distinct species remaining.")

    def step6_migratory_species_identification(self):
        self.log_print("[*] STEP 6: Migratory Confidence Scoring...")
        endemic_blocklist = [
            'Rhyticeros narcondami', 'Psittacula caniceps', 'Columba palumboides', 
            'Ninox affinis', 'Macropygia rufipennis', 'Accipiter butleri', 
            'Sturnia erythropygia', 'Megapodius nicobariensis', 'Spilornis elgini'
        ]
        
        # 1. Output species monthly presence phenology
        monthly_presence = self.df.groupby(['species_name', 'month']).size().reset_index(name='records')
        monthly_presence.to_csv(os.path.join(self.output_dir, 'species_monthly_presence.csv'), index=False)
        
        summer_months = [5, 6, 7, 8]
        winter_months = [11, 12, 1, 2]
        
        species_stats = []
        for species, group in self.df.groupby('species_name'):
            total_count = len(group)
            summer_count = group['month'].isin(summer_months).sum()
            winter_count = group['month'].isin(winter_months).sum()
            
            summer_ratio = summer_count / total_count if total_count > 0 else 0
            winter_ratio = winter_count / total_count if total_count > 0 else 0
            
            month_counts = group['month'].value_counts()
            active_months = len(month_counts)
            peak_month = month_counts.idxmax() if not month_counts.empty else np.nan
            
            # Temporal concentration (Entropy) - lower entropy = highly seasonal
            month_probs = month_counts / total_count
            entropy_score = entropy(month_probs) if len(month_probs) > 1 else 0
            
            # Composite Migratory Confidence Score
            migration_score = (winter_ratio * 2.0) - (summer_ratio * 3.0) + (1.0 / (entropy_score + 0.1))
            
            is_endemic = species in endemic_blocklist
            # Allow species that either meet the rigid heuristic OR have a high scientific confidence score
            meets_heuristic = (total_count >= 50 and summer_ratio <= 0.01)
            meets_score = (total_count >= 50 and migration_score >= 1.5 and summer_ratio <= 0.05)
            
            retained = 'retained' if (not is_endemic and (meets_heuristic or meets_score)) else 'removed'
            
            species_stats.append({
                'species_name': species,
                'total_count': total_count,
                'summer_ratio': summer_ratio,
                'winter_ratio': winter_ratio,
                'peak_month': peak_month,
                'active_months': active_months,
                'entropy_score': entropy_score,
                'migration_score': migration_score,
                'retained_or_removed': retained
            })
            
        stats_df = pd.DataFrame(species_stats)
        stats_df.to_csv(os.path.join(self.output_dir, 'species_validation_report.csv'), index=False)
        
        migratory_species = stats_df[stats_df['retained_or_removed'] == 'retained']['species_name']
        self.df = self.df[self.df['species_name'].isin(migratory_species)]
        
        self.track_attrition("6_Migratory")
        self.log_print(f"    -> Identified {len(migratory_species)} migratory species via Confidence Scoring.")

    def step7_spatial_thinning(self):
        self.log_print("[*] STEP 7: Statistical Spatial Thinning...")
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.df['latitude'] = np.round(self.df['latitude'], 3)
        self.df['longitude'] = np.round(self.df['longitude'], 3)
        
        original_records = len(self.df)
        species_before = self.df['species_name'].nunique()
        
        # We already randomized the entire DataFrame perfectly via .sample(frac=1) above.
        # Therefore, .head(1) and .head(4) are mathematically identical to uniform random sampling 
        # without replacement, but 100x faster and guaranteed not to drop columns in Pandas 2.x!
        self.df = self.df.groupby(['latitude', 'longitude', 'species_name']).head(1)
        self.df = self.df.groupby(['latitude', 'longitude']).head(4)        
        grid_density = self.df.groupby(['latitude', 'longitude']).size()
        
        with open(os.path.join(self.output_dir, 'spatial_thinning_report.csv'), 'w') as f:
            f.write(f"original_records,{original_records}\n")
            f.write(f"final_records,{len(self.df)}\n")
            f.write(f"species_before,{species_before}\n")
            f.write(f"species_after,{self.df['species_name'].nunique()}\n")
            f.write(f"max_grid_density,{grid_density.max()}\n")
            f.write(f"mean_grid_density,{grid_density.mean():.2f}\n")
            
        self.track_attrition("7_Thinning")
        self.log_print(f"    -> Thinned to {len(self.df)} records.")

    def step8_final_validation(self):
        self.log_print("[*] STEP 8: Hard Thresholds & Ecological Validation...")
        
        errors = []
        retention_pct = (self.manifest["species_counts"]["7_Thinning"] / self.manifest["species_counts"]["6_Migratory"]) * 100
        if retention_pct < 90.0: errors.append(f"Species retention {retention_pct:.1f}% < 90%")
        
        max_density = self.df.groupby(['latitude', 'longitude']).size().max()
        if max_density > 4: errors.append(f"Max grid density {max_density} > 4")
        
        lat_bounds = self.df['latitude'].between(6.0, 14.0).all()
        lon_bounds = self.df['longitude'].between(92.0, 94.5).all()
        if not (lat_bounds and lon_bounds): errors.append("Coordinates outside ANI bounds")
        
        year_bounds = self.df['year'].between(2005, 2025).all()
        if not year_bounds: errors.append("Years outside 2005-2025")
        
        nulls = self.df[['latitude', 'longitude', 'species_name', 'year', 'month']].isna().sum().sum()
        if nulls > 0: errors.append(f"Found {nulls} nulls in critical columns")

        status = "FAIL" if errors else "PASS"
        
        with open(os.path.join(self.output_dir, 'phase1_validation_report.csv'), 'w') as f:
            f.write("Status,Message\n")
            f.write(f"{status},{' | '.join(errors) if errors else 'All thresholds passed'}\n")
            
        pd.DataFrame(self.attrition_log).to_csv(os.path.join(self.output_dir, 'species_attrition_tracking.csv'), index=False)
        
        # Upgraded Ecological Coverage
        eco_df = pd.DataFrame({
            'richness_by_year': self.df.groupby('year')['species_name'].nunique()
        })
        
        andaman_mask = self.df['latitude'] > 10.5
        nicobar_mask = self.df['latitude'] <= 10.5
        
        eco_df['andaman_records'] = self.df[andaman_mask].groupby('year').size()
        eco_df['nicobar_records'] = self.df[nicobar_mask].groupby('year').size()
        eco_df['total_records'] = self.df.groupby('year').size()
        eco_df = eco_df.fillna(0)
        
        eco_df.to_csv(os.path.join(self.output_dir, 'ecological_coverage_report.csv'))

        if status == "FAIL":
            raise ValueError(f"PIPELINE FAILED HARD THRESHOLDS: {errors}")
            
        self.log_print(f"    -> Validation {status}!")

    def step9_visualization(self):
        self.log_print("[*] STEP 9: Generating Diagnostic Visualizations...")
        sns.set_theme(style="whitegrid")
        df_copy = self.df.copy()
        df_copy['month_name'] = df_copy['month'].astype(int).apply(lambda x: calendar.month_abbr[x])
        month_order = [calendar.month_abbr[i] for i in range(1, 13)]
        df_copy['month_name'] = pd.Categorical(df_copy['month_name'], categories=month_order, ordered=True)

        # 1. MBSQI_Phase1_Analysis.png
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Phase 1: Spatio-Temporal Data Engineering Results (MBSQI)', fontsize=18, fontweight='bold')
        top_10 = df_copy['species_name'].value_counts().head(10)
        sns.barplot(x=top_10.values, y=top_10.index, ax=axes[0, 0], hue=top_10.index, palette='viridis', legend=False)
        axes[0, 0].set_title('Top 10 Migratory Species (Frequency)')
        month_counts = df_copy['month_name'].value_counts().sort_index()
        sns.barplot(x=month_counts.index, y=month_counts.values, ax=axes[0, 1], color='skyblue')
        axes[0, 1].set_title('Monthly Stopover Phenology (Summer Gap)')
        sns.scatterplot(data=df_copy.sort_values('month'), x='longitude', y='latitude', ax=axes[1, 0], hue='month_name', palette='Spectral', alpha=0.6, s=15)
        axes[1, 0].set_title('Spatial Distribution across ANI')
        yearly_counts = df_copy['year'].value_counts().sort_index()
        sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, ax=axes[1, 1], marker='o', color='darkgreen', linewidth=2.5)
        axes[1, 1].set_title('Validated Records Growth (2005-2025)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'MBSQI_Phase1_Analysis.png'), dpi=300)
        plt.close()

        # 2. Spatial_Density_Heatmap.png
        plt.figure(figsize=(10, 8))
        sns.histplot(data=df_copy, x='longitude', y='latitude', bins=50, pthresh=.1, cmap="mako", cbar=True)
        plt.title('Spatial Density Heatmap (Post-Thinning)')
        plt.savefig(os.path.join(self.output_dir, 'Spatial_Density_Heatmap.png'), dpi=300)
        plt.close()
        
        # 3. Species_Retention_Analysis.png
        plt.figure(figsize=(10, 6))
        steps = [log['step'] for log in self.attrition_log]
        species = [log['species'] for log in self.attrition_log]
        sns.lineplot(x=steps, y=species, marker='o', color='crimson')
        plt.title('Species Attrition Pipeline')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'Species_Retention_Analysis.png'), dpi=300)
        plt.close()

        # 4. Yearwise_Species_Diversity.png
        plt.figure(figsize=(10, 6))
        yearly_div = df_copy.groupby('year')['species_name'].nunique()
        sns.barplot(x=yearly_div.index.astype(int), y=yearly_div.values, color='teal')
        plt.title('Yearwise Species Diversity')
        plt.savefig(os.path.join(self.output_dir, 'Yearwise_Species_Diversity.png'), dpi=300)
        plt.close()

        # 5. Observation_Density_vs_Richness.png
        plt.figure(figsize=(10, 6))
        grid_stats = df_copy.groupby(['latitude', 'longitude']).agg(records=('species_name', 'count'), species=('species_name', 'nunique'))
        sns.scatterplot(data=grid_stats, x='records', y='species', alpha=0.5)
        plt.title('Observation Density vs Species Richness')
        plt.savefig(os.path.join(self.output_dir, 'Observation_Density_vs_Richness.png'), dpi=300)
        plt.close()

        # 6. Observer_Contribution_Distribution.png
        if 'observer_name' in df_copy.columns:
            plt.figure(figsize=(10, 6))
            obs_counts = df_copy['observer_name'].value_counts()
            sns.histplot(obs_counts, bins=30, log_scale=(False, True))
            plt.title('Observer Contribution Distribution (Log Scale)')
            plt.xlabel('Records per Observer')
            plt.savefig(os.path.join(self.output_dir, 'Observer_Contribution_Distribution.png'), dpi=300)
            plt.close()

        # 7. Month_wise_Sampling_Histogram.png
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df_copy, x='month_name', shrink=.8)
        plt.title('Month-wise Sampling Histogram')
        plt.savefig(os.path.join(self.output_dir, 'Month_wise_Sampling_Histogram.png'), dpi=300)
        plt.close()
        
        self.log_print("    -> Visualizations generated successfully.")

    def step10_final_freeze(self):
        self.log_print("[*] STEP 10: Final Dataset Freeze & Manifest Generation...")
        output_csv = os.path.join(self.output_dir, "MIGRATORY_BIRDS_THINNED_FINAL.csv")
        self.df.to_csv(output_csv, index=False)
        
        with open(os.path.join(self.output_dir, "phase1_manifest.json"), 'w') as f:
            json.dump(self.manifest, f, indent=4)
            
        self.log_print("="*60)
        self.log_print("FINAL PHASE 1 DATA FREEZE SUMMARY")
        self.log_print("="*60)
        self.log_print(f"Final Records: {len(self.df)}")
        self.log_print(f"Final Species: {self.df['species_name'].nunique()}")
        self.log_print(f"Manifest Generated: phase1_manifest.json")

    def run(self):
        start_time = time.time()
        self.log_print("="*60)
        self.log_print("INITIALIZING RESEARCH-GRADE MBSQI PHASE 1 PIPELINE")
        self.log_print("="*60)
        
        self.step1_ingestion_and_harmonization()
        self.step2_feature_density_and_observer_bias()
        self.step3_temporal_standardization()
        self.step4_spatial_validation()
        self.step4_5_duplicate_validation()
        self.step5_taxonomic_standardization()
        self.step6_migratory_species_identification()
        self.step7_spatial_thinning()
        self.step8_final_validation()
        self.step9_visualization()
        self.step10_final_freeze()
        
        self.log_print(f"\nExecution Time: {(time.time() - start_time):.2f} seconds")
        self.log_print("PIPELINE COMPLETE.")

if __name__ == "__main__":
    INPUT_DIRECTORY = r"C:\Users\Abhishek\IIITH\IITH\INDEPENDENT_STUDY\PROJECT\PHASE1\EXTRACTED_ANI_BIRDS"
    OUTPUT_DIRECTORY = r"C:\Users\Abhishek\IIITH\IITH\INDEPENDENT_STUDY\PROJECT\PHASE1\CLEANED_DATA"
    pipeline = MBSQIPhase1Pipeline(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
    pipeline.run()