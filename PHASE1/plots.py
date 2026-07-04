import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import argparse

def log_print(message):
    """Prints a message to the console."""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode('ascii', 'replace').decode('ascii'))

def generate_visualizations(data_dir, plot_dir):
    """
    Generates a selection of diagnostic visualizations for the Phase 1 dataset.

    Args:
        data_dir (str): The directory containing the cleaned Phase 1 output files.
        plot_dir (str): The directory where the generated plots will be saved.
    """
    log_print("[*] Initializing Phase 1 Visualization Generation...")

    final_data_path = os.path.join(data_dir, 'MIGRATORY_BIRDS_THINNED_FINAL.csv')

    if not os.path.exists(final_data_path):
        log_print(f"FATAL: Final dataset not found at {final_data_path}")
        return
    df = pd.read_csv(final_data_path, low_memory=False)
    log_print(f"-> Loaded {len(df)} records from final dataset.")

    os.makedirs(plot_dir, exist_ok=True)

    # --- Plot 1: Spatial Distribution ---
    try:
        log_print("1/3 -> Generating Spatial Distribution Plot...")
        plt.style.use('grayscale')
        fig, ax = plt.subplots(figsize=(10, 10))
        
        sns.scatterplot(
            data=df, x='longitude', y='latitude', 
            alpha=0.5, s=15, color='black', ax=ax
        )
        
        ax.set_title('Spatial Distribution of Thinned Migratory Bird Observations', fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'MBSQI_Phase1_Spatial_Distribution.png'), dpi=300)
        plt.close()
        log_print("    -> Saved MBSQI_Phase1_Spatial_Distribution.png")
    except Exception as e:
        log_print(f"    ERROR generating spatial plot: {e}")

    # --- Plot 2: Year-wise Species Diversity ---
    try:
        log_print("2/3 -> Generating Year-wise Species Diversity Plot...")
        plt.style.use('grayscale')
        plt.figure(figsize=(12, 7))
        
        yearly_div = df.groupby('year')['species_name'].nunique().reset_index()
        
        sns.barplot(
            data=yearly_div, x='year', y='species_name', 
            color='darkgray', edgecolor='black'
        )
        
        plt.title('Year-wise Unique Migratory Species Observed', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Unique Species', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'Yearwise_Species_Diversity.png'), dpi=300)
        plt.close()
        log_print("    -> Saved Yearwise_Species_Diversity.png")
    except Exception as e:
        log_print(f"    ERROR generating year-wise diversity plot: {e}")

    # --- Plot 3: Monthly Observation Distribution ---
    try:
        log_print("3/3 -> Generating Monthly Observation Distribution Plot...")
        plt.style.use('grayscale')
        plt.figure(figsize=(12, 7))
        
        month_order = [calendar.month_abbr[i] for i in range(1, 13)]
        df['month_name'] = pd.Categorical(df['month'].apply(lambda x: calendar.month_abbr[int(x)]), categories=month_order, ordered=True)
        
        sns.countplot(
            data=df, x='month_name', 
            color='darkgray', edgecolor='black'
        )
        
        plt.title('Monthly Distribution of Migratory Bird Observations', fontsize=16, fontweight='bold')
        plt.xlabel('Month', fontsize=12, fontweight='bold')
        plt.ylabel('Total Number of Observations', fontsize=12, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'Monthly_Observation_Distribution.png'), dpi=300)
        plt.close()
        log_print("    -> Saved Monthly_Observation_Distribution.png")
    except Exception as e:
        log_print(f"    ERROR generating monthly distribution plot: {e}")

    log_print("\n[*] Visualization generation complete.")
    log_print(f"[*] All plots saved to: {os.path.abspath(plot_dir)}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate selected diagnostic plots for the MBSQI Phase 1 cleaned data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=r"c:\Users\Abhishek\IIITH\IITH\INDEPENDENT_STUDY\IS2\PHASE1\CLEANED_DATA",
        help="Path to the directory containing the Phase 1 cleaned data outputs.\n"
             "This directory must contain 'MIGRATORY_BIRDS_THINNED_FINAL.csv'."
    )
    parser.add_argument(
        '--plot_dir',
        type=str,
        default=r"c:\Users\Abhishek\IIITH\IITH\INDEPENDENT_STUDY\IS2\PHASE1\plots",
        help="Path to the directory where plots will be saved."
    )

    args = parser.parse_args()

    # Run the visualization generator
    generate_visualizations(args.data_dir, args.plot_dir)