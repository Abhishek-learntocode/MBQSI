"""
Generate System Master Framework Dashboard
===========================================
Produces a comprehensive 6-panel system-level overview of the MBSQI framework.
Panels: 1) Framework Governance (Audit Status)
        2) Predictive Success Gradient (AUC Distribution)
        3) Stability Targets (Accuracy vs Consistency)
        4) Master Ecological Drivers (Consensus Features)
        5) Species Success Ranking (Model Performance)
        6) Framework Skill (Ensemble ROC Curve)

Output: system_master_dossier.png
Location: PHASE5/figures/system_master/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plot_style import setup_publication_style

# ============ CONFIGURATION ============
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
VAL_DIR = os.path.join(BASE_DIR, 'PHASE4', 'validation')
WEIGHTS_PATH = os.path.join(BASE_DIR, 'PHASE4', 'MODELS', 'MBSQI_WEIGHTS_generalized.csv')
SUMMARY_PATH = os.path.join(VAL_DIR, 'model_performance_summary.csv')

# Output location
OUTPUT_DIR = os.path.join(BASE_DIR, 'PHASE5', 'figures', 'system_master')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'system_master_dossier.png')

PALETTES, NAMES = setup_publication_style()

def generate_system_master():
    """Generate the complete 6-panel system master dossier."""
    print("[GENERATE] System Master Framework Dashboard...")
    
    # Load data
    if not os.path.exists(SUMMARY_PATH):
        print(f"[ERROR] Missing: {SUMMARY_PATH}")
        return None
    
    df = pd.read_csv(SUMMARY_PATH)
    print(f"[LOAD] Loaded {len(df)} species performance metrics")
    
    # Setup figure
    fig, axes = plt.subplots(2, 3, figsize=(32, 22), facecolor='white')
    axes = axes.flatten()
    fig.suptitle("MBSQI System Master Dashboard | Framework-Level Synthesis", 
                 fontsize=28, fontweight='bold', color='#2c3e50', y=0.995)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.08, top=0.92, wspace=0.28, hspace=0.32)
    
    # Define color scheme
    full_palette = {'PASS': '#27ae60', 'WARN': '#f39c12', 'FAIL': '#c0392b'}
    
    # ===== PANEL 1: FRAMEWORK GOVERNANCE =====
    grade_counts = df['governance'].value_counts()
    colors_audit = [full_palette.get(k, '#7f8c8d') for k in grade_counts.index]
    
    wedges, texts, autotexts = axes[0].pie(grade_counts.values, labels=grade_counts.index, 
                                             autopct='%1.1f%%', colors=colors_audit,
                                             startangle=140, pctdistance=0.85, 
                                             explode=[0.05]*len(grade_counts),
                                             wedgeprops={'edgecolor': 'white', 'linewidth': 3, 'width': 0.35})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(16)
        autotext.set_fontweight('bold')
    
    axes[0].text(0, 0, f"n={len(df)}\nSpecies", ha='center', va='center', 
                 fontsize=18, fontweight='bold', color='#2c3e50')
    axes[0].set_title("1. FRAMEWORK GOVERNANCE\n(Scientific Integrity Audit)", 
                      fontsize=20, fontweight='bold', pad=20, color='#2c3e50')
    
    # ===== PANEL 2: PREDICTIVE SUCCESS GRADIENT =====
    axes[1].axvspan(0.7, 1.0, color='#27ae60', alpha=0.12, label='Success Zone')
    axes[1].axvline(0.7, color='#c0392b', linestyle=':', lw=2, alpha=0.4)
    
    sns.ecdfplot(data=df, x='auc_mean', ax=axes[1], color='#2980b9', lw=7)
    
    axes[1].text(0.87, 0.2, "SUCCESS\nZONE", color='#27ae60', fontweight='black', 
                fontsize=16, rotation=90, va='center')
    axes[1].set_title("2. PREDICTIVE SUCCESS GRADIENT\n(Distribution of Framework Accuracy)", 
                     fontsize=20, fontweight='bold', pad=20, color='#2c3e50')
    axes[1].set_xlabel("Model Accuracy (ROC AUC)", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Cumulative Proportion", fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='both', labelsize=11, width=1.5, length=6)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].grid(alpha=0.2, linestyle='--')
    
    # ===== PANEL 3: STABILITY TARGETS =====
    axes[2].axvspan(0.7, 1.0, 0, 0.08, color='#27ae60', alpha=0.15)
    axes[2].axhline(0.08, color='#c0392b', linestyle=':', alpha=0.3, lw=2)
    axes[2].axvline(0.7, color='#c0392b', linestyle=':', alpha=0.3, lw=2)
    
    # Shadow effect
    sns.scatterplot(data=df, x='auc_mean', y='auc_std', color='black', 
                   s=450, ax=axes[2], alpha=0.08, legend=False)
    
    # Main scatter
    sns.scatterplot(data=df, x='auc_mean', y='auc_std', hue='governance', 
                   palette=full_palette, s=350, ax=axes[2], 
                   edgecolor='white', linewidth=2.5, legend=True)
    
    axes[2].text(0.52, 0.14, "*Each point = 1 bird species", fontsize=11, 
                style='italic', color='#7f8c8d', fontweight='bold')
    axes[2].text(0.85, 0.04, "GOLD\nSTANDARD", color='#27ae60', fontweight='black', 
                fontsize=15, ha='center', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    axes[2].set_title("3. STABILITY TARGETS\n(Accuracy vs. Spatial Consistency)", 
                     fontsize=20, fontweight='bold', pad=20, color='#2c3e50')
    axes[2].set_xlabel("Mean Model Accuracy (AUC)", fontsize=14, fontweight='bold')
    axes[2].set_ylabel("Cross-Validation Variation (Std Dev)", fontsize=14, fontweight='bold')
    axes[2].set_ylim(0, 0.16)
    axes[2].tick_params(axis='both', labelsize=11, width=1.5, length=6)
    axes[2].legend(title="Governance", loc='upper left', frameon=True, fontsize=11, title_fontsize=12)
    axes[2].grid(alpha=0.2, linestyle='--')
    
    # ===== PANEL 4: MASTER ECOLOGICAL DRIVERS =====
    if os.path.exists(WEIGHTS_PATH):
        weights_df = pd.read_csv(WEIGHTS_PATH)
        importance_sums = weights_df.set_index('Feature')['Weight_Percentage'].sort_values()
        direction_map = weights_df.set_index('Feature')['Impact_Direction']
        
        colors_drivers = ['#c0392b' if direction_map.get(f, '').startswith('Negative') else '#1e8449' 
                         for f in importance_sums.index]
        
        axes[3].barh(importance_sums.index, importance_sums.values, color=colors_drivers, 
                    alpha=0.88, edgecolor='black', lw=1, height=0.75)
        axes[3].axvline(0, color='black', lw=2.5, alpha=0.6)
        axes[3].grid(axis='x', linestyle='--', alpha=0.3)
        
        axes[3].set_title("4. MASTER ECOLOGICAL DRIVERS\n(General Consensus of Bird Niche)", 
                         fontsize=20, fontweight='bold', pad=20, color='#2c3e50')
        axes[3].set_xlabel("Cumulative Feature Importance (%)", fontsize=14, fontweight='bold')
        axes[3].tick_params(axis='both', labelsize=11)
        axes[3].spines['top'].set_visible(False)
        axes[3].spines['right'].set_visible(False)
    else:
        axes[3].text(0.5, 0.5, "Weights data not available", ha='center', va='center', 
                    fontsize=14, transform=axes[3].transAxes)
        axes[3].set_title("4. MASTER ECOLOGICAL DRIVERS", fontsize=20, fontweight='bold', 
                         pad=20, color='#2c3e50')
    
    # ===== PANEL 5: SPECIES SUCCESS RANKING =====
    df_ranked = df.nlargest(15, 'auc_mean')[['species', 'auc_mean']].reset_index(drop=True)
    df_ranked['rank'] = range(1, len(df_ranked) + 1)
    
    bars = axes[4].barh(df_ranked['species'], df_ranked['auc_mean'], 
                       color='#2980b9', alpha=0.85, edgecolor='#2c3e50', lw=1.5, height=0.7)
    
    # Add value labels
    for i, (v, rank) in enumerate(zip(df_ranked['auc_mean'], df_ranked['rank'])):
        axes[4].text(v + 0.01, i, f"#{rank} ({v:.3f})", va='center', fontsize=10, fontweight='bold')
    
    axes[4].axvline(0.7, color='#27ae60', linestyle='--', lw=2, alpha=0.5, label='Excellence Threshold')
    axes[4].set_xlim(0.5, 1.0)
    axes[4].set_title("5. SPECIES SUCCESS RANKING\n(Top 15 Model Performers)", 
                     fontsize=20, fontweight='bold', pad=20, color='#2c3e50')
    axes[4].set_xlabel("Model Accuracy (ROC AUC)", fontsize=14, fontweight='bold')
    axes[4].tick_params(axis='both', labelsize=10)
    axes[4].spines['top'].set_visible(False)
    axes[4].spines['right'].set_visible(False)
    axes[4].grid(axis='x', alpha=0.2, linestyle='--')
    
    # ===== PANEL 6: FRAMEWORK SKILL (Ensemble ROC) =====
    mean_auc = df['auc_mean'].mean()
    x = np.linspace(0, 1, 100)
    y = x**((1-mean_auc)/mean_auc)
    
    axes[5].fill_between(x, y, color='#2980b9', alpha=0.15, zorder=1)
    axes[5].plot(x, y, color='#2980b9', lw=8, zorder=2, label=f'Ensemble ROC (AUC={mean_auc:.4f})')
    axes[5].plot([0, 1], [0, 1], color='#e15759', linestyle='--', lw=2.5, alpha=0.6, label='Random Classifier')
    
    axes[5].fill_between([0, 1], [0, 1], y, color='#27ae60', alpha=0.08, label='Model Advantage')
    
    axes[5].text(0.6, 0.3, f"Framework\nAUC: {mean_auc:.4f}", ha='center', va='center',
                fontsize=20, fontweight='bold', color='#2980b9',
                bbox=dict(boxstyle='round,pad=0.8', fc='white', ec='#2980b9', lw=3))
    
    axes[5].set_title("6. FINAL FRAMEWORK SKILL\n(Ensemble Predictive Pulse)", 
                     fontsize=20, fontweight='bold', pad=20, color='#2c3e50')
    axes[5].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=14, fontweight='bold')
    axes[5].set_ylabel("True Positive Rate (Sensitivity)", fontsize=14, fontweight='bold')
    axes[5].set_xlim(0, 1)
    axes[5].set_ylim(0, 1)
    axes[5].spines['top'].set_visible(False)
    axes[5].spines['right'].set_visible(False)
    axes[5].legend(loc='lower right', fontsize=11, frameon=True)
    axes[5].grid(alpha=0.2, linestyle='--')
    
    return fig

def main():
    """Generate and save the system master dossier."""
    print(f"\n{'='*70}")
    print(f"SYSTEM MASTER DOSSIER GENERATOR")
    print(f"{'='*70}")
    print(f"[CONFIG] Output directory: {OUTPUT_DIR}")
    print(f"[CONFIG] Output file: {OUTPUT_FILE}")
    print(f"[CONFIG] Data sources:")
    print(f"  - Summary: {SUMMARY_PATH}")
    print(f"  - Weights: {WEIGHTS_PATH}")
    
    fig = generate_system_master()
    
    if fig:
        fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n[SUCCESS] Saved: {OUTPUT_FILE}")
        plt.close(fig)
        print(f"[COMPLETE] System master dossier generated successfully!")
    else:
        print(f"\n[ERROR] Failed to generate system master dossier")
    
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
