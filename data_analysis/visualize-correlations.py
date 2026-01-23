import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
DATASET_ROOT = "dataset_root"
# List all CSV files you want to include
CSV_FILES = [
    os.path.join(DATASET_ROOT, "experiment_log.csv"),
    os.path.join(DATASET_ROOT, "picorv32_batch1.csv")
]
OUTPUT_PLOT = "physics_correlation_matrix_full.png"

def main():
    print("🔹 1. Loading Data...")
    df_list = []
    for f in CSV_FILES:
        if os.path.exists(f):
            print(f"   - Found {f}")
            df_list.append(pd.read_csv(f))
        else:
            print(f"   ❌ Warning: File not found: {f}")
    
    if not df_list:
        print("No CSV files loaded. Exiting.")
        return

    df = pd.concat(df_list, ignore_index=True)
    print(f"   Loaded {len(df)} total rows.")
    
    # --- 2. Select Columns to Correlate ---
    
    # INPUTS (The Knobs you change)
    inputs = [
        'aspect_ratio', 
        'core_util', 
        'density', 
        'cts_max_wire', 
        'cts_buf_dist', 
        'cts_cluster_size', 
        'cts_cluster_dia'
    ]
    
    # OUTPUTS (The Metrics you measure)
    outputs = [
        # Timing - Setup
        'skew_setup', 
        'setup_slack', 
        'setup_tns',
        'setup_vio_count',
        
        # Timing - Hold
        'skew_hold', 
        'hold_slack', 
        'hold_tns',
        'hold_vio_count',
        
        # Physical / Power
        'clock_buffers', 
        'clock_inverters',
        'wirelength', 
        'power_total', 
        'utilization'
    ]
    
    # Filter to only columns that actually exist in the CSV (prevents crashes on typos)
    valid_inputs = [c for c in inputs if c in df.columns]
    valid_outputs = [c for c in outputs if c in df.columns]
    
    if not valid_inputs or not valid_outputs:
        print("Error: Could not find specified columns in CSV.")
        return

    subset = df[valid_inputs + valid_outputs]
    
    # --- 3. Compute Correlation ---
    print("🔹 2. Computing Correlations...")
    # method='pearson' is standard. You could use 'spearman' for non-linear rank correlation.
    corr_matrix = subset.corr(method='pearson')
    
    # Slice the matrix: Rows = Inputs, Cols = Outputs
    heatmap_data = corr_matrix.loc[valid_inputs, valid_outputs]

    # --- 4. Plot ---
    plt.figure(figsize=(16, 10)) # Made figure wider to fit new columns
    
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        center=0,
        linewidths=0.5, 
        linecolor='gray',
        cbar_kws={'label': 'Correlation Coefficient (Pearson)'}
    )
    
    plt.title("Physical Design Physics: Input Knobs vs. Full Metrics Correlation")
    plt.xlabel("Output Metrics (Performance)")
    plt.ylabel("Input Knobs (Constraints)")
    plt.xticks(rotation=45, ha='right') # Rotate x labels for readability
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"✅ Full Correlation Heatmap saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()