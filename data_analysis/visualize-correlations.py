import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
DATASET_ROOT = "dataset_root"
CSV_FILES = [
    os.path.join(DATASET_ROOT, "picorv32_batch1.csv"),
    os.path.join(DATASET_ROOT, "picorv32_batch2.csv"),
    os.path.join(DATASET_ROOT, "aes_batch1.csv"),
    os.path.join(DATASET_ROOT, "aes_batch2.csv"),
    os.path.join(DATASET_ROOT, "aes_batch3.csv"),
    os.path.join(DATASET_ROOT, "sha256_batch1.csv"),
    os.path.join(DATASET_ROOT, "sha256_batch2.csv"),
    os.path.join(DATASET_ROOT, "sha256_batch3.csv"),
    os.path.join(DATASET_ROOT, "sha256_batch4.csv"),
    os.path.join(DATASET_ROOT, "ethmac_batch1.csv"),
    os.path.join(DATASET_ROOT, "ethmac_batch2.csv"),
    os.path.join(DATASET_ROOT, "ethmac_batch3.csv"),
    os.path.join(DATASET_ROOT, "ethmac_batch4.csv")
]
OUTPUT_PLOT = "physics_correlation_matrix_full.png"


def process_design(csv_paths, out_dir, design_name=None):
    """Compute and save correlation heatmap for given csv_paths into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    out_plot = os.path.join(out_dir, f"physics_correlation_matrix_full{('_' + design_name) if design_name else ''}.png")

    df_list = []
    for f in csv_paths:
        if os.path.exists(f):
            try:
                df_list.append(pd.read_csv(f))
            except Exception:
                pass
    if not df_list:
        print(f"❌ No CSV files loaded for design {design_name}.")
        return
    df = pd.concat(df_list, ignore_index=True)

    # Inputs / Outputs selection
    inputs = [
        'aspect_ratio', 
        'core_util', 
        'density', 
        'cts_max_wire', 
        'cts_buf_dist', 
        'cts_cluster_size', 
        'cts_cluster_dia'
    ]
    outputs = [
        'skew_setup', 
        'setup_slack', 
        'setup_tns',
        'setup_vio_count',
        'skew_hold', 
        'hold_slack', 
        'hold_tns',
        'hold_vio_count',
        'clock_buffers', 
        'clock_inverters',
        'wirelength', 
        'power_total', 
        'utilization'
    ]
    valid_inputs = [c for c in inputs if c in df.columns]
    valid_outputs = [c for c in outputs if c in df.columns]
    if not valid_inputs or not valid_outputs:
        print(f"Error: Could not find specified columns in CSV for {design_name}.")
        return
    subset = df[valid_inputs + valid_outputs]
    corr_matrix = subset.corr(method='pearson')
    heatmap_data = corr_matrix.loc[valid_inputs, valid_outputs]

    plt.figure(figsize=(16, 10))
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
    plt.title(f"Physical Design Physics: Input Knobs vs. Full Metrics Correlation ({design_name})")
    plt.xlabel("Output Metrics (Performance)")
    plt.ylabel("Input Knobs (Constraints)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"✅ Correlation heatmap saved to {out_plot}")

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