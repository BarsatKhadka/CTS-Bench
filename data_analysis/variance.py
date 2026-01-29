import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
DATASET_ROOT = "./dataset_root"
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
OUTPUT_REPORT = "dataset_full_stats_report.txt"
OUTPUT_PLOT = "dataset_full_distributions.png"

def main():
    print("🔹 1. Loading Data...")
    df_list = []
    for f in CSV_FILES:
        if os.path.exists(f):
            df_list.append(pd.read_csv(f))
    
    if not df_list: return
    df = pd.concat(df_list, ignore_index=True)
    print(f"   Loaded {len(df)} total data points.")

    # --- 2. Define ALL Metrics to Analyze ---
    # The full list you requested:
    target_metrics = [
        'utilization', 
        'wirelength', 
        'power_total',
        
        # Setup Timing
        'setup_slack', 
        'setup_tns',       # Total Negative Slack (Sum of all failures)
        'setup_vio_count', # How many paths failed?
        'skew_setup',      # Critical for setup fixes
        
        # Hold Timing
        'hold_slack', 
        'hold_tns',
        'hold_vio_count', 
        'skew_hold',       # Critical for hold fixes
        
        # Resources / Effort
        'clock_buffers', 
        'clock_inverters', 
        'timing_repair_buffers' # Buffers added specifically to fix hold times
    ]

    # Filter to only columns that actually exist
    valid_metrics = [c for c in target_metrics if c in df.columns]

    # --- 3. Generate Statistical Report ---
    print("\n🔹 2. Calculating Variance Statistics...")
    stats = df[valid_metrics].describe().T
    
    # Calculate "Coefficient of Variation" (CV) = StdDev / Mean
    # Note: For columns with mean near 0 (like TNS), CV might be huge/unstable, 
    # but it's still useful to see the relative spread.
    stats['CV (%)'] = (stats['std'] / stats['mean'].abs()) * 100
    
    # Select columns for the report
    report_df = stats[['min', 'max', 'mean', 'std', 'CV (%)']]
    
    print("\n--- FULL DATASET VARIANCE REPORT ---")
    print(report_df.round(2))
    
    # Save to text file
    with open(OUTPUT_REPORT, "w") as f:
        f.write(report_df.round(2).to_markdown())
    print(f"\n✅ Report saved to {OUTPUT_REPORT}")

    # --- 4. Plot Distributions ---
    print("\n🔹 3. Generating Distribution Plots...")
    
    num_plots = len(valid_metrics)
    rows = (num_plots // 4) + 1  # 4 plots per row
    
    plt.figure(figsize=(20, 4 * rows))
    plt.suptitle("Full Dataset Diversity: Distribution of All Metrics", fontsize=20)
    
    for i, col in enumerate(valid_metrics):
        plt.subplot(rows, 4, i + 1)
        
        # Color logic: Violations = Red, Slack = Orange, Physical = Blue
        if "vio" in col or "tns" in col:
            color = "firebrick"
        elif "slack" in col:
            color = "darkorange"
        else:
            color = "steelblue"
        
        sns.histplot(df[col], kde=True, color=color, bins=30)
        
        plt.title(f"{col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        
        # Add Range Text
        r_min, r_max = df[col].min(), df[col].max()
        plt.text(0.95, 0.95, f"Min: {r_min:.2f}\nMax: {r_max:.2f}", 
                 transform=plt.gca().transAxes, ha='right', va='top', 
                 bbox=dict(boxstyle="round", fc="white", alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"✅ Distribution Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()