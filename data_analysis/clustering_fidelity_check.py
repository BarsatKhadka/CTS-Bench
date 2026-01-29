import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- CONFIGURATION ---
DATASET_ROOT = "./dataset_root"
OUTPUT_ROOT = "clustering_outputs"  # Root folder for all results

# Fixed the missing comma in the list below
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

def make_radar_chart(df, categories, title, save_path):
    """Generates a Radar Chart for the Cluster Centroids."""
    if df.empty:
        print(f"⚠️ Skipping Radar Chart for {title} (Not enough data)")
        return

    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=7)
    plt.ylim(0, 1)
    
    colors = {"Golden (Clean)": "green", "Marginal": "orange", "Broken": "red"}
    
    for idx, row in df.iterrows():
        label = row['Label']
        values = row[categories].values.flatten().tolist()
        values += values[:1]
        
        color = colors.get(label, "blue")
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=label, color=color)
        ax.fill(angles, values, color=color, alpha=0.1)
        
    plt.title(title, size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig(save_path, dpi=300)
    plt.close() # Close memory to prevent leaks in loop
    print(f"✅ Radar Chart saved to {save_path}")

def generate_plots_for_subset(valid_df, output_folder, run_name):
    """Reusable function to generate plots for a specific subset of data."""
    
    # ensure folder exists
    os.makedirs(output_folder, exist_ok=True)

    # --- 1. Prepare Radar Data ---
    radar_df = valid_df.copy()
    min_max = MinMaxScaler()
    
    # Invert TNS so that "Larger = Worse" (consistent with other metrics)
    radar_df['Setup Severity'] = -1 * radar_df['setup_tns']
    radar_df['Hold Severity']  = -1 * radar_df['hold_tns']
    
    plot_feats_map = {
        'Setup Severity': 'Setup Severity',
        'Hold Severity': 'Hold Severity',
        'setup_vio_count': 'Vio Count',
        'wirelength': 'Wirelength',
        'power_total': 'Power',
        'clock_buffers': 'Buffers'
    }
    
    cols_to_norm = list(plot_feats_map.keys())
    
    # Safety check: if standard deviation is 0 (all values same), minmax fails/warns
    # We catch this by checking if dataframe is large enough
    if len(radar_df) > 0:
        radar_df[cols_to_norm] = min_max.fit_transform(radar_df[cols_to_norm])
        radar_df = radar_df.rename(columns=plot_feats_map)
        final_cats = list(plot_feats_map.values())
        
        radar_centers = radar_df.groupby('Label')[final_cats].mean().reset_index()
        radar_path = os.path.join(output_folder, f"{run_name}_radar_profile.png")
        make_radar_chart(radar_centers, final_cats, f"{run_name}: Cluster Profiles", radar_path)
    
    # --- 2. Box Plots ---
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"{run_name}: Metric Distributions", fontsize=16)
    metrics_to_plot = ['setup_tns', 'setup_vio_count', 'hold_tns', 'wirelength', 'power_total', 'clock_buffers']
    
    for i, col in enumerate(metrics_to_plot):
        plt.subplot(2, 3, i+1)
        sns.boxplot(data=valid_df, x="Label", y=col, 
                    order=["Golden (Clean)", "Marginal", "Broken"],
                    palette={"Golden (Clean)": "green", "Marginal": "orange", "Broken": "red"})
        plt.title(col)
        plt.xlabel("")
        plt.xticks(rotation=15)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    box_path = os.path.join(output_folder, f"{run_name}_box_metrics.png")
    plt.savefig(box_path, dpi=300)
    plt.close()
    print(f"✅ Box Plots saved to {box_path}")


def main():
    print("🔹 1. Loading Data...")
    df_list = []
    
    # Create Root Output Directory
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    for file_name in CSV_FILES:
        # Extract Design Name from filename (e.g., "picorv32_batch1.csv" -> "picorv32")
        # Logic: split by underscore, take first part
        base_name = os.path.basename(file_name)
        design_name = base_name.split('_')[0] 

        if os.path.exists(file_name):
            try:
                temp_df = pd.read_csv(file_name)
                temp_df['Design'] = design_name  # Tag the data
                temp_df['SourceFile'] = base_name
                df_list.append(temp_df)
            except Exception as e:
                print(f"⚠️ Error reading {file_name}: {e}")
        else:
            print(f"⚠️ File not found: {file_name}")
            
    if not df_list:
        print("❌ No data found.")
        return

    df = pd.concat(df_list, ignore_index=True)
    
    # --- 2. Select Features ---
    features = [
        'setup_tns',        
        'hold_tns',         
        'setup_vio_count',  
        'wirelength',       
        'power_total',      
        'clock_buffers'     
    ]
    
    X_raw = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # --- 3. Run Global K-Means (Context is important!) ---
    # We cluster on ALL data first. This ensures "Broken" means the same thing
    # for AES as it does for PicoRV32 (Global Standard).
    
    print("🔹 2. Running Global K-Means (k=3)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Metrics
    inertia = kmeans.inertia_
    sil_score = silhouette_score(X_scaled, clusters)
    print("\n📊 --- GLOBAL CLUSTERING METRICS ---")
    print(f"   Inertia: {inertia:.2f}")
    print(f"   Silhouette Score: {sil_score:.4f}")
    print("------------------------------------------\n")

    # --- 4. Labeling ---
    df.loc[X_raw.index, 'Cluster'] = clusters
    valid_df = df.dropna(subset=['Cluster'])
    
    # Define labels based on worst setup timing (setup_tns)
    profile = valid_df.groupby('Cluster')[features].mean()
    profile['abs_tns'] = profile['setup_tns'].abs()
    sorted_clusters = profile.sort_values(by='abs_tns', ascending=True)
    
    cluster_names = {
        sorted_clusters.index[0]: "Golden (Clean)",
        sorted_clusters.index[1]: "Marginal",
        sorted_clusters.index[2]: "Broken"
    }
    valid_df['Label'] = valid_df['Cluster'].map(cluster_names)
    
    # --- 5. Generate Outputs ---
    
    print("🔹 3. Generating Output Folders...")

    # A. Whole Run (All Designs Combined)
    all_designs_dir = os.path.join(OUTPUT_ROOT, "00_ALL_DESIGNS")
    print(f"\n   Processing: ALL DESIGNS -> {all_designs_dir}")
    generate_plots_for_subset(valid_df, all_designs_dir, "All_Designs")
    
    # Save the labeled CSV for the whole run
    valid_df.to_csv(os.path.join(all_designs_dir, "all_designs_labeled.csv"), index=False)

    # B. Per Design Loop
    unique_designs = valid_df['Design'].unique()
    
    for design in unique_designs:
        design_dir = os.path.join(OUTPUT_ROOT, design)
        design_subset = valid_df[valid_df['Design'] == design]
        
        print(f"   Processing: {design} -> {design_dir}")
        generate_plots_for_subset(design_subset, design_dir, design)

    print(f"\n✅ Done! Check the '{OUTPUT_ROOT}' directory.")

if __name__ == "__main__":
    main()