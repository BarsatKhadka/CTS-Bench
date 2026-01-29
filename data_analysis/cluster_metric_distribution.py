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
OUTPUT_RADAR = "cluster_radar_profile.png"
OUTPUT_BOX = "cluster_metric_distributions.png"

def make_radar_chart(df, categories, title):
    """Generates a Radar Chart for the Cluster Centroids."""
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
    plt.savefig(OUTPUT_RADAR, dpi=300)
    print(f"✅ Radar Chart saved to {OUTPUT_RADAR}")

def main():
    print("🔹 1. Loading Data...")
    df_list = []
    for file_name in CSV_FILES:
        full_path = os.path.join(DATASET_ROOT, file_name)
        if os.path.exists(full_path):
            try:
                df_list.append(pd.read_csv(full_path))
            except: pass
            
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
    
    # --- 3. Run K-Means & Calculate Validation Metrics ---
    print("🔹 2. Running K-Means (k=3)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # --- NEW: Calculate Justification Metrics ---
    inertia = kmeans.inertia_
    sil_score = silhouette_score(X_scaled, clusters)
    
    print("\n📊 --- CLUSTERING VALIDATION METRICS ---")
    print(f"   Inertia (Sum of Squared Distances): {inertia:.2f}")
    print(f"   Silhouette Score: {sil_score:.4f}")
    print("   (> 0.5 is usually considered 'Good Structure')")
    print("------------------------------------------\n")

    # --- 4. Labeling & Profiling (Same as before) ---
    df.loc[X_raw.index, 'Cluster'] = clusters
    valid_df = df.dropna(subset=['Cluster'])
    
    profile = valid_df.groupby('Cluster')[features].mean()
    profile['abs_tns'] = profile['setup_tns'].abs()
    sorted_clusters = profile.sort_values(by='abs_tns', ascending=True)
    
    cluster_names = {
        sorted_clusters.index[0]: "Golden (Clean)",
        sorted_clusters.index[1]: "Marginal",
        sorted_clusters.index[2]: "Broken"
    }
    valid_df['Label'] = valid_df['Cluster'].map(cluster_names)
    
    # --- 5. Generate Charts ---
    radar_df = valid_df.copy()
    min_max = MinMaxScaler()
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
    radar_df[cols_to_norm] = min_max.fit_transform(radar_df[cols_to_norm])
    radar_df = radar_df.rename(columns=plot_feats_map)
    final_cats = list(plot_feats_map.values())
    
    radar_centers = radar_df.groupby('Label')[final_cats].mean().reset_index()
    make_radar_chart(radar_centers, final_cats, "Cluster Profiles: Normalized 'Cost' Metrics (Larger = Worse)")

    # Box Plots
    plt.figure(figsize=(15, 10))
    plt.suptitle("Distribution of Metrics by Cluster", fontsize=16)
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
    plt.savefig(OUTPUT_BOX, dpi=300)
    print(f"✅ Box Plots saved to {OUTPUT_BOX}")

if __name__ == "__main__":
    main()