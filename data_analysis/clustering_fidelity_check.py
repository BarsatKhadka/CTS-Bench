import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data

# --- CONFIGURATION ---
DATASET_ROOT = "./dataset_root"
CSV_FILES = [
    os.path.join(DATASET_ROOT, "experiment_log.csv"),
    os.path.join(DATASET_ROOT, "picorv32_batch1.csv")
]
OUTPUT_PLOT = "clustering_fidelity_check.png"

def get_graph_stats(path):
    """Loads graph and returns basic topology stats."""
    try:
        # FIX: weights_only=False for security check bypass
        data = torch.load(path, weights_only=False)
        
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
        
        # Calculate Center of Mass (Mean of X, Y coords)
        # Assuming pos is in the first 2 columns of x
        pos_mean = data.x[:, :2].mean(dim=0).numpy()
        print(pos_mean)
        
        return {
            "nodes": num_nodes,
            "edges": num_edges,
            "degree": avg_degree,
            "cx": pos_mean[0],
            "cy": pos_mean[1]
        }
    except Exception as e:
        return None

def main():
    print("🔹 1. Loading Logs...")
    df_list = []
    for f in CSV_FILES:
        if os.path.exists(f):
            df_list.append(pd.read_csv(f))
    
    if not df_list: return
    df = pd.concat(df_list, ignore_index=True)
    
    # Drop duplicates to analyze unique placements only
    unique_df = df.drop_duplicates(subset=['placement_id'])
    print(f"   Analyzing {len(unique_df)} unique graph pairs...")

    stats = []

    print("🔹 2. Comparing Raw vs. Clustered...")
    for idx, row in unique_df.iterrows():
        raw_path = row['raw_graph_path']
        cluster_path = row['cluster_graph_path']
        
        # Handle relative paths if needed
        if not os.path.exists(raw_path): continue
        if not os.path.exists(cluster_path): continue
        
        r_stats = get_graph_stats(raw_path)
        c_stats = get_graph_stats(cluster_path)
        
        if r_stats and c_stats:
            stats.append({
                "placement_id": row['placement_id'],
                "raw_nodes": r_stats['nodes'],
                "cluster_nodes": c_stats['nodes'],
                "compression_ratio": r_stats['nodes'] / c_stats['nodes'],
                "raw_degree": r_stats['degree'],
                "cluster_degree": c_stats['degree'],
                "center_shift": ((r_stats['cx'] - c_stats['cx'])**2 + (r_stats['cy'] - c_stats['cy'])**2)**0.5
            })

    results = pd.DataFrame(stats)
    
    # --- 3. Plotting the Fidelity Report ---
    plt.figure(figsize=(14, 6))

    # Plot A: Compression Consistency
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=results, x="raw_nodes", y="cluster_nodes", alpha=0.7)
    # Add a "Perfect Linearity" reference line
    m, b = np.polyfit(results["raw_nodes"], results["cluster_nodes"], 1)
    plt.plot(results["raw_nodes"], m*results["raw_nodes"] + b, color="red", linestyle="--", label=f"Trend (Ratio ~ {results['compression_ratio'].mean():.1f}x)")
    plt.title("Compression Consistency\n(Raw vs Clustered Node Count)")
    plt.xlabel("Raw Nodes (Ground Truth)")
    plt.ylabel("Clustered Nodes (Proxy)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot B: Topology Preservation (Degree)
    plt.subplot(1, 2, 2)
    sns.histplot(results["cluster_degree"], kde=True, color="green", label="Clustered")
    sns.histplot(results["raw_degree"], kde=True, color="blue", label="Raw")
    plt.title("Topology Preservation\n(Average Node Degree Distribution)")
    plt.xlabel("Avg Degree (Connections per Node)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"✅ Fidelity Report saved to {OUTPUT_PLOT}")
    print(f"   Avg Compression Ratio: {results['compression_ratio'].mean():.2f}x")
    print(f"   Avg Center Mass Shift: {results['center_shift'].mean():.4f} units")

import numpy as np # Needed for polyfit
if __name__ == "__main__":
    main()