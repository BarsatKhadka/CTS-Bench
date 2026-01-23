import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

# --- CONFIGURATION ---
DATASET_ROOT = "dataset_root"

# List all your CSV files here
CSV_FILES = [
    os.path.join(DATASET_ROOT, "experiment_log.csv"),
    os.path.join(DATASET_ROOT, "picorv32_batch1.csv")
]

OUTPUT_PLOT_PCA = "design_space_pca.png"
OUTPUT_PLOT_TSNE = "design_space_tsne.png"

def get_graph_embedding(file_path):
    """
    Loads a Clustered Graph .pt file and calculates a fixed-size 'Geometric Fingerprint'.
    """
    try:
        data = torch.load(file_path)
        
        # [cx, cy, sx, sy, size_log, n_ff, n_logic, ...] 
        x = data.x.numpy()
        
        # "Fingerprint" = Mean + Std Dev of all node features
        feat_means = np.mean(x, axis=0)
        feat_stds = np.std(x, axis=0)
        
        embedding = np.concatenate([feat_means, feat_stds])
        return embedding
        
    except Exception as e:
        # It is common to miss a few files if a run crashed, so just print warning
        print(f"   ⚠️ Error loading {file_path}: {e}")
        return None

def main():
    print("🔹 1. Loading and Merging Log Files...")
    
    df_list = []
    for csv_file in CSV_FILES:
        if os.path.exists(csv_file):
            print(f"   - Found {csv_file}")
            df_list.append(pd.read_csv(csv_file))
        else:
            print(f"   ❌ Warning: File not found: {csv_file}")
            
    if not df_list:
        print("No CSV files loaded. Exiting.")
        return

    # Combine into one big DataFrame
    df = pd.concat(df_list, ignore_index=True)
    
    # We only want ONE entry per unique placement
    # The graph structure is identical for all 10 CTS runs, so we drop duplicates.
    unique_placements = df.drop_duplicates(subset=['placement_id'])
    
    print(f"   Total Runs: {len(df)}")
    print(f"   Unique Placements to Analyze: {len(unique_placements)}")

    # --- Extract Features ---
    embeddings = []
    valid_indices = []
    
    print("\n🔹 2. Extracting Geometric Features from Graphs...")
    # Iterate through the unique placements
    count = 0
    for idx, row in unique_placements.iterrows():
        path = row['cluster_graph_path']
        
        # Handle relative paths just in case
        if not os.path.isabs(path):
            # If path starts with 'dataset_root', and we are running FROM 'dataset_root', 
            # we might need to adjust. Assuming script runs from project root:
            if not os.path.exists(path):
                # Try prepending current dir if needed, or leave as is
                pass 

        if os.path.exists(path):
            emb = get_graph_embedding(path)
            if emb is not None:
                embeddings.append(emb)
                valid_indices.append(idx)
        else:
            # Only print warning for the first few missing files to avoid spam
            if count < 5: print(f"   ⚠️ File not found: {path}")
            count += 1

    if not embeddings:
        print("No valid embeddings found to plot.")
        return

    X = np.array(embeddings)
    
    # Standardize features (Mean=0, Std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- PCA Analysis ---
    print("\n🔹 3. Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_
    print(f"   PCA Variance Captured: PC1={var[0]:.2f}, PC2={var[1]:.2f}")

    # --- Plotting PCA ---
    plt.figure(figsize=(10, 8))
    
    # We use 'valid_indices' to make sure we only grab metadata for graphs that actually loaded
    metadata = unique_placements.loc[valid_indices]
    
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=metadata['aspect_ratio'].astype(str), # Color by AR
        style=metadata['core_util'].astype(str),  # Shape by Density
        s=80, alpha=0.8, palette="viridis"
    )
    
    plt.title(f"Design Space Diversity (PCA)\n{len(embeddings)} Unique Placements")
    plt.xlabel(f"PC1 ({var[0]:.0%} var)")
    plt.ylabel(f"PC2 ({var[1]:.0%} var)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PCA, dpi=300)
    print(f"✅ PCA Plot saved to {OUTPUT_PLOT_PCA}")

    # --- t-SNE Analysis ---
    print("\n🔹 4. Running t-SNE...")
    # Perplexity adjustment for small datasets
    perp = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=metadata['synth_strategy'].astype(str), # Try Strategy for t-SNE color
        s=80, alpha=0.8, palette="deep"
    )
    plt.title("Design Space Manifold (t-SNE)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_TSNE, dpi=300)
    print(f"✅ t-SNE Plot saved to {OUTPUT_PLOT_TSNE}")

if __name__ == "__main__":
    main()