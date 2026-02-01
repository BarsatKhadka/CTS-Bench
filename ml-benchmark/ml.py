import pandas as pd
import torch
import torch.nn as nn
import os
import time
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, global_mean_pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

DATA_DIR = "dataset_root"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">>> Device: {DEVICE}")


class FusionDataset(Dataset):
    def __init__(self, df,  scaler, is_train=True):
        self.df = df
        # self.cache = graph_cache
        
        # We need ALL knobs (Placement + CTS) because the model needs context
        p_cols = ['aspect_ratio', 'core_util', 'density', 'synth_strategy', 'io_mode', 'time_driven', 'routability_driven']
        c_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
        
        # Normalize inputs so MLPs learn easily
        knobs = df[p_cols + c_cols].values
        if is_train:
            self.scaler = StandardScaler()
            self.norm_knobs = self.scaler.fit_transform(knobs)
        else:
            self.scaler = scaler
            self.norm_knobs = self.scaler.transform(knobs)

    def len(self):
        return len(self.df)

    def get(self, idx):
        # 1. LOOKUP: Get the shared graph geometry
        row = self.df.iloc[idx]
        data = self.cache[row['placement_id']].clone()
        
    
        #  first 7 are Placement, last 4 are CTS
        all_k = torch.tensor(self.norm_knobs[idx], dtype=torch.float)
        data.place_knobs = all_k[:7].unsqueeze(0)
        data.cts_knobs   = all_k[7:].unsqueeze(0)
        
        # 3. TARGET: The Pareto Gap
        data.y = torch.tensor([row['pareto_dist']], dtype=torch.float).unsqueeze(0)
        return data

def build_cache(df, col_name):
    print(f"    Caching unique graphs from {col_name}...")
    cache = {}
    unique = df.drop_duplicates(subset='placement_id')
    
    for i, row in unique.iterrows():
        path = row[col_name]
    
        if not os.path.exists(path):
            print(f" Graph file missing: {path}")
            return
        
        if os.path.exists(path):
            cache[row['placement_id']] = torch.load(path, weights_only=False)
            
    print(f"    ✅ Loaded {len(cache)} unique graphs into RAM.")
    return cache

train_df = pd.read_csv(os.path.join(DATA_DIR, "clocknet_unified_manifest.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "clocknet_unified_manifest_test.csv"))

build_cache(train_df, 'raw_graph_path')

