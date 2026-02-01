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


class FusionModel(nn.Module):
    def __init__(self, backbone, in_channels):
        super().__init__()
        # Branch 1: Graph (Geometry)
        if backbone == 'GCN': self.gnn = GCNConv(in_channels, 64)
        elif backbone == 'SAGE': self.gnn = SAGEConv(in_channels, 64)
        elif backbone == 'GATv2': self.gnn = GATv2Conv(in_channels, 64, heads=1, edge_dim=1)
        
        # Branch 2 & 3: Knobs (Context)
        self.place_mlp = nn.Sequential(nn.Linear(7, 32), nn.ReLU())
        self.cts_mlp = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        
        # Fusion Head (Predictor)
        self.head = nn.Sequential(nn.Linear(64+32+32, 64), nn.ReLU(), nn.Linear(64, 1))  
        

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        if isinstance(self.gnn, GATv2Conv):
            out = self.gnn(x, edge_index, edge_attr=edge_attr)
        else:
            out = self.gnn(x, edge_index)
            
        g = global_mean_pool(out.relu(), data.batch)
        p = self.place_mlp(data.place_knobs)
        c = self.cts_mlp(data.cts_knobs)
        return self.head(torch.cat([g, p, c], dim=1))

def benchmark(col_name, feat_dim, label):
    train_df = pd.read_csv(os.path.join(DATA_DIR, "clocknet_unified_manifest.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "clocknet_unified_manifest_test.csv"))

    cat_cols = ['synth_strategy']

    for c in cat_cols:
        cat_type = pd.CategoricalDtype(categories=train_df[c].unique())
        train_df[c] = train_df[c].astype(cat_type).cat.codes
        test_df[c]  = test_df[c].astype(cat_type).cat.codes


    return train_df, test_df

train_df, test_df = benchmark('raw_graph_path', feat_dim=10, label='Raw')











