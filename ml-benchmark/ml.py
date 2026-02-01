import pandas as pd
import torch
import torch.nn as nn
import os
import time
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, global_mean_pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

DATA_DIR = "dataset_root" # <--- VERIFY PATH
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">>> Device: {DEVICE}")

# --- 1. DATASET ---
class FusionDataset(Dataset):
    def __init__(self, df, cache, scaler, is_train=True):
        self.df = df
        self.cache = cache
        
        # Scalar Knobs
        p_cols = ['aspect_ratio', 'core_util', 'density', 'synth_strategy', 'io_mode', 'time_driven', 'routability_driven']
        c_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
        
        # TARGETS: Individual physical metrics
        self.target_cols = ['gap_skew', 'gap_power', 'gap_wl'] 
        
        knobs = df[p_cols + c_cols].values
        if is_train:
            self.scaler = StandardScaler()
            self.norm_knobs = self.scaler.fit_transform(knobs)
        else:
            self.scaler = scaler
            self.norm_knobs = self.scaler.transform(knobs)

    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = self.cache[row['placement_id']].clone()
        
        all_k = torch.tensor(self.norm_knobs[idx], dtype=torch.float)
        data.place_knobs = all_k[:7].unsqueeze(0)
        data.cts_knobs   = all_k[7:].unsqueeze(0)
        
        # TARGET: Multi-variate y (Shape: [1, 3])
        targets = row[self.target_cols].values.astype(np.float32)
        data.y = torch.tensor(targets, dtype=torch.float).unsqueeze(0)
        
        return data

# --- 2. CACHE ---
def build_cache(df, col_name):
    print(f"    Caching unique graphs from {col_name}...")
    cache = {}
    unique = df.drop_duplicates(subset='placement_id')
    for i, row in unique.iterrows():
        path = row[col_name]
        if os.path.exists(path):
            cache[row['placement_id']] = torch.load(path, weights_only=False)
    print(f"    ✅ Loaded {len(cache)} unique graphs.")
    return cache

# --- 3. MODEL ---
class FusionModel(nn.Module):
    def __init__(self, backbone, in_channels, hidden_dim=64, dropout=0.5, place_dim=32, cts_dim=16):
        super().__init__()
        self.dropout_rate = dropout
        
        if backbone == 'GCN': self.gnn = GCNConv(in_channels, hidden_dim)
        elif backbone == 'SAGE': self.gnn = SAGEConv(in_channels, hidden_dim)
        elif backbone == 'GATv2': self.gnn = GATv2Conv(in_channels, hidden_dim, heads=1, edge_dim=1)
        
        self.place_mlp = nn.Sequential(nn.Linear(7, place_dim), nn.ReLU())
        self.cts_mlp = nn.Sequential(nn.Linear(4, cts_dim), nn.ReLU())
        
        fusion_in = hidden_dim + place_dim + cts_dim
        # Change output to 3
        self.head = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, 3) 
        )  

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        if isinstance(self.gnn, GATv2Conv):
            out = self.gnn(x, edge_index, edge_attr=edge_attr)
        else:
            out = self.gnn(x, edge_index)
            
        out = out.relu()
        out = nn.functional.dropout(out, p=self.dropout_rate, training=self.training)
        g = global_mean_pool(out, data.batch)
        
        p = self.place_mlp(data.place_knobs)
        p = nn.functional.dropout(p, p=self.dropout_rate, training=self.training)
        
        c = self.cts_mlp(data.cts_knobs)
        c = nn.functional.dropout(c, p=self.dropout_rate, training=self.training)
        
        return self.head(torch.cat([g, p, c], dim=1))

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# --- 4. BENCHMARK ---
def benchmark(col_name, feat_dim, label, hidden_dim=64, place_dim=32, cts_dim=16, lr=0.001):
    print(f"\n>>> STARTING WORKLOAD: {label}")
    full_train_df = pd.read_csv(os.path.join(DATA_DIR, "clocknet_unified_manifest.csv"))
    zero_shot_df = pd.read_csv(os.path.join(DATA_DIR, "clocknet_unified_manifest_test.csv"))
    train_df, val_df = train_test_split(full_train_df, test_size=0.2, random_state=42)

    for c in ['synth_strategy']:
        unique_vals = sorted(train_df[c].dropna().unique())
        cat_type = pd.api.types.CategoricalDtype(categories=unique_vals, ordered=True)
        for d in [train_df, val_df, zero_shot_df]:
            d[c] = d[c].astype(cat_type).cat.codes

    cache = build_cache(full_train_df, col_name)
    cache.update(build_cache(zero_shot_df, col_name))
    
    train_ds = FusionDataset(train_df, cache, None, True)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(FusionDataset(val_df, cache, train_ds.scaler, False), batch_size=32)
    zs_loader = DataLoader(FusionDataset(zero_shot_df, cache, train_ds.scaler, False), batch_size=32)
    
    target_names = ['Skew', 'Power', 'Wire']
    workload_summary = []

    for net in ['GCN', 'SAGE', 'GATv2']:
        print(f"  Profiling {net}...")
        log_data = []
        # Reset peak memory specifically for this GNN run
        if DEVICE.type == 'cuda': torch.cuda.reset_peak_memory_stats()
        
        try:
            model = FusionModel(net, feat_dim, hidden_dim, 0.5, place_dim, cts_dim).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            crit = nn.MSELoss()
            
            start_time = time.time()
            for epoch in range(1, 101):
                model.train()
                for batch in train_loader:
                    batch = batch.to(DEVICE); opt.zero_grad()
                    out = model(batch)
                    loss = crit(out, batch.y); loss.backward(); opt.step()
                
                def get_component_metrics(loader):
                    model.eval(); p_list, t_list = [], []
                    with torch.no_grad():
                        for b in loader:
                            b = b.to(DEVICE)
                            p_list.append(model(b).cpu().numpy())
                            t_list.append(b.y.cpu().numpy().reshape(-1, 3))
                    
                    p_all = np.concatenate(p_list, axis=0)
                    t_all = np.concatenate(t_list, axis=0)
                    r2_comp = r2_score(t_all, p_all, multioutput='raw_values')
                    mae_comp = mean_absolute_error(t_all, p_all, multioutput='raw_values')
                    return r2_comp, mae_comp, np.mean(r2_comp)

                s_r2s, s_maes, s_avg_r2 = get_component_metrics(val_loader)
                u_r2s, u_maes, u_avg_r2 = get_component_metrics(zs_loader)
                
                elapsed = time.time() - start_time
                epoch_log = {"Epoch": epoch, "Time": elapsed, "S_Avg_R2": s_avg_r2, "U_Avg_R2": u_avg_r2}
                
                for i, name in enumerate(target_names):
                    epoch_log[f"S_R2_{name}"] = s_r2s[i]
                    epoch_log[f"S_MAE_{name}"] = s_maes[i] # Added Seen MAE Logging
                    epoch_log[f"U_R2_{name}"] = u_r2s[i]
                    epoch_log[f"U_MAE_{name}"] = u_maes[i]
                
                log_data.append(epoch_log)

                if epoch % 10 == 0:
                    print(f"    Ep {epoch}: Seen R2={s_avg_r2:.2f} | Unseen MAE(Sk/Po/Wi)={u_maes[0]:.4f}/{u_maes[1]:.4f}/{u_maes[2]:.4f}")

            # Capture peak VRAM used during the entire 100-epoch run
            mem_mb = torch.cuda.max_memory_allocated() / 1024**2 if DEVICE.type == 'cuda' else 0
            best_zs = max(log_data, key=lambda x: x['U_Avg_R2'])
            
            workload_summary.append({
                "Graph": label, "Net": net, "VRAM_MB": f"{mem_mb:.1f}",
                "Time_Total": f"{time.time() - start_time:.1f}s",
                "Peak_U_R2_Avg": f"{best_zs['U_Avg_R2']:.3f}",
                "U_R2_Skew": f"{log_data[-1]['U_R2_Skew']:.3f}",
                "U_R2_Power": f"{log_data[-1]['U_R2_Power']:.3f}",
                "U_R2_Wire": f"{log_data[-1]['U_R2_Wire']:.3f}",
                "S_MAE_Wire_Final": f"{log_data[-1]['S_MAE_Wire']:.4f}"
            })
            pd.DataFrame(log_data).to_csv(f"full_log_{label}_{net}.csv", index=False)

        except Exception as e: print(f"Error {net}: {e}")
            
    return workload_summary

if __name__ == "__main__":
    all_summaries = []

    # 1. Clustered Run (High Capacity Config)
    #    This proves efficiency (Low VRAM, Low Time)
    c_summary = benchmark('cluster_graph_path', feat_dim=10, label="Clustered", 
                          hidden_dim=16, place_dim=8, cts_dim=4, lr=0.0005)
    all_summaries.extend(c_summary)
    
    # 2. Raw Run (Standard Config)
    #    This proves accuracy vs cost trade-off
    r_summary = benchmark('raw_graph_path', feat_dim=4, label="Raw", 
                          hidden_dim=64, place_dim=32, cts_dim=16, lr=0.001)
    all_summaries.extend(r_summary)
    
    # 3. Save Final Workload Table
    df = pd.DataFrame(all_summaries)
    df.to_csv("mlbench_workload_summary.csv", index=False)
    
    print("\n\n=== MLBENCH WORKLOAD SUMMARY ===")
    print(df.to_string(index=False))