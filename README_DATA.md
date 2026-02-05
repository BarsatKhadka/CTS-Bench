# CTS-Bench: Graph Coarsening Benchmark for GNN Efficiency in Physical Design

A comprehensive benchmark suite measuring the trade-off between graph resolution and learning efficiency in Electronic Design Automation (EDA), with a focus on the Placement–Clock Tree Synthesis (CTS) interface.

## Dataset Overview

**CTS-Bench** contains **4,860 converged physical design solutions** across five hardware architectures, providing paired Raw (gate-level) and Clustered graph representations with ground-truth CTS metrics.

### Key Statistics
- **Total Data Points**: 4,860 designs
- **Architectures**: PicoRV32, AES, SHA256, EthMAC (training), Zipdiv (zero-shot test)
- **Placements per Architecture**: 486 unique placements (except Zipdiv: 500)
- **CTS Variants per Placement**: 10 distinct clock trees
- **Graph Representations**: Raw gate-level + Physics-aware clustered (13.3× compression)
- **Metrics**: 15 ground-truth QoR measurements per run

## Data Structure

### Primary Manifests

Located in `dataset_root/`:

- **`clocknet_unified_manifest.csv`** - Main training dataset
  - 3,860 samples from PicoRV32, AES, SHA256, EthMAC
  - 80/20 train/validation split for in-distribution evaluation
  - Includes: placement parameters, CTS knobs, 15 QoR metrics, graph paths

- **`clocknet_unified_manifest_test.csv`** - Zero-shot generalization test set
  - 500 samples from Zipdiv architecture
  - Structurally distinct from training designs
  - Evaluates out-of-distribution generalization capability

### CSV Column Structure

Both manifests contain:

| Column Category | Content |
|---|---|
| **Metadata** | design_id, placement_seed, cts_variant |
| **Placement Knobs** | synth_strategy, aspect_ratio, io_mode, core_utilization, target_density, time_driven, routability_driven |
| **CTS Knobs** | sink_max_dia, max_wire_length, cluster_size, buffer_distance |
| **QoR Metrics** | skew_setup, skew_hold, slack, tns, total_power, dynamic_power, static_power, wirelength, cell_utilization, clock_buffers, inverters, routing_congestion, repair_buffers |
| **Gap Metrics** | pareto_gap_skew, pareto_gap_power, pareto_gap_wirelength, total_pareto_distance |
| **Graph Paths** | path_raw_graph, path_clustered_graph |

## Graph Representations

### Raw Graphs
- **Node Type**: Standard cells (flip-flops + combinational logic)
- **Node Count**: 5,000–40,000 nodes per design
- **Node Features**: 
  - Normalized x, y coordinates
  - is_ff (binary: flip-flop indicator)
  - log-transformed switching activity (from SAIF)
- **Edge Features**: Manhattan distance between connected cells
- **Format**: PyTorch Geometric tensors

### Clustered Graphs
- **Compression Ratio**: ~13.3× average reduction
- **Clustering Algorithm**: 
  1. Atomic cluster formation (BFS from flip-flops)
  2. High-spread filtering (preserve geometric diversity)
  3. Gravity-vector-aligned merging (electrical domain awareness)
- **Node Count**: 400–3,000 nodes per design
- **Node Features** (10 dimensions):
  - Cluster centroid (x, y)
  - Spatial spread (σ_x, σ_y)
  - log-transformed cluster size
  - Composition (N_FF, N_Logic)
  - Aggregated switching activity (max, sum, toggle counts)
- **Edge Features**: Manhattan distance between cluster centroids
- **Format**: PyTorch Geometric tensors

## Directory Structure

```
CTS-Knob-Aware-placement/
├── dataset_root/                    # Main data directory
│   ├── clocknet_unified_manifest.csv           # Training manifest (3,860 rows)
│   ├── clocknet_unified_manifest_test.csv      # Test manifest (500 rows, Zipdiv)
│   ├── aes_batch1-3.csv            # AES design subsets
│   ├── ethmac_batch1-4.csv         # EthMAC design subsets
│   ├── picorv32_batch1-2.csv       # PicoRV32 design subsets
│   ├── sha256_batch1-4.csv         # SHA256 design subsets
│   ├── test_zipdiv.csv             # Zipdiv test data
│   ├── graphs/                      # Graph files (raw + clustered)
│   │   ├── aes/
│   │   ├── ethmac/
│   │   ├── picorv32/
│   │   ├── sha256/
│   │   └── zipdiv/
│   └── experiment_log.csv          # Training run logs
├── designs/                         # RTL & design specifications
│   ├── aes/, ethmac/, picorv32/, sha256/, zipdiv/
│   ├── base.sdc, primitives.v, sky130_fd_sc_hd.v
├── clustering_outputs_GMM/          # Clustering analysis
│   ├── 00_AGGREGATED_COUNTS/
│   └── [per-design directories]
├── fidelity_outputs/                # Fidelity evaluation results
│   ├── 00_ALL_DESIGNS/
│   └── [per-design directories]
├── correlation_outputs/             # Correlation analysis
│   ├── 00_ALL_DESIGNS/
│   └── [per-design directories]
├── stats_outputs/                   # Statistical summaries
├── runs/                            # Training experiment runs
│   └── zipdiv_run_YYYYMMDD_HHMMSS/  # Timestamped results
├── scripts/                         # Generation & analysis scripts
├── data_analysis/                   # Visualization & metric analysis
│   ├── raw-mae-chart.py
│   ├── clustered-mae-chart.py
│   ├── clustering_fidelity_check.py
│   ├── variance.py
│   └── [other analysis scripts]
├── ml-benchmark/                    # ML training code
│   └── ml.py                        # Multi-modal GNN training
└── requirements.txt                 # Python dependencies
```

## Design-Specific Data

Individual design data files are available for granular analysis:

```
dataset_root/
├── aes_batch1.csv, aes_batch2.csv, aes_batch3.csv
├── ethmac_batch1.csv, ethmac_batch2.csv, ethmac_batch3.csv, ethmac_batch4.csv
├── picorv32_batch1.csv, picorv32_batch2.csv
├── sha256_batch1.csv, sha256_batch2.csv, sha256_batch3.csv, sha256_batch4.csv
└── test_zipdiv.csv
```

Each file contains the same column structure as the main manifest but filtered for that specific architecture.

## Key Findings & Benchmarks

### Efficiency Trade-off
- **VRAM Reduction**: 17.2× (Raw → Clustered)
- **Training Speedup**: 3× 
- **Memory Usage**:
  - Raw GATv2: 1.2–2.5 GB/batch
  - Clustered GATv2: ~70–150 MB/batch

## Usage Examples

### Loading & Exploring the Data

```python
import pandas as pd
import torch_geometric

# Load main training manifest
df_train = pd.read_csv('dataset_root/clocknet_unified_manifest.csv')
print(f"Training samples: {len(df_train)}")
print(f"Columns: {df_train.columns.tolist()}")

# Load zero-shot test set
df_test = pd.read_csv('dataset_root/clocknet_unified_manifest_test.csv')
print(f"Test samples (Zipdiv): {len(df_test)}")

# Access graph paths
raw_graph_path = df_train.iloc[0]['path_raw_graph']
clustered_graph_path = df_train.iloc[0]['path_clustered_graph']

# Load graph data (example using PyTorch Geometric)
raw_data = torch.load(f'dataset_root/{raw_graph_path}')
clustered_data = torch.load(f'dataset_root/{clustered_graph_path}')

print(f"Raw graph nodes: {raw_data.num_nodes}, edges: {raw_data.num_edges}")
print(f"Clustered graph nodes: {clustered_data.num_nodes}, edges: {clustered_data.num_edges}")
```

### Filtering by Architecture

```python
# Get only AES data
df_aes = pd.read_csv('dataset_root/aes_batch1.csv')

# Get only PicoRV32 data
df_picorv32 = pd.read_csv('dataset_root/picorv32_batch1.csv')

# Combine multiple batches
df_sha256 = pd.concat([
    pd.read_csv('dataset_root/sha256_batch1.csv'),
    pd.read_csv('dataset_root/sha256_batch2.csv'),
    pd.read_csv('dataset_root/sha256_batch3.csv'),
    pd.read_csv('dataset_root/sha256_batch4.csv'),
])
```

### QoR Metric Analysis

```python
# Analyze skew metrics
skew_seen = df_train['skew_setup'].describe()
print(f"Skew statistics (seen architectures):\n{skew_seen}")

# Compute Pareto efficiency
df_train['is_pareto_optimal'] = (
    (df_train['pareto_gap_skew'] == 1.0) &
    (df_train['pareto_gap_power'] == 1.0) &
    (df_train['pareto_gap_wirelength'] == 1.0)
)
print(f"Pareto optimal designs: {df_train['is_pareto_optimal'].sum()}")

# Compare raw vs clustered graph compression
df_train['compression_ratio'] = (
    df_train['raw_node_count'] / df_train['clustered_node_count']
)
print(f"Average compression: {df_train['compression_ratio'].mean():.1f}×")
```

## ML Training

Refer to [ml-benchmark/ml.py](ml-benchmark/ml.py) for training multi-modal GNN models on CTS-Bench.

**Training Configuration**:
- Models: GCN, GraphSAGE, GATv2
- Task: Multi-target regression (Skew, Power, Wirelength)
- Epochs: 100
- Batch Size: 32
- Hardware: NVIDIA RTX 5060 (8GB VRAM)

## Analysis Scripts

Located in `data_analysis/`:

- `raw-mae-chart.py` - Visualize MAE and R² for raw graphs
- `clustered-mae-chart.py` - Visualize MAE and R² for clustered graphs
- `clustering_fidelity_check.py` - Evaluate clustering quality
- `variance.py` - Variance analysis across designs
- `visualize-correlations.py` - Metric correlation heatmaps


---

**Last Updated**: February 2026
