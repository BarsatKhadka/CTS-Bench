import pandas as pd

# Load the dataset
df = pd.read_csv('dataset_root/main_dataset.csv')

# 1. Identify the 'Best Case' (Minima) for each specific design
# This treats each architecture as its own system environment
design_baselines = df.groupby('design_name').agg({
    'skew_setup': 'min',
    'power_total': 'min',
    'wirelength': 'min'
}).rename(columns={
    'skew_setup': 'min_skew',
    'power_total': 'min_power',
    'wirelength': 'min_wl'
})

# Merge these design-specific baselines back into the main data
df = df.merge(design_baselines, on='design_name')

# 2. Calculate the Gap Ratios
# 1.0 means it is the best for that design; >1.0 means it is less efficient
df['gap_skew'] = df['skew_setup'] / df['min_skew']
df['gap_power'] = df['power_total'] / df['min_power']
df['gap_wl'] = df['wirelength'] / df['min_wl']

# 3. Identify the Primary Bottleneck per run
# We find which of the three metrics has the LARGEST gap (deviated most from its best)
df['max_gap'] = df[['gap_skew', 'gap_power', 'gap_wl']].max(axis=1)

# 4. Normalize by the Row Max (Bottleneck Normalization)
# This forces the "Worst" metric in every row to be 1.0, highlighting the bottleneck
df['norm_gap_skew'] = df['gap_skew'] / df['max_gap']
df['norm_gap_power'] = df['gap_power'] / df['max_gap']
df['norm_gap_wl'] = df['gap_wl'] / df['max_gap']

# 5. Calculate Pareto Efficiency Score (Euclidean Distance from [1,1,1])
# Lower is better (closer to the theoretical 'perfect' for that design)
df['pareto_dist'] = ((df['gap_skew']-1)**2 + (df['gap_power']-1)**2 + (df['gap_wl']-1)**2)**0.5

# Select and save the relevant benchmarking columns
output_columns = [
    'run_id', 'design_name', 'gap_skew', 'gap_power', 'gap_wl',
    'norm_gap_skew', 'norm_gap_power', 'norm_gap_wl', 'pareto_dist'
]

df[output_columns].to_csv('clocknet_pareto_benchmarks.csv', index=False)

print("Pareto Gap Analysis complete. Saved to 'clocknet_pareto_benchmarks.csv'")