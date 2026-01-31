import pandas as pd

# Load the dataset
df = pd.read_csv('dataset_root/main_dataset.csv')

# 1. Identify the 'Best Case' (Minima) for each specific design
# This treats each architecture (AES, PicoRV32, etc.) as its own system environment
design_baselines = df.groupby('design_name').agg({
    'skew_setup': 'min',
    'power_total': 'min',
    'wirelength': 'min'
}).rename(columns={
    'skew_setup': 'min_skew',
    'power_total': 'min_power',
    'wirelength': 'min_wl'
})

# Display the baselines (the 1.0 anchor points for your Pareto analysis)
print("Design-Specific Anchor Points (Minima):")
print(design_baselines)

# 2. Merge these design-specific baselines back into the main data
df = df.merge(design_baselines, on='design_name')

# 3. Calculate the Gap Ratios (Distance from the design-best)
# 1.0 means the run is the best in its class; > 1.0 indicates a performance penalty
df['gap_skew'] = df['skew_setup'] / df['min_skew']
df['gap_power'] = df['power_total'] / df['min_power']
df['gap_wl'] = df['wirelength'] / df['min_wl']

# Select relevant columns to view the results
# This allows you to see how 'far' each run is from the best-case of its own design
output_preview = df[['run_id', 'design_name', 'gap_skew', 'gap_power', 'gap_wl']]
print("\nPreview of Normalized Gap Ratios:")
print(output_preview.head(10))

# Save the baseline-normalized data
df.to_csv('clocknet_design_normalized.csv', index=False)