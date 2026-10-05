import pandas as pd
import numpy as np
from functools import reduce
from datetime import datetime
import os

# Set paths
metrics_dir = './MSDS/concurrent data/metrics'
output_dir = './MSDS/processed'
os.makedirs(output_dir, exist_ok=True)

# Load and preprocess metric files
files = os.listdir(metrics_dir)
dfs = []

for file in files:
    if '.csv' in file and 'wally' in file:
        df = pd.read_csv(os.path.join(metrics_dir, file))
        df = df.drop(columns=['load.cpucore', 'load.min1', 'load.min5', 'load.min15'])
        dfs.append(df)

# Find common time window
start = dfs[0].min()['now']
end = dfs[0].max()['now']
for df in dfs:
    if df.min()['now'] > start:
        start = df.min()['now']
    if df.max()['now'] < end:
        end = df.max()['now']

# Reformat and align
id_vars = ['now']
dfs2 = []
for df in dfs:
    df = df.drop(np.argwhere(list(df['now'] < start)).reshape(-1))
    df = df.drop(np.argwhere(list(df['now'] > end)).reshape(-1))
    melted = df.melt(id_vars=id_vars).dropna()
    df = melted.pivot_table(index=id_vars, columns="variable", values="value")
    dfs2.append(df)
dfs = dfs2

# Merge on time
df_merged = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), dfs)

# Subsample (every 5th row)
df_merged = df_merged.iloc[::5]

# Format datetime index
new_index = []
for i in df_merged.index:
    dt = datetime.strptime(i[:-5], '%Y-%m-%d %H:%M:%S')
    new_index.append(dt.strftime('%Y-%m-%dT%H:%M:%SZ'))
df_merged.index = new_index

# Drop first 10%, then split
start = round(df_merged.shape[0] * 0.1)
df_merged = df_merged[start:]
split = round(df_merged.shape[0] / 2)

# Save CSVs
df_merged[:split].to_csv(os.path.join(output_dir, 'train.csv'))
df_merged[split:].to_csv(os.path.join(output_dir, 'test.csv'))

# Dummy labels (all 0s) for test set
d = pd.DataFrame(0, index=np.arange(df_merged[split:].shape[0]), columns=df_merged.columns)
d.to_csv(os.path.join(output_dir, 'test_label.csv'))

# Print shapes
print("Data saved to:", output_dir)
print("Shape of train.csv:", df_merged[:split].shape)
print("Shape of test.csv:", df_merged[split:].shape)
print("Shape of labels.csv:", d.shape)
# Create compact test_labels with one label per timestep
# Rule: if any feature at a time step has a 1 → label = 1
time_steps = df_merged[split:].index  # same timestamps as test set
labels_compact = (d.max(axis=1) > 0).astype(int)  # 1 if any column is 1, else 0

# Create and save the compact label file
df_test_label = pd.DataFrame({
    'timestep': time_steps,
    'label': labels_compact
})
df_test_label.to_csv(os.path.join(output_dir, 'test_localization_labels.csv'), index=False)

print("🧾 test_localization_labels saved with shape:", df_test_label.shape)
