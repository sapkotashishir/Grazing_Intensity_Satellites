# %% import libraries
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 

# %% load the data from csv file 
data_grid_10m = pd.read_csv('/home/shishir/Documents/Graze_Sat/Results/grazing_satellite_features_grid10m.csv')
# drop rows with NaN values in "au_days" column
# data_grid_10m = data_grid_10m.dropna()
print(f"Data shape: {data_grid_10m.shape}")
print(data_grid_10m.head())
print(data_grid_10m.info())
#keep only bin_idx values 1,2,3,4
data_grid_10m = data_grid_10m[data_grid_10m['bin_idx'].isin([1, 2, 3, 4])]
print(data_grid_10m['bin_idx'].unique())
# keep only rows where "au_days" is not NaN and greater than 0 and less than 3.0
data_grid_10m = data_grid_10m[(data_grid_10m['au_days'].notnull()) & (data_grid_10m['au_days'] > 0) & (data_grid_10m['au_days'] < 3.0)]

# %%
# group by "grid_id" and plot the time series of "grazing_intensity" for each grid_id
grid_ids = data_grid_10m['cell_id'].unique()
print(f"Number of unique grid_ids: {len(grid_ids)}")
print(f"Unique grid_ids: {grid_ids}")
plt.figure(figsize=(12, 6))
for grid_id in grid_ids:
    print(f"Plotting grid_id: {grid_id}")
    grid_data = data_grid_10m[data_grid_10m['cell_id'] == grid_id]
    plt.plot(grid_data['bin_idx'], grid_data['au_days'], label=f'Grid {grid_id}')
plt.xlabel('Bin Index')
plt.ylabel('AU Days')
plt.title('AU Days Time Series for Each Grid ID')
# %%
# for each bin_idx, plot the distribution of "au_days" values
# create separate histograms for each bin_idx in the same figure
axes = []
plt.figure(figsize=(12, 8))
for bin_idx in sorted(data_grid_10m['bin_idx'].unique()):
    ax = plt.subplot(2, 2, bin_idx)
    axes.append(ax)
    bin_data = data_grid_10m[data_grid_10m['bin_idx'] == bin_idx]
    ax.hist(bin_data['au_days'], bins=20, alpha=0.7, color='blue')
    ax.set_title(f'Bin Index {bin_idx}')
    ax.set_xlabel('AU Days')
    ax.set_ylabel('Frequency')

plt.tight_layout()
# %%
