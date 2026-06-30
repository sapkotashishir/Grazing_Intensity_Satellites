#%%
# import libraries 
import os 
import numpy as np 
import pandas as pd 
import geopandas as gpd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
print("Libraries imported successfully")

# %%
# GPS IMPORT & BASIC CLEANING
# =============================================================================
GPS_PATH = "/home/shishir/Documents/Graze_Sat/Data/GPS_cattle_move_2025.csv"
Plot_DIR = "/home/shishir/Documents/Graze_Sat/Results/Plots"
os.makedirs(Plot_DIR, exist_ok=True)
gps_raw = pd.read_csv(GPS_PATH)

gps = (
    gps_raw
    .rename(columns={
        'AnimalID'        : 'animal_id',
        'Ingress Date (UTC)': 'time',
        'Latitude'        : 'lat',
        'Longitude'       : 'lon',
    })
    .copy()
)

gps['time'] = pd.to_datetime(gps['time'], utc=True, errors='coerce')

# Remove impossible coordinates and null rows
valid_coords = (
    gps['lat'].between(-90, 90)
    & gps['lon'].between(-180, 180)
    & ~((gps['lat'] == 0) & (gps['lon'] == 0))
)
gps = gps[valid_coords].dropna(subset=['animal_id', 'time']).copy()
gps['date'] = gps['time'].dt.floor('D')

print(f"Raw fixes after coordinate cleaning : {len(gps):,}")
print(f"Unique animals                       : {gps['animal_id'].nunique()}")
print(f"Date range                           : {gps['date'].min().date()}  →  {gps['date'].max().date()}")

# %%
# IMPORT THE 10 M GRID & SPATIAL CLEANING
# =============================================================================
# [6] — PLOT GPS FIXES OVER SENTINEL-1 GRID
# =============================================================================

# Convert GPS table to GeoDataFrame
gps_gdf = gpd.GeoDataFrame(
    gps,
    geometry=gpd.points_from_xy(gps['lon'], gps['lat']),
    crs='EPSG:4326'
)
cells_gdf = gpd.read_file("/home/shishir/Documents/Graze_Sat/Data/grid_10m/s1_grid.shp")
s1_crs = cells_gdf.crs

# Reproject to Sentinel-1 CRS
gps_gdf = gps_gdf.to_crs(s1_crs)

# Spatial join: keep only points that intersect a grid cell
gps_in_grid = gpd.sjoin(
    gps_gdf,
    cells_gdf[['cell_id', 'geometry']],
    how='inner',
    predicate='within'
)

# Plotting Grid and Paddock Boundary
fig, ax = plt.subplots(figsize=(10,10))

# Grid
cells_gdf.plot(
    ax=ax,
    facecolor='none',
    edgecolor='lightgray',
    linewidth=0.3
)
# load paddock boundary
paddock_10 = gpd.read_file("/home/shishir/Documents/Graze_Sat/Data/paddock_10.shp")
# Paddock boundary
paddock_10.to_crs(s1_crs).plot(
    ax=ax,
    facecolor='none',
    edgecolor='red',
    linewidth=2
)

gps_in_grid.plot(
    ax=ax,
    color='blue',
    markersize=1,
    alpha=0.2
)

ax.set_title('Animal GPS fixes over Sentinel-1 grid')
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')
ax.set_aspect('equal')

plt.tight_layout()
plt.show()

print(f"GPS fixes inside grid: {len(gps_in_grid):,}")
print(f"Unique animals: {gps_in_grid['animal_id'].nunique()}")

# How many fixes does each animal actually have?
counts = gps_in_grid['animal_id'].value_counts()
print(counts.describe())
print("Animals with <5 fixes:", (counts < 5).sum())

# Drop animals with too few fixes
MIN_FIXES = 5
keep_ids = counts[counts >= MIN_FIXES].index
dropped = counts[counts < MIN_FIXES]
print(f"Dropping {len(dropped)} animals:", dropped.index.tolist())

# %%
# MOVEMENT METRICS (TIME GAP, DISTANCE, SPEED) & FILTERING
# =============================================================================

# FIX: Filter 'gps_in_grid' directly using the verified keep_ids to resolve the overwrite bug
gps = gps_in_grid[gps_in_grid['animal_id'].isin(keep_ids)].sort_values(['animal_id', 'time']).reset_index(drop=True)
print("Remaining after filtering low-fix animals:", gps.shape, gps['animal_id'].nunique(), "animals")

# Time gap between consecutive fixes (per animal)
gps['dt_seconds'] = (
    gps.groupby('animal_id')['time']
    .diff()
    .dt.total_seconds()
    .fillna(0)
)
gps['time_local'] = gps['time'].dt.tz_convert('US/Central')
gps['hour_local'] = gps['time_local'].dt.hour
mean_dt   = gps['dt_seconds'].mean()
median_dt = gps['dt_seconds'].median()
print(f"Mean fix interval   : {mean_dt/60:.2f} minutes")
print(f"Median fix interval : {median_dt/60:.2f} minutes")

def haversine_meters(lon1, lat1, lon2, lat2):
    """Vectorised Haversine distance in metres."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6_367_000 * 2 * np.arcsin(np.sqrt(a))   # metres

gps['prev_lat'] = gps.groupby('animal_id')['lat'].shift(1)
gps['prev_lon'] = gps.groupby('animal_id')['lon'].shift(1)

gps['distance_meters'] = haversine_meters(
    gps['lon'], gps['lat'], gps['prev_lon'], gps['prev_lat']
).fillna(0)

gps['speed_mps'] = np.where(
    gps['dt_seconds'] > 0,
    gps['distance_meters'] / gps['dt_seconds'],
    0
)

gps = gps.drop(columns=['prev_lat', 'prev_lon'])

# --- GPS quality-control thresholds (data-driven + biological floor) ---
SPEED_P995 = gps['speed_mps'].quantile(0.995)
GAP_P995   = gps['dt_seconds'][gps['dt_seconds'] > 0].quantile(0.995)

MAX_SPEED_MPS    = max(SPEED_P995, 3.0)
MAX_TIME_GAP_SEC = max(GAP_P995, 10 * 60)

print(f"Final speed threshold : {MAX_SPEED_MPS:.2f} m/s")
print(f"Final gap threshold   : {MAX_TIME_GAP_SEC/60:.1f} min")

# Apply flags
gps['flag_long_gap']         = gps['dt_seconds'] > MAX_TIME_GAP_SEC
gps['flag_impossible_speed'] = gps['speed_mps']  > MAX_SPEED_MPS

flagged_gap   = gps['flag_long_gap'].mean() * 100
flagged_speed = gps['flag_impossible_speed'].mean() * 100
flagged_either = (gps['flag_long_gap'] | gps['flag_impossible_speed']).mean() * 100
print(f"Flagged — long gap : {flagged_gap:.2f}%")
print(f"Flagged — impossible speed : {flagged_speed:.2f}%")
print(f"Flagged — either condition : {flagged_either:.2f}%")

# Filter and clean up
gps_clean = gps[~gps['flag_long_gap'] & ~gps['flag_impossible_speed']].copy()
print(f"Clean fixes retained : {len(gps_clean):,} ({len(gps_clean)/len(gps)*100:.1f}%)")
print(f"Unique animals before: {gps['animal_id'].nunique()}")
print(f"Unique animals after : {gps_clean['animal_id'].nunique()}")

# %%
# PER-ANIMAL EXPLORATORY BOX PLOTS
# =============================================================================
animal_order = sorted(gps_clean['animal_id'].unique())

# Drop synthetic zeros (first fix per animal)
gps_plot = gps_clean[gps_clean['dt_seconds'] > 0].copy()
gps_plot['dt_minutes'] = gps_plot['dt_seconds'] / 60.0

fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# --- Speed ---
sns.boxplot(
    data=gps_plot, x='animal_id', y='speed_mps',
    order=animal_order, ax=axes[0],
    showfliers=True, fliersize=2, color='steelblue'
)
axes[0].set_yscale('log')
axes[0].set_ylabel('Speed (m/s, log scale)')
axes[0].set_title('Per-Animal Speed Distribution')
axes[0].axhline(3.0, color='red', linestyle='--', alpha=0.6, label='3 m/s reference')
axes[0].legend(loc='upper right')

# --- Distance ---
sns.boxplot(
    data=gps_plot, x='animal_id', y='distance_meters',
    order=animal_order, ax=axes[1],
    showfliers=True, fliersize=2, color='darkorange'
)
axes[1].set_yscale('log')
axes[1].set_ylabel('Distance between fixes (m, log scale)')
axes[1].set_title('Per-Animal Step-Distance Distribution')

# --- Time gap ---
sns.boxplot(
    data=gps_plot, x='animal_id', y='dt_minutes',
    order=animal_order, ax=axes[2],
    showfliers=True, fliersize=2, color='seagreen'
)
axes[2].set_yscale('log')
axes[2].set_ylabel('Time gap (min, log scale)')
axes[2].set_xlabel('Animal ID')
axes[2].set_title('Per-Animal Time-Gap Between Consecutive Fixes')

# Cadence reference lines
for ref_min, lbl in [(5, '5 min'), (15, '15 min'), (60, '1 h')]:
    axes[2].axhline(ref_min, color='black', linestyle=':', alpha=0.4)
    axes[2].text(len(animal_order) - 1, ref_min, f' {lbl}',
                 va='center', ha='left', fontsize=8, color='black')

plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(Plot_DIR, 'per_animal_speed_distance_dt_box.png'),
            dpi=200, bbox_inches='tight')
plt.show()

# %%
# FEATURE ENGINEERING & K-MEANS CLUSTERING
# =============================================================================

# Combined acceleration variability magnitude
gps_clean['accel_sd_mag'] = np.sqrt(
    gps_clean['StdDevAccelerationX']**2 +
    gps_clean['StdDevAccelerationY']**2 +
    gps_clean['StdDevAccelerationZ']**2
)

# Log-transform speed to tame the right-skew
gps_clean['speed_log'] = np.log1p(gps_clean['speed_mps'].clip(lower=0))

# Z-score each feature within animal_id
def zscore(s):
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std

gps_clean['accel_sd_mag_z'] = gps_clean.groupby('animal_id')['accel_sd_mag'].transform(zscore)
gps_clean['speed_log_z'] = gps_clean.groupby('animal_id')['speed_log'].transform(zscore)
gps_clean[['accel_sd_mag_z', 'speed_log_z']] = gps_clean[['accel_sd_mag_z', 'speed_log_z']].fillna(0)

# Fit K-Means with k=3
feature_cols = ['accel_sd_mag_z', 'speed_log_z']
X = gps_clean[feature_cols].values

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
gps_clean['cluster'] = kmeans.fit_predict(X)

print("\nCluster counts:")
print(gps_clean['cluster'].value_counts())
print("\nCluster centers:\n", kmeans.cluster_centers_)

# Map cluster numbers to behavior names dynamically by mean speed rank

centers = kmeans.cluster_centers_
order = np.argsort(centers[:, feature_cols.index('speed_log_z')])
label_map = {int(order[0]): 'resting', int(order[1]): 'grazing', int(order[2]): 'traveling'}
print("\nDynamic Cluster Label Mapping:", label_map)

gps_clean['behavior_pred'] = gps_clean['cluster'].map(label_map)
print("\nPredicted Behavior Distribution:")
print(gps_clean['behavior_pred'].value_counts())

# Evaluate clustering structure via Silhouette Score
idx = np.random.RandomState(42).choice(len(X), min(20000, len(X)), replace=False)
sil = silhouette_score(X[idx], gps_clean['cluster'].values[idx])
print(f"\nSilhouette score (sampled): {sil:.4f}")

# Diurnal sanity check
if 'hour_local' in gps_clean.columns:
    print("\nDiurnal Activity Profile (Proportion by Hour):")
    diurnal = pd.crosstab(gps_clean['hour_local'], gps_clean['behavior_pred'], normalize='index')
    print(diurnal.round(2))

# %%
# VISUALIZE CLUSTERING SPACE WITH CENTROIDS
# =============================================================================
feat_x = 'speed_log_z'
feat_y = 'accel_sd_mag_z'

colors = {'resting': '#888780', 'grazing': '#1D9E75', 'traveling': '#D85A30'}

fig, ax = plt.subplots(figsize=(9, 7))

# --- Scatter each behavior cluster ---
for behavior, color in colors.items():
    sub = gps_clean[gps_clean['behavior_pred'] == behavior]
    sub = sub.sample(min(2000, len(sub)), random_state=42)
    ax.scatter(sub[feat_x], sub[feat_y],
               s=10, alpha=0.4, color=color, label=behavior, edgecolor='none')

# --- Overlay K-Means centroids ---
centers = kmeans.cluster_centers_  
for cid, (acc_z, spd_z) in enumerate(centers):
    behavior = label_map[cid] 
    ax.scatter(spd_z, acc_z,
               s=300, color=colors[behavior],
               edgecolor='black', linewidth=2, zorder=5)
    ax.annotate(f' {behavior}\n centroid',
                (spd_z, acc_z), fontsize=9, fontweight='bold',
                xytext=(8, 8), textcoords='offset points')

ax.set_xlabel('speed_log_z  (z-scored log-speed)')
ax.set_ylabel('accel_sd_mag_z  (z-scored accel std-dev magnitude)')
ax.set_title('Real K-Means Clustering Space\n(features used by the algorithm, not raw values)')
ax.axhline(0, color='gray', linestyle=':', alpha=0.4)
ax.axvline(0, color='gray', linestyle=':', alpha=0.4)
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(Plot_DIR, 'kmeans_real_feature_space.png'), dpi=200, bbox_inches='tight')
plt.show()
# %%
# %%
# VISUALIZE CLUSTERING SPACE WITH CENTROIDS (CLEANED VERSION)
# =============================================================================
feat_x = 'speed_log_z'
feat_y = 'accel_sd_mag_z'

colors = {'resting': '#888780', 'grazing': '#1D9E75', 'traveling': '#D85A30'}
# Custom text offsets for each behavior to prevent overlaps
text_offsets = {
    'resting': (-40, -25),     # Push down and left
    'grazing': (-45, 15),      # Push up and left
    'traveling': (15, 5)       # Push right
}

fig, ax = plt.subplots(figsize=(10, 8))

# --- Scatter each behavior cluster ---
for behavior, color in colors.items():
    sub = gps_clean[gps_clean['behavior_pred'] == behavior]
    # Increase sample size slightly since we are zooming in
    sub = sub.sample(min(5000, len(sub)), random_state=42)
    ax.scatter(sub[feat_x], sub[feat_y],
               s=8, alpha=0.3, color=color, label=behavior, edgecolor='none')

# --- Overlay K-Means centroids ---
centers = kmeans.cluster_centers_  
for cid, (acc_z, spd_z) in enumerate(centers):
    behavior = label_map[cid] 
    
    # Plot centroid marker
    ax.scatter(spd_z, acc_z,
               s=250, color=colors[behavior],
               edgecolor='black', linewidth=2, zorder=5)
    
    # Annotate with custom offsets to avoid overlap
    ax.annotate(f'{behavior.capitalize()}\nCentroid',
                (spd_z, acc_z), 
                fontsize=10, 
                fontweight='bold',
                xytext=text_offsets[behavior], 
                textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='black', lw=0.8, alpha=0.7))

ax.set_xlabel('z-scored log-speed)', fontsize=11)
ax.set_ylabel('z-scored accel std-dev magnitude)', fontsize=11)
ax.set_title('K-Means Clustering Space', fontsize=13, fontweight='bold', pad=15)

# --- Clean up axes limits to handle the long tail ---
ax.set_xlim(-2.5, 8.0)  # Focuses on the dense cluster space, hiding extreme outliers
ax.set_ylim(-3.0, 10.0) # Focuses on the dense accelerometer space

ax.axhline(0, color='gray', linestyle=':', alpha=0.4)
ax.axvline(0, color='gray', linestyle=':', alpha=0.4)
ax.legend(loc='upper right', framealpha=0.9, fontsize=11)
ax.grid(True, linestyle='--', alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(Plot_DIR, 'kmeans_real_feature_space_clean.png'), dpi=200, bbox_inches='tight')
plt.show()
# %%

# %%
# EXPORT GRAZING AND RESTING BEHAVIOR
# =============================================================================

# Define the target behaviors we want to extract
target_behaviors = ['grazing', 'resting']

# Filter the dataset
gps_behavior_subset = gps_clean[gps_clean['behavior_pred'].isin(target_behaviors)].copy()

print(f"Total rows extracted: {len(gps_behavior_subset):,}")
print(gps_behavior_subset['behavior_pred'].value_counts())

# Define output paths
EXPORT_DIR = "/home/shishir/Documents/Graze_Sat/Data/"
os.makedirs(EXPORT_DIR, exist_ok=True)

csv_out_path = os.path.join(EXPORT_DIR, "cattle_grazing_resting_2025.csv")
gpkg_out_path = os.path.join(EXPORT_DIR, "cattle_behaviors_spatial.gpkg")

# --- Option 1: Export as a standard CSV ---
# (We drop the geometry column or convert it to text so standard pandas can write it easily)
csv_export_df = pd.DataFrame(gps_behavior_subset.drop(columns=['geometry']))
csv_export_df.to_csv(csv_out_path, index=False)
print(f"Successfully exported CSV to: {csv_out_path}")

# --- Option 2: Export as a Spatial GeoPackage ---
# (Perfect for dragging straight into QGIS/ArcGIS)
# Note: datetime columns with timezones can sometimes throw errors in older GIS drivers, 
# so we convert timezones to string format for safe geospatial storage.
gps_spatial_export = gps_behavior_subset.copy()
gps_spatial_export['time'] = gps_spatial_export['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
gps_spatial_export['time_local'] = gps_spatial_export['time_local'].dt.strftime('%Y-%m-%d %H:%M:%S')

gps_spatial_export.to_file(gpkg_out_path, driver="GPKG", layer="grazing_resting_points")
print(f"Successfully exported Geospatial GeoPackage to: {gpkg_out_path}")

# %%
# Cell 16: Diurnal stacked bar chart
if 'hour_local' in gps_clean.columns:
    fig, ax = plt.subplots(figsize=(10,5))
    bottom = np.zeros(24)
    for b in ['resting','grazing','traveling']:
        vals = diurnal[b].reindex(range(24)).fillna(0).values
        ax.bar(range(24), vals, bottom=bottom, label=b, color=colors[b])
        bottom += vals
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Proportion"); ax.legend()
    plt.show()
# %%
