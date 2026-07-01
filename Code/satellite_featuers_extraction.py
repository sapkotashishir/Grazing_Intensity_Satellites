#%%
# ===========================================================================
# IMPORTS & EARTH ENGINE INITIALIZATION
# ===========================================================================
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

from shapely.geometry import box, Point
from sklearn.preprocessing import StandardScaler

import ee
import geemap

ee.Authenticate()
ee.Initialize(project='shishirbeeville')

print("Earth Engine initialized successfully.")

#%%
# ===========================================================================
# PATHS & CONFIG
# ===========================================================================
grazing_resting_csv = '/home/shishir/Documents/Graze_Sat/Data/cattle_grazing_resting_2025.csv'
grid_10 = "/home/shishir/Documents/Graze_Sat/Data/grid_10m/s1_grid.shp"
grid_20 = "/home/shishir/Documents/Graze_Sat/Data/grid_20m/grid_20m.shp"
grid_30 = "/home/shishir/Documents/Graze_Sat/Data/grid_30m/grid_30m.shp"

GRID_SHP = grid_10

OUTPUT_DIR = "/home/shishir/Documents/Graze_Sat/Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#%%
# ===========================================================================
# STEP 1 — LOAD GRAZING GPS AND GRID
# ===========================================================================
grazing = pd.read_csv(grazing_resting_csv)

# Ensure time is parsed and date exists
grazing['time'] = pd.to_datetime(grazing['time'], utc=True, errors='coerce')
grazing['date'] = grazing['time'].dt.floor('D')

cells_gdf = gpd.read_file(GRID_SHP)

# --- Summary Report ---
print("=" * 40)
print("DATA LOADING & CLEANING SUMMARY")
print("=" * 40)
print(f"Clean GPS fixes:     {len(grazing):,}")
print(f"Unique animals:      {grazing['animal_id'].nunique()}")
print(
    f"Date range:          {grazing['date'].min().strftime('%Y-%m-%d')} to {grazing['date'].max().strftime('%Y-%m-%d')}"
)
print("-" * 40)
print(f"Grid cells loaded:   {len(cells_gdf):,}")
print(f"Grid CRS:            {cells_gdf.crs}")
print("=" * 40)

print(grazing.head())

#%%
# ===========================================================================
# STEP 2 — DYNAMIC FIX INTERVALS (HOURS)
# ===========================================================================
gps_foraging = grazing.copy()
gps_sorted = gps_foraging.sort_values(['animal_id', 'time']).copy()

# 1. Time difference to previous fix within animal
gps_sorted['dt'] = gps_sorted.groupby('animal_id')['time'].diff()

# 2. Fill first fix per animal with global median dt
median_dt = gps_sorted['dt'].median()
gps_sorted['dt'] = gps_sorted['dt'].fillna(median_dt)

# 3. Convert to hours
gps_sorted['fix_interval_hr'] = gps_sorted['dt'].dt.total_seconds() / 3600.0

# Cap extreme gaps (e.g., connection loss)
MAX_GAP_MINUTES = 60.0  # 1 hour
gps_sorted['fix_interval_hr'] = gps_sorted['fix_interval_hr'].clip(upper=MAX_GAP_MINUTES / 60.0)

print("Dynamic fix intervals calculated and capped at 1 hour max.")
print(gps_sorted[['animal_id', 'time', 'fix_interval_hr']].head())

#%%
# ===========================================================================
# STEP 3 — SPATIAL JOIN GPS FIXES TO GRID CELLS
# ===========================================================================
gps_gdf = gpd.GeoDataFrame(
    gps_sorted.copy(),
    geometry=gpd.points_from_xy(gps_sorted['lon'], gps_sorted['lat']),
    crs='EPSG:4326'
).to_crs(cells_gdf.crs)

# Remove old spatial-join columns if present
gps_gdf = gps_gdf.drop(columns=['index_left', 'index_right'], errors='ignore')

gps_joined = gpd.sjoin(
    gps_gdf,
    cells_gdf[['cell_id', 'geometry']],
    how='inner',
    predicate='within'
)

gps_joined['cell_id'] = gps_joined['cell_id_right'].astype(int)

print("Mapped fixes:", len(gps_joined))
print(gps_joined[['animal_id', 'time', 'cell_id']].head())

#%%
# ===========================================================================
# STEP 4 — DYNAMIC ANIMAL UNIT CONTRIBUTION
# ===========================================================================
AU_PER_ANIMAL = 1.4  # adjust if different AU per animal

gps_joined['animal_hours'] = gps_joined['fix_interval_hr'] * AU_PER_ANIMAL
gps_joined['au_day_fraction'] = gps_joined['animal_hours'] / 24.0

# Aggregate per animal x date x cell
daily_cell_use = (
    gps_joined
    .groupby(['animal_id', 'date', 'cell_id'], as_index=False)
    .agg(
        n_fixes=('animal_id', 'size'),
        animal_hours=('animal_hours', 'sum'),
        au_day_fraction=('au_day_fraction', 'sum')
    )
)

print("\nDaily cell use calculation complete (sample):")
print(daily_cell_use.head())

#%%
# ===========================================================================
# STEP 4B — VALIDATION: TOTAL TRACKING PER ANIMAL-DAY
# ===========================================================================
check = (
    daily_cell_use
    .groupby(['animal_id', 'date'], as_index=False)
    .agg(
        total_au_day=('au_day_fraction', 'sum'),
        total_hours=('animal_hours', 'sum'),
        n_cells_used=('cell_id', 'nunique')
    )
)

print("\nValidation Check (Total tracking hours per animal-day):")
print(check.head(10))

#%%
# ===========================================================================
# STEP 5 — DEFINE 12-DAY BINS
# ===========================================================================
# Epoch aligned to Sentinel-1 window start
epoch_start = pd.Timestamp('2025-06-17', tz='UTC')

daily_cell_use['date'] = pd.to_datetime(daily_cell_use['date'], utc=True)

# Bin index: 12-day periods since epoch_start
daily_cell_use['bin_idx'] = ((daily_cell_use['date'] - epoch_start).dt.days // 12).astype(int)
daily_cell_use['bin_start'] = epoch_start + pd.to_timedelta(daily_cell_use['bin_idx'] * 12, unit='D')
daily_cell_use['bin_end'] = daily_cell_use['bin_start'] + pd.Timedelta(days=11)

print("\n12-day binning complete (sample):")
print(daily_cell_use[['animal_id', 'date', 'cell_id', 'bin_idx', 'bin_start', 'bin_end']].head())

#%%
# ===========================================================================
# STEP 5B — CELL x 12-DAY BIN GRAZING METRICS (GAPS FILLED WITH 0)
# ===========================================================================
# Aggregate over each cell x 12-day bin
cell_bin_use_gaps = (
    daily_cell_use
    .groupby(['cell_id', 'bin_idx', 'bin_start', 'bin_end'], as_index=False)
    .agg(
        n_animals=('animal_id', 'nunique'),
        animal_days_records=('date', 'count'),
        total_animal_hours=('animal_hours', 'sum'),
        au_days=('au_day_fraction', 'sum')
    )
)

print("\nCell x bin aggregation (without gaps filled):")
print(cell_bin_use_gaps.head())

# --- Build full grid using all cells from S1 grid ---
all_cells = cells_gdf['cell_id'].unique()

min_bin = cell_bin_use_gaps['bin_idx'].min()
max_bin = cell_bin_use_gaps['bin_idx'].max()
all_bins = range(min_bin, max_bin + 1)

full_index = pd.MultiIndex.from_product(
    [all_cells, all_bins],
    names=['cell_id', 'bin_idx']
)

# Reindex to insert rows for (cell, bin) with no grazing data
cell_bin_use = (
    cell_bin_use_gaps
    .set_index(['cell_id', 'bin_idx'])
    .reindex(full_index)
    .reset_index()
)

# Fill metrics with 0 where no GPS grazing occurred
metric_cols = ['n_animals', 'animal_days_records', 'total_animal_hours', 'au_days']
cell_bin_use[metric_cols] = cell_bin_use[metric_cols].fillna(0)

# Reconstruct bin_start/bin_end for all rows based on bin_idx
cell_bin_use['bin_start'] = epoch_start + pd.to_timedelta(cell_bin_use['bin_idx'] * 12, unit='D')
cell_bin_use['bin_end'] = cell_bin_use['bin_start'] + pd.Timedelta(days=11)

# Clean types and sort
int_cols = ['n_animals', 'animal_days_records']
cell_bin_use[int_cols] = cell_bin_use[int_cols].astype(int)

cell_bin_use = cell_bin_use.sort_values(['cell_id', 'bin_idx']).reset_index(drop=True)

print("\nNumber of cells in cell_bin_use:", cell_bin_use['cell_id'].nunique())
print("\nFinal Grazing Metrics per Cell per 12-day bin (gaps filled with 0):")
print(cell_bin_use.head(16))

#%%
# ===========================================================================
# STEP 6 — GEE GEOMETRY & TIME CONFIGURATION
# ===========================================================================
cells_ee = geemap.geopandas_to_ee(cells_gdf[['cell_id', 'geometry']])
paddock_geom = cells_ee.geometry()

# Exact 12-day bin start dates
# NOTE: bin_starts[i] -> bin_starts[i+1] defines bin_idx = i (via enumerate below).
#   bin0 = 06-17 -> 06-29   (baseline only, used to seed *_change features)
#   bin1 = 06-29 -> 07-11   (first bin actually used in modeling)
#   ...
bin_starts = [
    '2025-06-17', '2025-06-29', '2025-07-11', '2025-07-23', '2025-08-04',
    '2025-08-16'
]

bin_start_millis = ee.List([ee.Date(d).millis() for d in bin_starts])

s2_bands = ['NDVI', 'NDRE', 'NDWI', 'LSWI', 'EVI', 'B2', 'B3', 'B4', 'B5', 'B8', 'B8A', 'B11']
s1_bands = ['VV', 'VH', 'VH_VV_ratio']

#%%
# ===========================================================================
# STEP 7 — PREPROCESSING FUNCTIONS (S2 CLOUD MASK / INDICES, S1 SMOOTHING)
# ===========================================================================
def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return ee.Image(image.updateMask(mask).divide(10000.0).copyProperties(image, ['system:time_start']))

def calculate_s2_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndre = image.normalizedDifference(['B8A', 'B5']).rename('NDRE')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    lswi = image.normalizedDifference(['B8', 'B11']).rename('LSWI')
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {'NIR': image.select('B8'), 'RED': image.select('B4'), 'BLUE': image.select('B2')}
    ).rename('EVI')
    return image.addBands([ndvi, ndre, ndwi, lswi, evi])

def process_s1(image):
    radius = 30
    vv_smooth = image.select('VV').focal_mean(radius=radius, units='meters', kernelType='circle').rename('VV')
    vh_smooth = image.select('VH').focal_mean(radius=radius, units='meters', kernelType='circle').rename('VH')
    vh_vv_ratio = vh_smooth.subtract(vv_smooth).rename('VH_VV_ratio')
    return image.addBands([vv_smooth, vh_smooth, vh_vv_ratio], overwrite=True)

#%%
# ===========================================================================
# STEP 8 — RAW ACQUISITION (BROADENED WINDOWS, FOR INTERPOLATION ANCHORS)
# ===========================================================================
s2_coll = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(paddock_geom)
    .filterDate('2025-05-01', '2025-11-30')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
    .map(mask_s2_clouds)
    .map(calculate_s2_indices)
    .select(s2_bands)
)

s1_coll = (
    ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(paddock_geom)
    .filterDate('2025-05-01', '2025-11-30')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .map(process_s1)
    .select(s1_bands)
)

#%%
# ===========================================================================
# STEP 9 — MOSAIC GENERATION & 12-DAY TIME BINNING
# ===========================================================================
def create_initial_mosaics(coll):
    def bin_mapping(start_millis):
        start_date = ee.Date(start_millis)
        end_date = start_date.advance(12, 'day')

        bin_images = coll.filterDate(start_date, end_date)
        has_data = bin_images.size().gt(0)

        composite = ee.Image(ee.Algorithms.If(
            has_data,
            bin_images.mean(),
            ee.Image.constant(0).rename([coll.first().bandNames().get(0)]).select([])
        ))

        return (composite
                .set('system:time_start', start_millis)
                .set('has_data', has_data)
                .set('image_count', bin_images.size()))

    return ee.ImageCollection(bin_start_millis.map(bin_mapping))

s2_binned_raw = create_initial_mosaics(s2_coll)
s1_binned_raw = create_initial_mosaics(s1_coll)

#%%
# ===========================================================================
# STEP 10 — TEMPORAL INTERPOLATION ENGINE (FILLS BINS WITH NO IMAGERY)
# ===========================================================================
def interpolate_collection(binned_coll, raw_coll, all_bands):
    """Interpolates using un-binned historical images as anchor points."""

    # Inject pixel-level time bands safely into raw collections
    raw_with_time = raw_coll.map(lambda img: img.addBands(
        ee.Image.constant(img.get('system:time_start')).rename('time')
    ))

    def interpolate_bin(start_millis):
        target_time = ee.Number(start_millis)
        target_image = binned_coll.filter(ee.Filter.eq('system:time_start', target_time)).first()
        has_data = target_image.get('has_data')
        orig_count = target_image.get('image_count')

        # Closest historical and future anchor images from raw timelines
        img_before = ee.Image(raw_with_time
                              .filter(ee.Filter.lt('system:time_start', target_time))
                              .sort('system:time_start', False)
                              .first())

        img_after = ee.Image(raw_with_time
                             .filter(ee.Filter.gt('system:time_start', target_time))
                             .sort('system:time_start', True)
                             .first())

        t1 = img_before.select('time')
        t2 = img_after.select('time')
        t_target = ee.Image.constant(target_time)

        # Linear interpolation
        time_ratio = t_target.subtract(t1).divide(t2.subtract(t1))
        interpolated_image = img_before.add(img_after.subtract(img_before).multiply(time_ratio))

        final_image = ee.Image(ee.Algorithms.If(
            has_data,
            target_image,
            interpolated_image
        )).select(all_bands)

        is_interpolated = ee.Number(ee.Algorithms.If(has_data, 0, 1))

        return (final_image
                .set('system:time_start', target_time)
                .set('interpolated', is_interpolated)
                .set('orig_count', orig_count))

    return ee.ImageCollection(bin_start_millis.map(interpolate_bin))

s2_interpolated = interpolate_collection(s2_binned_raw, s2_coll, s2_bands)
s1_interpolated = interpolate_collection(s1_binned_raw, s1_coll, s1_bands)

#%%
# ===========================================================================
# STEP 11 — FEATURE EXTRACTION (PER CELL x BIN)
# ===========================================================================
all_records = []
print('Extracting features using deep-search temporal interpolation...')

for i, start_str in enumerate(bin_starts[:-1]):
    start_millis = ee.Date(start_str).millis()

    s2_img = s2_interpolated.filter(ee.Filter.eq('system:time_start', start_millis)).first()
    s1_img = s1_interpolated.filter(ee.Filter.eq('system:time_start', start_millis)).first()

    s2_is_interp = s2_img.get('interpolated').getInfo()
    s1_is_interp = s1_img.get('interpolated').getInfo()
    s2_orig_count = s2_img.get('orig_count').getInfo()
    s1_orig_count = s1_img.get('orig_count').getInfo()

    combined_img = ee.Image.cat([
        s2_img.select(s2_bands),
        s1_img.select(s1_bands)
    ])

    sampled_features = combined_img.reduceRegions(
        collection=cells_ee,
        reducer=ee.Reducer.mean(),
        scale=10
    ).getInfo()

    for feat in sampled_features['features']:
        props = feat['properties']
        props['cell_id'] = props.get('cell_id', feat.get('id'))
        props['bin_idx'] = i
        props['bin_start'] = start_str
        props['bin_end'] = bin_starts[i + 1]
        props['s2_count'] = s2_orig_count
        props['s1_count'] = s1_orig_count
        props['s2_interpolated'] = s2_is_interp
        props['s1_interpolated'] = s1_is_interp
        all_records.append(props)

satellite_df = pd.DataFrame(all_records)
print("\nExtraction complete.")
print(satellite_df.head())

#%%
# ===========================================================================
# STEP 12 — MERGE GRAZING + SATELLITE FEATURES
# ===========================================================================
analysis_df = pd.merge(
    cell_bin_use,
    satellite_df,
    on=['cell_id', 'bin_idx'],
    how='left'
)

print("Columns after merge:")
print(analysis_df.columns.tolist())

#%%
# ===========================================================================
# STEP 13 — RESOLVE DUPLICATE TEMPORAL COLUMNS (FROM MERGE SUFFIXES)
# ===========================================================================
for col in ['bin_start', 'bin_end']:
    x = f'{col}_x'
    y = f'{col}_y'

    if x in analysis_df.columns:
        analysis_df[col] = analysis_df[x]
    elif y in analysis_df.columns:
        analysis_df[col] = analysis_df[y]

dup_cols = [c for c in analysis_df.columns if c.endswith('_x') or c.endswith('_y')]
analysis_df = analysis_df.drop(columns=dup_cols, errors='ignore')

#%%
# ===========================================================================
# STEP 14 — IDENTIFY SATELLITE FEATURE COLUMNS
# ===========================================================================
exclude_cols = {
    'cell_id',
    'bin_idx',
    'bin_start',
    'bin_end',
    'au_days',
    'n_animals',
    'animal_days_records',
    'total_animal_hours',
    'total_forage_removal',
    's1_count',
    's2_count',
    's1_interpolated',
    's2_interpolated'
}

numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()

feature_cols = [
    c for c in numeric_cols
    if c not in exclude_cols
    and not c.endswith('_change')
]

print("\nSatellite features:")
print(feature_cols)

#%%
# ===========================================================================
# STEP 15 — KEEP REQUIRED COLUMNS ONLY
# ===========================================================================
base_cols = ['cell_id', 'bin_idx', 'bin_start', 'au_days']

if 'bin_end' in analysis_df.columns:
    base_cols.append('bin_end')

keep_cols = [c for c in base_cols + feature_cols if c in analysis_df.columns]
analysis_df = analysis_df[keep_cols]

#%%
# ===========================================================================
# STEP 16 — COMPUTE FORAGE REMOVAL (DERIVED FROM au_days — DO NOT USE AS A
# MODEL FEATURE, IT IS DETERMINISTIC FROM THE TARGET AND WILL LEAK)
# ===========================================================================
FORAGE_LBS_PER_AU_DAY = 26.0

analysis_df['total_forage_removal'] = analysis_df['au_days'] * FORAGE_LBS_PER_AU_DAY

#%%
# ===========================================================================
# STEP 17 — SORT TEMPORALLY
# ===========================================================================
analysis_df = analysis_df.sort_values(['cell_id', 'bin_idx']).reset_index(drop=True)

#%%
# ===========================================================================
# STEP 18 — FILL MISSING SATELLITE VALUES (WITHIN-CELL bfill/ffill)
# ===========================================================================
analysis_df[feature_cols] = (
    analysis_df
    .groupby('cell_id')[feature_cols]
    .transform(lambda x: x.bfill().ffill())
)

#%%
# ===========================================================================
# STEP 19 — CROSS-BIN CHANGE FEATURES (bin_n - bin_{n-1}, PER CELL)
# ===========================================================================
for col in feature_cols:
    analysis_df[f'{col}_change'] = (
        analysis_df
        .groupby('cell_id')[col]
        .transform(lambda x: x - x.shift(1))
    )

#%%
# ===========================================================================
# STEP 20 — DROP BASELINE BIN0 (NO PRIOR BIN -> _change WOULD BE NaN)
# Final dataset starts at bin1, with bin0 having served its purpose as the
# seed value for the *_change features above.
# ===========================================================================
analysis_df = analysis_df.query("bin_idx >= 1").reset_index(drop=True)

#%%
# ===========================================================================
# STEP 21 — FINAL COLUMN ORDERING
# ===========================================================================
change_cols = [f'{c}_change' for c in feature_cols]

final_cols = (
    ['cell_id', 'bin_idx', 'bin_start']
    + (['bin_end'] if 'bin_end' in analysis_df.columns else [])
    + ['au_days', 'total_forage_removal']
    + feature_cols
    + change_cols
)

analysis_df = analysis_df[final_cols]

print("\nFinal dataframe shape:")
print(analysis_df.shape)
print("\nPreview:")
print(analysis_df.head())

#%%
# ===========================================================================
# STEP 22 — QUICK CHECKS
# ===========================================================================
print(analysis_df.columns.to_list())

#%%
print(
    analysis_df[
        ['cell_id', 'bin_idx', 'bin_start', 'bin_end', 'au_days', 'total_forage_removal']
    ].head(15)
)
#%%
# export to CSV
output_csv = os.path.join(OUTPUT_DIR, 'grazing_satellite_features_grid10m.csv')
analysis_df.to_csv(output_csv, index=False)
print(f"\nFinal dataset exported to: {output_csv}")
# %%
