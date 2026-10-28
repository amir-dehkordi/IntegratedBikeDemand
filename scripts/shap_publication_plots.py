"""
shap_publication_plots.py
-------------------------
Publication SHAP figures for the XGBoost demand model.

Model specification (this version)
  - The separate AM-peak / PM-peak indicators are replaced by ONE indicator,
    is_commuting_time = 1 for weekday trips in 07:00-10:00 or 16:00-20:00.
  - Panel rows are station x day x time block x trip stage. Two time blocks:
      Commute      weekday 07:00-10:00 and 16:00-20:00   (7 h)
      Non_Commute  all other weekday hours (17 h); on weekends the whole day (24 h)
  - Blocks differ in length, so by default (USE_HOURLY_RATE = True) the model predicts trips
    PER HOUR and each row is weighted by its length in hours; R2 / RMSE / MAE are computed on
    trip counts (predicted rate x block hours). SHAP values are then in trips-per-hour units.

Evaluation design (same as model_comparison_tuned_temporal.py)
  - Temporal split: the LAST HOLDOUT_DAYS (14) days are withheld as the test set.
  - XGBoost hyperparameters were tuned by random search (30 candidates) with
    expanding-window TimeSeriesSplit cross-validation on data preceding the
    evaluation period (see XGB_PARAMS below).
  - The model is fitted on the training period only. SHAP values are computed with
    that model for the rows selected by SHAP_ON.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

sys.path.append(str(Path(__file__).resolve().parents[1]))
import paths

# =====================================================================
# SETTINGS
# =====================================================================
HOLDOUT_DAYS = 14      # last N days are the temporal test set
SHAP_ON = "all"        # rows explained by the fitted model: "all" | "train" | "test"
SEED = 42

# Commuting hours (weekdays only). End hours are exclusive.
AM_PEAK = (7, 10)
PM_PEAK = (16, 20)
COMMUTE_HOURS = (AM_PEAK[1] - AM_PEAK[0]) + (PM_PEAK[1] - PM_PEAK[0])

# True: model trips per hour and weight rows by block length (recommended).
# False: model raw trip counts per block.
USE_HOURLY_RATE = True

# NOTE: these values were tuned for the OLD target (trip counts, separate AM/PM indicators).
# Re-run the comparison script with the new specification and update them if they change
# (and the dictionary in window_shap_sensitivity.py / model_5.py).
XGB_PARAMS = dict(
    n_estimators=300, max_depth=8, learning_rate=0.03, subsample=0.7,
    colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=10.0, n_jobs=-1,
)

# =====================================================================
# 1. BUILD THE STATION x DAY x PERIOD x STAGE PANEL (same as the comparison script)
# =====================================================================
print('Building dataset (same pipeline as model_comparison_tuned_temporal.py)')
trips_df = pd.read_csv(paths.TRIPS_CSV, dtype={'start_station_id': str, 'end_station_id': str})
sld_df = pd.read_csv(paths.SLD_STATION_CSV)
landuse_df = pd.read_csv(paths.LANDUSE_500M_CSV)
weather_df = pd.read_csv(paths.WEATHER_CSV)
station_acs_df = pd.read_csv(paths.ACS_STATION_CSV)
station_osm_df = pd.read_csv(paths.OSM_STATION_CSV)

trips_df['started_at'] = pd.to_datetime(trips_df['started_at'], errors='coerce')
trips_df['date'] = trips_df['started_at'].dt.date
trips_df['hour'] = trips_df['started_at'].dt.hour
weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date


def is_commute_hour(hour):
    return (AM_PEAK[0] <= hour < AM_PEAK[1]) or (PM_PEAK[0] <= hour < PM_PEAK[1])


trips_df = trips_df.dropna(subset=['started_at']).copy()
_weekend = trips_df['started_at'].dt.weekday >= 5
_commuting = (~_weekend) & trips_df['hour'].apply(is_commute_hour)
trips_df['time_block'] = np.where(_commuting, 'Commute', 'Non_Commute')
peak_trips = trips_df.copy()
peak_trips['end_station_id'] = pd.to_numeric(peak_trips['end_station_id'], errors='coerce')
peak_trips['start_station_id'] = pd.to_numeric(peak_trips['start_station_id'], errors='coerce')
peak_trips['anchor_station_id'] = np.where(
    peak_trips['trip_type'] == 'first_mile',
    peak_trips['end_station_id'],
    peak_trips['start_station_id'],
)
peak_trips = peak_trips.dropna(subset=['anchor_station_id', 'date'])

PANEL_KEYS = ['date', 'anchor_station_id', 'time_block', 'trip_type']
observed_demand = peak_trips.groupby(PANEL_KEYS).size().reset_index(name='trip_demand')

all_dates = pd.date_range(start=observed_demand['date'].min(), end=observed_demand['date'].max()).date
all_stations = observed_demand['anchor_station_id'].unique()
grid_index = pd.MultiIndex.from_product(
    [all_dates, all_stations, ['Commute', 'Non_Commute'], ['first_mile', 'last_mile']],
    names=PANEL_KEYS,
)
full_panel_grid = pd.DataFrame(index=grid_index).reset_index()
# is_weekend = 1 on Saturday/Sunday (the label matches the coding)
full_panel_grid['is_weekend'] = full_panel_grid['date'].apply(lambda x: 1 if x.weekday() >= 5 else 0)
# no commuting block on weekends
full_panel_grid = full_panel_grid[
    ~((full_panel_grid['is_weekend'] == 1) & (full_panel_grid['time_block'] == 'Commute'))
]
panel_data = pd.merge(full_panel_grid, observed_demand, on=PANEL_KEYS, how='left').fillna({'trip_demand': 0})

final_modeled_df = panel_data.copy()
final_modeled_df['is_first_mile'] = (final_modeled_df['trip_type'] == 'first_mile').astype(int)
final_modeled_df['is_commuting_time'] = (final_modeled_df['time_block'] == 'Commute').astype(int)
final_modeled_df['period_hours'] = np.where(
    final_modeled_df['is_commuting_time'] == 1, COMMUTE_HOURS,
    np.where(final_modeled_df['is_weekend'] == 1, 24, 24 - COMMUTE_HOURS),
).astype(float)
final_modeled_df['trip_rate'] = final_modeled_df['trip_demand'] / final_modeled_df['period_hours']

# Structural merge
sld_df['Bike Station ID'] = pd.to_numeric(sld_df['Bike Station ID'], errors='coerce').round(2)
landuse_df['Bike Station ID'] = pd.to_numeric(landuse_df['Bike Station ID'], errors='coerce').round(2)
station_acs_df['Bike Station ID'] = pd.to_numeric(station_acs_df['Bike Station ID'], errors='coerce').round(2)
station_osm_df['Bike Station ID'] = pd.to_numeric(station_osm_df['Bike Station ID'], errors='coerce').round(2)
final_modeled_df['anchor_station_id'] = pd.to_numeric(final_modeled_df['anchor_station_id'], errors='coerce').round(2)

if 'jobs_count' in station_acs_df.columns and 'job_density' not in station_acs_df.columns:
    station_acs_df = station_acs_df.rename(columns={'jobs_count': 'job_density'})
for col in ['job_density', 'age24_30_pct']:
    if col not in station_acs_df.columns:
        station_acs_df[col] = 0.0

sld_clean = sld_df[['Bike Station ID', 'IntDensity', 'TransitAccess']].copy()
spatial_base = pd.merge(
    sld_clean,
    landuse_df[['Bike Station ID', 'residential_pct', 'commercial_pct', 'industrial_pct', 'institutional_pct']],
    on='Bike Station ID', how='inner',
)
spatial_complete = pd.merge(spatial_base, station_acs_df, on='Bike Station ID', how='inner')
spatial_complete = pd.merge(
    spatial_complete, station_osm_df[['Bike Station ID', 'road_length_m']],
    on='Bike Station ID', how='inner',
)
mid_dataset = pd.merge(
    final_modeled_df, spatial_complete,
    left_on='anchor_station_id', right_on='Bike Station ID', how='inner',
)
final_dataset = pd.merge(mid_dataset, weather_df, on='date', how='inner').reset_index(drop=True)

feature_cols = [
    'is_first_mile', 'is_commuting_time', 'is_weekend', 'temp_max_c',
    'precipitation_mm', 'max_wind_speed_kmh', 'mean_humidity_pct', 'dock_capacity',
    'ferry_distance_m', 'landuse_entropy', 'job_density', 'IntDensity',
    'TransitAccess', 'road_length_m', 'residential_pct', 'commercial_pct',
    'industrial_pct', 'institutional_pct', 'population_density', 'zero_vehicle_pct',
    'age18_24_pct', 'age24_30_pct', 'higher_edu_pct', 'median_income',
]

print(f"Panel: {len(final_dataset):,} rows | {final_dataset['anchor_station_id'].nunique()} stations | "
      f"{final_dataset['date'].nunique()} days | total demand {int(final_dataset['trip_demand'].sum()):,}")
print(f"Target: {'trips per hour, rows weighted by block hours' if USE_HOURLY_RATE else 'trip counts per block'}")

# Diagnostic: block length vs. trips (this is why the target is a rate)
print("\nDemand by time block (hours = block length):")
print(final_dataset.groupby(['time_block', 'is_weekend'])
      .agg(rows=('trip_demand', 'size'), hours=('period_hours', 'first'),
           mean_trips_per_row=('trip_demand', 'mean'), mean_trips_per_hour=('trip_rate', 'mean'))
      .round(3).to_string())
print()

# 4 FEATURE GROUPS FRAMEWORK
FEATURE_GROUPS = {
    'Station & trip characteristics': [
        'dock_capacity', 'ferry_distance_m', 'is_first_mile',
    ],
    'Built environment': [
        'residential_pct', 'commercial_pct', 'industrial_pct', 'institutional_pct',
        'landuse_entropy', 'TransitAccess', 'road_length_m', 'IntDensity',
    ],
    'Socio-demographic': [
        'job_density', 'population_density', 'median_income', 'higher_edu_pct',
        'zero_vehicle_pct', 'age18_24_pct', 'age24_30_pct',
    ],
    'Temporal & weather': [
        'is_commuting_time', 'is_weekend', 'precipitation_mm', 'temp_max_c',
        'mean_humidity_pct', 'max_wind_speed_kmh',
    ],
}

GROUP_COLORS = {
    'Station & trip characteristics': '#55a868',
    'Built environment': '#c44e52',
    'Socio-demographic': '#8172b3',
    'Temporal & weather': '#ccb974',
}


def feature_group(feature_name):
    for group, members in FEATURE_GROUPS.items():
        if feature_name in members:
            return group
    return 'Other'


# =====================================================================
# 2. TEMPORAL SPLIT, FIT, HOLDOUT CHECK
# =====================================================================
X = final_dataset[feature_cols]
y = final_dataset['trip_demand']                      # trip counts (used for the metrics)
hours = final_dataset['period_hours'].to_numpy(dtype=float)
y_fit = (y.to_numpy(dtype=float) / hours) if USE_HOURLY_RATE else y.to_numpy(dtype=float)
w_fit = hours if USE_HOURLY_RATE else None

unique_dates = np.sort(final_dataset['date'].unique())
if len(unique_dates) <= HOLDOUT_DAYS:
    raise ValueError(f"Only {len(unique_dates)} days available; need more than HOLDOUT_DAYS={HOLDOUT_DAYS}.")
cutoff_date = unique_dates[-HOLDOUT_DAYS]
train_mask = (final_dataset['date'] < cutoff_date).to_numpy()
test_mask = ~train_mask

X_train, y_train = X.loc[train_mask], y.loc[train_mask]
X_test, y_test = X.loc[test_mask], y.loc[test_mask]
print(f"Training period: {unique_dates[0]} to {unique_dates[-HOLDOUT_DAYS - 1]} ({len(X_train):,} rows)")
print(f"Test period (last {HOLDOUT_DAYS} days): {cutoff_date} to {unique_dates[-1]} ({len(X_test):,} rows)")

model = xgb.XGBRegressor(**XGB_PARAMS, random_state=SEED)
model.fit(X_train, y_fit[train_mask], sample_weight=None if w_fit is None else w_fit[train_mask])

y_pred_check = model.predict(X_test)
if USE_HOURLY_RATE:
    y_pred_check = y_pred_check * hours[test_mask]       # back to trips per block
print(f"\n--- TEMPORAL HOLDOUT: R2 = {r2_score(y_test, y_pred_check):.4f} | "
      f"RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_check)):.2f} | "
      f"MAE = {mean_absolute_error(y_test, y_pred_check):.2f} ---\n")

# =====================================================================
# 3. SHAP VALUES
# =====================================================================
X_shap = {'all': X, 'train': X_train, 'test': X_test}[SHAP_ON]
print(f'Calculating SHAP values (model fitted on the training period; rows explained: {SHAP_ON}, n={len(X_shap):,})...')
explainer = shap.TreeExplainer(model)
shap_arr = np.asarray(explainer.shap_values(X_shap))
if shap_arr.ndim == 3:
    shap_arr = shap_arr[:, :, 0]
abs_shap = np.abs(shap_arr)

mean_importance = abs_shap.mean(axis=0)

order = np.argsort(mean_importance)[::-1]
features_ordered = [feature_cols[i] for i in order]
mean_ordered = mean_importance[order]

os.makedirs(paths.SHAP_PLOTS_DIR, exist_ok=True)

sns.set_style('ticks')
sns.set_context('paper', font_scale=1.1, rc={'font.family': 'serif', 'font.serif': ['Times New Roman']})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.default'] = 'regular'
paper_dpi = 600

DISPLAY_NAME_OVERRIDES = {
    'job_density': 'Employment Density',
    'dock_capacity': 'Dock Capacity',
    'TransitAccess': 'Transit Access',
    'IntDensity': 'Intersection Density',
    'is_commuting_time': 'Commuting Time',
    'is_first_mile': 'Trip Stage',
    'age18_24_pct': 'Age 18–24 (%)',
    'age24_30_pct': 'Age 25–30 (%)',
    'ferry_distance_m': 'Ferry Distance (m)',
    'road_length_m': 'Road Length (m)',
    'is_weekend': 'Weekend',
}


def clean_name(f):
    if f in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[f]

    if f.lower().endswith('_c'):
        base = f[:-2]
        return base.replace('_', ' ').title() + ' (°C)'

    return (f.replace('_', ' ')
             .title()
             .replace('Pct', '(%)')
             .replace('Kmh', '(km/h)')
             .replace(' Mm', ' (mm)'))


clean_feature_names = [clean_name(f) for f in features_ordered]   # importance order (bar chart)
clean_names_in_column_order = [clean_name(f) for f in feature_cols]  # column order (beeswarm)

# Table of global importance for the Results section
pd.DataFrame({
    'feature': features_ordered,
    'label': clean_feature_names,
    'group': [feature_group(f) for f in features_ordered],
    'mean_abs_shap': mean_ordered,
    'share_pct': 100.0 * mean_ordered / mean_ordered.sum(),
}).to_csv(f'{paths.SHAP_PLOTS_DIR}/global_importance_table.csv', index=False)

# =======================================================
# 1. CLEAN GLOBAL SHAP IMPORTANCE BAR CHART
# =======================================================
plt.figure(figsize=(9, 7))

bar_colors = [GROUP_COLORS.get(feature_group(f), 'steelblue') for f in features_ordered]
bars = plt.barh(range(len(features_ordered)), mean_ordered,
                color=bar_colors, edgecolor='black', alpha=0.85, height=0.6)

for bar in bars:
    width = bar.get_width()
    plt.text(width + (max(mean_ordered) * 0.02), bar.get_y() + bar.get_height() / 2,
             f'{width:.2f}',
             va='center', ha='left', fontsize=12, family='serif')

plt.yticks(range(len(features_ordered)), clean_feature_names, fontsize=12, family='serif')
plt.xlabel('Global SHAP value', fontsize=12, family='serif')

sns.despine(left=True, bottom=False)
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(f'{paths.SHAP_PLOTS_DIR}/global_importance.png', dpi=600)
plt.savefig(f'{paths.SHAP_PLOTS_DIR}/global_importance.svg')
plt.close()

# =======================================================
# 1b. RELATIVE IMPORTANCE OF VARIABLE GROUPS (4 GROUPS)
# =======================================================
group_importance = {}
for group, members in FEATURE_GROUPS.items():
    idx = [feature_cols.index(f) for f in members if f in feature_cols]
    group_importance[group] = abs_shap[:, idx].mean(axis=0).sum()

group_series = pd.Series(group_importance).sort_values(ascending=True)
group_pct = 100.0 * group_series / group_series.sum()
group_pct.sort_values(ascending=False).round(2).to_csv(f'{paths.SHAP_PLOTS_DIR}/group_importance_table.csv', header=['share_pct'])

plt.figure(figsize=(8, 4.5))
group_bars = plt.barh(
    range(len(group_pct)),
    group_pct.values,
    color=[GROUP_COLORS[g] for g in group_pct.index],
    edgecolor='black',
    alpha=0.85,
    height=0.65,
)
for bar, pct, raw in zip(group_bars, group_pct.values, group_series.loc[group_pct.index].values):
    plt.text(
        bar.get_width() + 0.4,
        bar.get_y() + bar.get_height() / 2,
        f'{pct:.1f}%  ({raw:.2f})',
        va='center',
        ha='left',
        fontsize=10,
        family='serif',
    )

plt.yticks(range(len(group_pct)), group_pct.index, fontsize=12, family='serif')
plt.xlabel('Relative importance (%)', fontsize=12, family='serif')
sns.despine(left=True, bottom=False)
plt.xlim(0, max(group_pct.values) * 1.25)
plt.tight_layout()
plt.savefig(f'{paths.SHAP_PLOTS_DIR}/group_importance.png', dpi=600)
plt.savefig(f'{paths.SHAP_PLOTS_DIR}/group_importance.svg')
plt.close()

# =======================================================
# 2. BEESWARM SUMMARY
# =======================================================
# feature_names must follow the COLUMN order of shap_arr / X_shap (not the importance order);
# shap.summary_plot sorts the features by importance itself.
sns.set_style('whitegrid')
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_arr, X_shap, feature_names=clean_names_in_column_order, show=False)
plt.tight_layout()
plt.savefig(f'{paths.SHAP_PLOTS_DIR}/beeswarm_summary.png', dpi=paper_dpi)
plt.savefig(f'{paths.SHAP_PLOTS_DIR}/beeswarm_summary.svg')
plt.close()

# =======================================================
# 3. CLEANED DEPENDENCE PLOTS (POLYNOMIAL RIDGE SMOOTHING)
# =======================================================
for feat in feature_cols:
    idx = feature_cols.index(feat)
    xv = X_shap[feat].to_numpy()
    yv = shap_arr[:, idx]

    # 3rd-degree polynomial ridge regression for a smooth trend line
    model_smooth = make_pipeline(PolynomialFeatures(3), Ridge(alpha=1.0))
    model_smooth.fit(xv.reshape(-1, 1), yv)
    xs_smooth = np.linspace(np.nanpercentile(xv, 1), np.nanpercentile(xv, 99), 200)
    ys_smooth = model_smooth.predict(xs_smooth.reshape(-1, 1))

    plt.figure(figsize=(8, 6))
    plt.scatter(xv, yv, color='steelblue', alpha=0.55, s=35, edgecolor='none')
    plt.plot(xs_smooth, ys_smooth, color='black', linewidth=2.25, label='Smoothed trend')

    plt.xlabel(clean_name(feat), fontsize=15, family='Times New Roman', labelpad=15)
    plt.ylabel('SHAP value', fontsize=15, family='Times New Roman')

    plt.legend(frameon=False, fontsize=12, loc='upper left')

    plt.gca().set_title('')
    plt.tight_layout()
    plt.savefig(f'{paths.SHAP_PLOTS_DIR}/dependence_{feat}.png', dpi=paper_dpi)
    plt.savefig(f'{paths.SHAP_PLOTS_DIR}/dependence_{feat}.svg')
    plt.close()

print(f'Saved publication plots and tables in {paths.SHAP_PLOTS_DIR}/')
