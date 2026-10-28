"""
model_comparison_tuned_temporal.py
----------------------------------
Temporal benchmark of demand models for the bike-ferry station-period panel.
Evaluates models strictly on the final 2 weeks of summer 2024.
Models compared: OLS, Poisson GLM, LightGBM and XGBoost.
"""

import sys
import warnings
from itertools import combinations
from pathlib import Path

import pandas as pd
import numpy as np
import shap
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, mean_poisson_deviance,
)
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

sys.path.append(str(Path(__file__).resolve().parents[1]))
import paths

# ---------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------
HOLDOUT_DAYS = 14    # length of final test block in days (last 2 weeks)
N_ITER = 30          # random-search budget, identical for every tuned model
INNER_SPLITS = 3     # TimeSeriesSplit folds used for tuning
SHAP_SAMPLE = 1500   # max rows explained per model for SHAP
TOP_K = 5            # top-k overlap for SHAP ranking agreement
SEED = 42

# =====================================================================
# 1. LOAD DATA & BUILD MATRIX
# =====================================================================
print("Loading datasets and building spatial-temporal matrix...")
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

def assign_peak_period(hour):
    if pd.isna(hour): return "Off_Peak"
    if 7 <= hour < 10: return "AM_Peak"
    if 16 <= hour < 19: return "PM_Peak"
    return "Off_Peak"

trips_df['peak_period'] = trips_df['hour'].apply(assign_peak_period)
peak_trips = trips_df.copy()
peak_trips['end_station_id'] = pd.to_numeric(peak_trips['end_station_id'], errors='coerce')
peak_trips['start_station_id'] = pd.to_numeric(peak_trips['start_station_id'], errors='coerce')
peak_trips['anchor_station_id'] = np.where(
    peak_trips['trip_type'] == 'first_mile',
    peak_trips['end_station_id'],
    peak_trips['start_station_id'],
)
peak_trips = peak_trips.dropna(subset=['anchor_station_id', 'date'])

observed_demand = (
    peak_trips.groupby(['date', 'anchor_station_id', 'peak_period', 'trip_type'])
    .size()
    .reset_index(name='trip_demand')
)

all_dates = pd.date_range(start=observed_demand['date'].min(), end=observed_demand['date'].max()).date
all_stations = observed_demand['anchor_station_id'].unique()
grid_index = pd.MultiIndex.from_product(
    [all_dates, all_stations, ['AM_Peak', 'PM_Peak', 'Off_Peak'], ['first_mile', 'last_mile']],
    names=['date', 'anchor_station_id', 'peak_period', 'trip_type'],
)
full_panel_grid = pd.DataFrame(index=grid_index).reset_index()
panel_data = pd.merge(full_panel_grid, observed_demand, on=['date', 'anchor_station_id', 'peak_period', 'trip_type'], how='left').fillna({'trip_demand': 0})

final_modeled_df = panel_data.copy()
final_modeled_df['is_weekend'] = final_modeled_df['date'].apply(lambda x: 1 if x.weekday() >= 5 else 0)
final_modeled_df['is_first_mile'] = (final_modeled_df['trip_type'] == 'first_mile').astype(int)
final_modeled_df['is_AM_peak'] = (final_modeled_df['peak_period'] == 'AM_Peak').astype(int)
final_modeled_df['is_PM_peak'] = (final_modeled_df['peak_period'] == 'PM_Peak').astype(int)

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
spatial_complete = pd.merge(spatial_complete, station_osm_df[['Bike Station ID', 'road_length_m']], on='Bike Station ID', how='inner')
mid_dataset = pd.merge(final_modeled_df, spatial_complete, left_on='anchor_station_id', right_on='Bike Station ID', how='inner')
final_dataset = pd.merge(mid_dataset, weather_df, on='date', how='inner').sort_values('date').reset_index(drop=True)

feature_cols = [
    'is_first_mile', 'is_AM_peak', 'is_PM_peak',
    'is_weekend', 'temp_max_c', 'precipitation_mm',
    'max_wind_speed_kmh', 'mean_humidity_pct', 'dock_capacity', 'ferry_distance_m', 'landuse_entropy',
    'job_density', 'IntDensity', 'TransitAccess', 'road_length_m',
    'residential_pct', 'commercial_pct', 'industrial_pct', 'institutional_pct',
    'population_density', 'zero_vehicle_pct', 'age18_24_pct', 'age24_30_pct', 'higher_edu_pct', 'median_income',
]

# =====================================================================
# 2. SINGLE TEMPORAL SPLIT (LAST 2 WEEKS HOLDOUT)
# =====================================================================
unique_dates = np.sort(final_dataset['date'].unique())
cutoff_date = unique_dates[-HOLDOUT_DAYS]

train_mask = final_dataset['date'] < cutoff_date
test_mask = final_dataset['date'] >= cutoff_date

X_tune = final_dataset.loc[train_mask, feature_cols]
y_tune = final_dataset.loc[train_mask, 'trip_demand']

X_tr = X_tune
y_tr = y_tune
X_te = final_dataset.loc[test_mask, feature_cols]
y_te = final_dataset.loc[test_mask, 'trip_demand']

print("\nData Split Summary:")
print(f" - Training set: {unique_dates[0]} to {unique_dates[-HOLDOUT_DAYS-1]} ({len(X_tr):,} rows)")
print(f" - Test set (Last 2 Weeks): {cutoff_date} to {unique_dates[-1]} ({len(X_te):,} rows)")

tscv = TimeSeriesSplit(n_splits=INNER_SPLITS)

# =====================================================================
# 3. MODELS, SEARCH SPACES & EQUAL-BUDGET TUNING
# (OLS is untuned; Poisson GLM, LightGBM and XGBoost are tuned)
# =====================================================================
tuning_configs = {
    'Poisson GLM': {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('glm', PoissonRegressor(max_iter=2000)),
        ]),
        'params': {'glm__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1.0]},
    },
    'LightGBM': {
        'model': lgb.LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1),
        'params': {
            'n_estimators': [300, 500, 700],
            'max_depth': [4, 6, 8],
            'num_leaves': [15, 31, 63],
            'learning_rate': [0.03, 0.05, 0.1],
            'subsample': [0.7, 0.85, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0.1, 0.5, 1.0],
            'reg_lambda': [1.0, 5.0, 10.0],
        },
    },
    'XGBoost': {
        'model': xgb.XGBRegressor(random_state=SEED, n_jobs=-1),
        'params': {
            'n_estimators': [300, 500, 700],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.03, 0.05, 0.1],
            'subsample': [0.7, 0.85, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0.1, 0.5, 1.0],
            'reg_lambda': [1.0, 5.0, 10.0],
        },
    },
}

def grid_size(params):
    n = 1
    for v in params.values():
        n *= len(v)
    return n

print("\n" + "=" * 75)
print(f" TUNING (random search, budget = {N_ITER} candidates per model, TimeSeriesSplit CV) ")
print("=" * 75)

tuned_models = {}
inner_cv_rmse = {}
best_params_dict = {}

for name, cfg in tuning_configs.items():
    n_iter = min(N_ITER, grid_size(cfg['params']))
    print(f"\nTuning {name} ({n_iter} candidates)...")
    search = RandomizedSearchCV(
        estimator=cfg['model'],
        param_distributions=cfg['params'],
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_mean_squared_error',
        random_state=SEED,
        n_jobs=-1,
    )
    search.fit(X_tune, y_tune)
    tuned_models[name] = search.best_estimator_
    inner_cv_rmse[name] = float(np.sqrt(-search.best_score_))
    best_params_dict[name] = search.best_params_
    print(f"Optimal hyperparameters for {name}:")
    for param, val in search.best_params_.items():
        print(f"  - {param}: {val}")

# Untuned linear reference
ols = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])

models = {
    'OLS': ols,
    'Poisson GLM': tuned_models['Poisson GLM'],
    'LightGBM': tuned_models['LightGBM'],
    'XGBoost': tuned_models['XGBoost'],
}

# =====================================================================
# 4. METRICS & EVALUATION
# =====================================================================
def evaluate(y_true, pred):
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    nz = y_true > 0
    return {
        'R2': r2_score(y_true, pred),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, pred))),
        'MAE': mean_absolute_error(y_true, pred),
        'Poisson deviance': mean_poisson_deviance(y_true, np.clip(pred, 1e-6, None)),
        'RMSE (non-zero rows)': float(np.sqrt(mean_squared_error(y_true[nz], pred[nz]))) if nz.any() else np.nan,
        'Negative pred. share': float((pred < 0).mean()),
    }

# =====================================================================
# 5. SINGLE HOLDOUT EVALUATION
# =====================================================================
print("\n" + "=" * 75)
print(" EVALUATING MODELS ON FINAL 2-WEEK HOLDOUT SET ")
print("=" * 75)

results_list = []
final_fold_models = {}

# Record fitted models
for name, est in models.items():
    print(f"Fitting and evaluating {name}...")
    m = clone(est)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    
    final_fold_models[name] = m
    
    row = {'Model': name}
    row.update(evaluate(y_te, pred))
    row['Inner CV Best RMSE'] = inner_cv_rmse.get(name, np.nan)
    results_list.append(row)

final_df = pd.DataFrame(results_list).sort_values('RMSE', ascending=False)

print("\n" + "=" * 75)
print("   MODEL BENCHMARK RESULTS — FINAL 2-WEEK TEMPORAL HOLDOUT   ")
print("=" * 75)
print(f"Holdout window: {cutoff_date} to {unique_dates[-1]} ({len(X_te):,} test observations)")
print("-" * 75)
print(final_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("=" * 75)

# =====================================================================
# 6. SHAP RANK AGREEMENT (XGBoost vs LightGBM)
# =====================================================================
print("\n" + "=" * 75)
print("   SHAP FEATURE-RANK AGREEMENT ON THE FINAL HOLDOUT   ")
print("=" * 75)

shap_X = X_te.sample(n=min(SHAP_SAMPLE, len(X_te)), random_state=SEED)
importance = {}
for name in ['XGBoost', 'LightGBM']:
    if name in final_fold_models:
        try:
            explainer = shap.TreeExplainer(final_fold_models[name])
            sv = explainer.shap_values(shap_X, check_additivity=False)
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 0]
            importance[name] = np.abs(sv).mean(axis=0)
        except Exception as exc:
            print(f"SHAP failed for {name}: {exc}")

if len(importance) >= 2:
    imp_df = pd.DataFrame(importance, index=feature_cols)
    agreement_rows = []
    for a, b in combinations(importance.keys(), 2):
        rho, _ = spearmanr(imp_df[a], imp_df[b])
        top_a = set(imp_df[a].nlargest(TOP_K).index)
        top_b = set(imp_df[b].nlargest(TOP_K).index)
        agreement_rows.append({
            'Model A': a, 'Model B': b,
            'Spearman rho': rho,
            f'Top-{TOP_K} overlap': len(top_a & top_b),
        })
    agreement_df = pd.DataFrame(agreement_rows)
    print(agreement_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for name in importance:
        print(f"\nTop {TOP_K} features by mean |SHAP| — {name}:")
        print(", ".join(imp_df[name].nlargest(TOP_K).index))

    imp_df.sort_values('XGBoost', ascending=False).to_csv(
        paths.SCRIPTS_DIR / 'shap_importance_by_model.csv')
    agreement_df.to_csv(paths.SCRIPTS_DIR / 'shap_rank_agreement.csv', index=False)

# =====================================================================
# 7. SAVE OUTPUTS
# =====================================================================
final_df.to_csv(paths.SCRIPTS_DIR / 'model_comparison_last_2_weeks_results.csv', index=False)
print(f"\nSaved final holdout report to: {paths.SCRIPTS_DIR / 'model_comparison_last_2_weeks_results.csv'}")