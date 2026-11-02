"""
hourly_rate_shap_sensitivity.py
-------------------------------
Exposure check for the dependent variable.

The main model predicts the RAW COUNT of integrated trips per date x station x period x trip stage.
The periods differ in length (AM 07-10 = 3 h, PM 16-19 = 3 h, off-peak = 18 h), so the period
indicators partly encode exposure time.  This script refits the model on HOURLY RATES and compares the
SHAP importance hierarchy with the count model:

  count          main model                         y = trips
  rate           trips per hour, unweighted         y = trips / hours
  rate_weighted  trips per hour, weighted by hours  y = trips / hours, sample_weight = hours
                 (each row counts in proportion to the hours it covers, so the fit targets the same
                  total counts as the count model but without the period-length shortcut)

Everything else is identical to the manuscript model: same panel (16 stations x 92 days x 3 periods x
2 trip stages), same 25 features, same tuned XGBoost parameters, temporal holdout (last 14 days),
TreeSHAP on all rows.  A seed noise floor (5 reseeded count models) is reported for reference.

Outputs (folder `hourly_rate_sensitivity/` next to this script):
  period_exposure_table.csv       mean trips per period and per hour (checks 12.16 / 26.78 / 39.83 ...)
  rate_shap_importance.csv        mean |SHAP| per feature and model
  rate_shap_share_pct.csv         % share per feature and model
  rate_shap_ranks.csv             rank per feature and model
  rate_shap_group_share.csv       % share per feature group and model
  rate_shap_summary.csv           Spearman / Kendall / top-k overlap / R2 vs the count model
  rate_shap_seed_noise.csv        seed noise floor
  rate_shap_rank_heatmap.png/svg  rank heatmap
  rate_shap_importance_bars.png   side-by-side bars (count vs rate), coloured by group

Run from the folder that contains paths.py.
Needs: pandas, numpy, scipy, scikit-learn, xgboost, shap, matplotlib.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================================
# CONFIGURATION
# =====================================================================
STUDY_START = "2024-06-01"
STUDY_END = "2024-08-31"
HOLDOUT_DAYS = 14
N_NOISE_SEEDS = 5
TOP_K_LIST = (5, 10)
DIRECTION_MIN = 0.05
SEED = 42

PERIOD_HOURS = {"AM_Peak": 3, "PM_Peak": 3, "Off_Peak": 18}   # 07-10, 16-19, remaining 18 h

XGB_PARAMS = dict(
    n_estimators=300, max_depth=8, learning_rate=0.03, subsample=0.7,
    colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=10.0, n_jobs=-1,
)

MODELS = ["count", "rate", "rate_weighted"]
BASE = "count"

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "hourly_rate_sensitivity"
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

FEATURES = [
    'is_first_mile', 'is_AM_peak', 'is_PM_peak',
    'is_weekend', 'temp_max_c', 'precipitation_mm',
    'max_wind_speed_kmh', 'mean_humidity_pct', 'dock_capacity', 'ferry_distance_m', 'landuse_entropy',
    'job_density', 'IntDensity', 'TransitAccess', 'road_length_m',
    'residential_pct', 'commercial_pct', 'industrial_pct', 'institutional_pct',
    'population_density', 'zero_vehicle_pct', 'age18_24_pct', 'age24_30_pct', 'higher_edu_pct', 'median_income',
]

FEATURE_GROUPS = {
    'Station characteristics': ['dock_capacity', 'ferry_distance_m', 'is_first_mile'],
    'Built environment': ['residential_pct', 'commercial_pct', 'industrial_pct', 'institutional_pct',
                          'landuse_entropy', 'TransitAccess', 'road_length_m', 'IntDensity'],
    'Socio-demographic': ['job_density', 'population_density', 'median_income', 'higher_edu_pct',
                          'zero_vehicle_pct', 'age18_24_pct', 'age24_30_pct'],
    'Temporal & weather': ['is_AM_peak', 'is_PM_peak', 'is_weekend', 'precipitation_mm', 'temp_max_c',
                           'mean_humidity_pct', 'max_wind_speed_kmh'],
}
GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
GROUP_COLORS = {'Station characteristics': '#1f77b4', 'Built environment': '#2ca02c',
                'Socio-demographic': '#ff7f0e', 'Temporal & weather': '#7f7f7f'}

DISPLAY = {
    'job_density': 'Employment Density', 'dock_capacity': 'Dock Capacity',
    'TransitAccess': 'Transit Access', 'IntDensity': 'Intersection Density',
    'is_AM_peak': 'Is AM Peak', 'is_PM_peak': 'Is PM Peak', 'is_first_mile': 'Trip Stage',
    'is_weekend': 'Weekend', 'age18_24_pct': 'Age 18–24 (%)', 'age24_30_pct': 'Age 25–30 (%)',
    'ferry_distance_m': 'Ferry Distance (m)', 'road_length_m': 'Road Length (m)',
    'temp_max_c': 'Temp Max (°C)', 'precipitation_mm': 'Precipitation (mm)',
    'max_wind_speed_kmh': 'Max Wind Speed (km/h)', 'mean_humidity_pct': 'Mean Humidity (%)',
    'landuse_entropy': 'Landuse Entropy', 'population_density': 'Population Density',
    'zero_vehicle_pct': 'Zero Vehicle (%)', 'higher_edu_pct': 'Higher Edu (%)',
    'median_income': 'Median Income', 'residential_pct': 'Residential (%)',
    'commercial_pct': 'Commercial (%)', 'industrial_pct': 'Industrial (%)',
    'institutional_pct': 'Institutional (%)',
}
clean = lambda f: DISPLAY.get(f, f)


def sid(s):
    return pd.to_numeric(s, errors="coerce").round(2)


# =====================================================================
# DATA (same construction as the manuscript model)
# =====================================================================
def assign_peak_period(hour):
    if pd.isna(hour):
        return "Off_Peak"
    if 7 <= hour < 10:
        return "AM_Peak"
    if 16 <= hour < 19:
        return "PM_Peak"
    return "Off_Peak"


def load_static_tables(paths):
    def rd(p):
        d = pd.read_csv(p)
        d["Bike Station ID"] = sid(d["Bike Station ID"])
        return d.drop_duplicates("Bike Station ID")

    sld = rd(paths.SLD_STATION_CSV)[["Bike Station ID", "IntDensity", "TransitAccess"]]
    lu = rd(paths.LANDUSE_500M_CSV)[["Bike Station ID", "residential_pct", "commercial_pct",
                                     "industrial_pct", "institutional_pct"]]
    acs = rd(paths.ACS_STATION_CSV)[["Bike Station ID", "dock_capacity", "ferry_distance_m", "landuse_entropy",
                                     "job_density", "population_density", "zero_vehicle_pct", "age18_24_pct",
                                     "age24_30_pct", "higher_edu_pct", "median_income"]]
    osm = rd(paths.OSM_STATION_CSV)[["Bike Station ID", "road_length_m"]]
    return sld.merge(lu, on="Bike Station ID").merge(acs, on="Bike Station ID").merge(osm, on="Bike Station ID")


def load_inputs(paths):
    trips = pd.read_csv(paths.TRIPS_CSV, dtype={'start_station_id': str, 'end_station_id': str})
    weather = pd.read_csv(paths.WEATHER_CSV)
    weather['date'] = pd.to_datetime(weather['date']).dt.date
    return trips, weather, load_static_tables(paths)


def build_dataset(trips, weather, static):
    t = trips[["started_at", "start_station_id", "end_station_id", "trip_type"]].copy()
    t["started_at"] = pd.to_datetime(t["started_at"], errors="coerce")
    t["date"] = t["started_at"].dt.date
    t["peak_period"] = t["started_at"].dt.hour.apply(assign_peak_period)
    end = pd.to_numeric(t["end_station_id"], errors="coerce")
    start = pd.to_numeric(t["start_station_id"], errors="coerce")
    t["anchor_station_id"] = np.where(t["trip_type"] == "first_mile", end, start)
    t = t.dropna(subset=["anchor_station_id", "date"])
    keys = ['date', 'anchor_station_id', 'peak_period', 'trip_type']
    obs = t.groupby(keys).size().reset_index(name='trip_demand')

    stations = np.sort(obs['anchor_station_id'].unique())
    dates = pd.date_range(start=STUDY_START, end=STUDY_END).date
    grid_index = pd.MultiIndex.from_product(
        [dates, stations, list(PERIOD_HOURS), ['first_mile', 'last_mile']], names=keys)
    grid = pd.DataFrame(index=grid_index).reset_index()
    panel = pd.merge(grid, obs, on=keys, how='left').fillna({'trip_demand': 0})
    panel['is_weekend'] = panel['date'].apply(lambda x: 1 if x.weekday() >= 5 else 0)
    panel['is_first_mile'] = (panel['trip_type'] == 'first_mile').astype(int)
    panel['is_AM_peak'] = (panel['peak_period'] == 'AM_Peak').astype(int)
    panel['is_PM_peak'] = (panel['peak_period'] == 'PM_Peak').astype(int)
    panel['anchor_station_id'] = sid(panel['anchor_station_id'])
    panel['hours'] = panel['peak_period'].map(PERIOD_HOURS).astype(float)
    panel['trip_rate'] = panel['trip_demand'] / panel['hours']

    ds = pd.merge(panel, static, left_on='anchor_station_id', right_on='Bike Station ID', how='inner')
    ds = pd.merge(ds, weather, on='date', how='inner')
    return ds


# =====================================================================
# MODELLING
# =====================================================================
def feature_direction(x, s):
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)
    if np.nanstd(x) == 0 or np.nanstd(s) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return spearmanr(x, s)[0]


def split(ds):
    last = sorted(ds['date'].unique())[-HOLDOUT_DAYS:]
    m = ds['date'].isin(last)
    return ds[~m], ds[m]


def fit_and_explain(ds, model_name, seed):
    train, test = split(ds)
    ytr, yte = (train['trip_demand'], test['trip_demand']) if model_name == "count" \
        else (train['trip_rate'], test['trip_rate'])
    w = train['hours'] if model_name == "rate_weighted" else None
    model = xgb.XGBRegressor(**XGB_PARAMS, random_state=seed)
    model.fit(train[FEATURES], ytr, sample_weight=w)
    pred = model.predict(test[FEATURES])
    # put every model on the COUNT scale so R2/RMSE are comparable
    pred_count = pred if model_name == "count" else pred * test['hours'].to_numpy()
    X = ds[FEATURES]
    sv = np.asarray(shap.TreeExplainer(model).shap_values(X))
    if sv.ndim == 3:
        sv = sv[:, :, 0]
    return {
        'imp': np.abs(sv).mean(axis=0),
        'dir': np.array([feature_direction(X[c], sv[:, i]) for i, c in enumerate(FEATURES)]),
        'r2_own': r2_score(yte, pred), 'r2_count': r2_score(test['trip_demand'], pred_count),
        'rmse_count': float(np.sqrt(mean_squared_error(test['trip_demand'], pred_count))),
    }


def compare(imp, base_imp):
    a = pd.Series(base_imp, index=FEATURES)
    b = pd.Series(imp, index=FEATURES)
    shift = (b.rank(ascending=False, method="min") - a.rank(ascending=False, method="min")).abs()
    out = {'spearman_rho': spearmanr(a, b)[0], 'kendall_tau': kendalltau(a, b)[0],
           'max_rank_shift': int(shift.max()), 'max_shift_feature': clean(shift.idxmax())}
    for k in TOP_K_LIST:
        out[f'top{k}_overlap'] = len(set(a.nlargest(k).index) & set(b.nlargest(k).index))
    return out


# =====================================================================
# FIGURES
# =====================================================================
def plot_rank_heatmap(rank_df, stem):
    plt.rcParams['font.family'] = 'serif'
    order = rank_df[BASE].sort_values().index
    mat = rank_df.loc[order, MODELS].to_numpy()
    n = len(order)
    labels = {"count": "Count\n(main model)", "rate": "Rate\n(trips/h)", "rate_weighted": "Rate\n(hour-weighted)"}
    fig, ax = plt.subplots(figsize=(6.5, 8.5))
    ax.imshow(mat, cmap="Blues_r", aspect="auto", vmin=1, vmax=n)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([labels[m] for m in MODELS])
    ax.set_yticks(range(n))
    ax.set_yticklabels([clean(f) for f in order], fontsize=10)
    for i in range(n):
        for j in range(len(MODELS)):
            v = int(mat[i, j])
            ax.text(j, i, v, ha="center", va="center", fontsize=9, color="white" if v <= n * 0.45 else "black")
    plt.tight_layout()
    plt.savefig(f"{stem}.png", dpi=600)
    plt.savefig(f"{stem}.svg")
    plt.close()


def plot_bars(share_df, stem, top=15):
    plt.rcParams['font.family'] = 'serif'
    order = share_df[BASE].sort_values(ascending=False).index[:top][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), sharey=True)
    for ax, m, title in zip(axes, [BASE, "rate"], ["Count model (main)", "Hourly-rate model"]):
        ax.barh([clean(f) for f in order], share_df.loc[order, m], color=[GROUP_COLORS[GROUP_OF[f]] for f in order])
        ax.set_xlabel("Share of total mean |SHAP| (%)")
        ax.set_title(title, fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    axes[1].legend(handles, GROUP_COLORS.keys(), fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{stem}.png", dpi=400)
    plt.close()


# =====================================================================
# MAIN
# =====================================================================
def run(ds):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per = ds.groupby('peak_period').agg(mean_trips_per_row=('trip_demand', 'mean'), hours=('hours', 'first'))
    per['mean_trips_per_hour'] = per['mean_trips_per_row'] / per['hours']
    per = per.loc[list(PERIOD_HOURS)]
    per.to_csv(OUT_DIR / "period_exposure_table.csv")
    print("=" * 72)
    print("EXPOSURE: mean integrated trips per station-day-stage row, by period")
    print("=" * 72)
    print(per.round(2).to_string())

    print("\nFitting models (count, rate, rate_weighted)...")
    res = {m: fit_and_explain(ds, m, SEED) for m in MODELS}
    for m in MODELS:
        print(f"  {m:<14} holdout R2 (own target) = {res[m]['r2_own']:.3f} | R2 on counts = {res[m]['r2_count']:.3f} "
              f"| RMSE on counts = {res[m]['rmse_count']:.3f}")

    imp = pd.DataFrame({m: res[m]['imp'] for m in MODELS}, index=FEATURES)
    share = imp / imp.sum() * 100
    rank = imp.rank(ascending=False, method="min").astype(int)
    group = pd.DataFrame({m: {g: imp.loc[fs, m].sum() / imp[m].sum() * 100 for g, fs in FEATURE_GROUPS.items()}
                          for m in MODELS})

    rows = []
    for m in MODELS:
        r = {'model': m, 'holdout_R2_own': res[m]['r2_own'], 'holdout_R2_counts': res[m]['r2_count'],
             'holdout_RMSE_counts': res[m]['rmse_count']}
        r.update(compare(res[m]['imp'], res[BASE]['imp']))
        # agreement among the non-temporal variables only
        nt = [f for f in FEATURES if GROUP_OF[f] != 'Temporal & weather']
        r['spearman_rho_non_temporal'] = spearmanr(pd.Series(res[BASE]['imp'], index=FEATURES)[nt],
                                                   pd.Series(res[m]['imp'], index=FEATURES)[nt])[0]
        rows.append(r)
    summ = pd.DataFrame(rows)

    # seed noise floor for the count model
    noise = []
    for s in range(1, N_NOISE_SEEDS + 1):
        rr = fit_and_explain(ds, BASE, SEED + s)
        c = compare(rr['imp'], res[BASE]['imp'])
        noise.append({'seed': SEED + s, **{k: c[k] for k in ['spearman_rho', 'kendall_tau', 'top5_overlap', 'top10_overlap']}})
    noise = pd.DataFrame(noise)

    # outputs
    imp.to_csv(OUT_DIR / "rate_shap_importance.csv")
    share.to_csv(OUT_DIR / "rate_shap_share_pct.csv")
    rank.to_csv(OUT_DIR / "rate_shap_ranks.csv")
    group.to_csv(OUT_DIR / "rate_shap_group_share.csv")
    summ.to_csv(OUT_DIR / "rate_shap_summary.csv", index=False)
    noise.to_csv(OUT_DIR / "rate_shap_seed_noise.csv", index=False)
    plot_rank_heatmap(rank, str(OUT_DIR / "rate_shap_rank_heatmap"))
    plot_bars(share, str(OUT_DIR / "rate_shap_importance_bars"))

    # console report
    order = rank[BASE].sort_values().index
    show = rank.loc[order].copy()
    show.index = [clean(f) for f in order]
    print("\n" + "=" * 72)
    print("IMPORTANCE RANK (1 = most important)")
    print("=" * 72)
    print(show.to_string())

    print("\n" + "=" * 72)
    print("GROUP IMPORTANCE SHARE (% of total mean |SHAP|)")
    print("=" * 72)
    print(group.round(1).to_string())

    print("\n" + "=" * 72)
    print("PERIOD INDICATORS: share of total importance (%) and rank")
    print("=" * 72)
    pk = pd.DataFrame({m: {'Is AM Peak share': share.loc['is_AM_peak', m], 'Is AM Peak rank': rank.loc['is_AM_peak', m],
                           'Is PM Peak share': share.loc['is_PM_peak', m], 'Is PM Peak rank': rank.loc['is_PM_peak', m]}
                       for m in MODELS})
    print(pk.round(2).to_string())

    print("\n" + "=" * 72)
    print("AGREEMENT WITH THE COUNT MODEL")
    print("=" * 72)
    cols = ['model', 'holdout_R2_counts', 'spearman_rho', 'kendall_tau', 'spearman_rho_non_temporal',
            'top5_overlap', 'top10_overlap', 'max_rank_shift', 'max_shift_feature']
    print(summ[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 72)
    print(f"SEED NOISE FLOOR ({N_NOISE_SEEDS} reseeded count models)")
    print("=" * 72)
    print(noise.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    top_c = list(pd.Series(res[BASE]['imp'], index=FEATURES).nlargest(5).index)
    top_r = list(pd.Series(res['rate']['imp'], index=FEATURES).nlargest(5).index)
    print("\nTop-5 count model:", [clean(f) for f in top_c])
    print("Top-5 rate model :", [clean(f) for f in top_r])
    print(f"\nSaved to: {OUT_DIR}")


def main():
    import paths
    trips, weather, static = load_inputs(paths)
    ds = build_dataset(trips, weather, static)
    print(f"Panel rows: {len(ds):,} | stations: {ds['anchor_station_id'].nunique()} | days: {ds['date'].nunique()}")
    run(ds)


if __name__ == "__main__":
    main()