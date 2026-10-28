"""
window_shap_sensitivity.py
--------------------------
Does the XGBoost feature-importance ranking change when the ferry-transfer window
used to identify integrated bike-ferry trips changes?

Model specification (this version)
  - The separate AM-peak / PM-peak indicators are replaced by ONE indicator,
    is_commuting_time = 1 for weekday trips in 07:00-10:00 or 16:00-20:00.
  - Panel rows are station x day x time block x trip stage, with two time blocks:
      Commute      weekday 07:00-10:00 and 16:00-20:00   (7 h)
      Non_Commute  all other weekday hours (17 h); on weekends the whole day (24 h)
    Weekends have no commuting block, because there are no commuting peaks on weekends.
  - Blocks differ in length, so by default (USE_HOURLY_RATE = True) the model predicts trips
    PER HOUR and each row is weighted by its length in hours. Without this, a long block
    would hold more trips than a short one for no behavioural reason, and the commuting
    indicator would pick up a spurious negative sign. R2 / RMSE / MAE are still computed
    on trip counts (predicted rate x block hours), so they stay comparable to count models.
  - Temporal evaluation: the last HOLDOUT_DAYS days are the test set.
  - Every window uses the SAME stations and dates (those of the baseline window).
  - SHAP importance is the mean |SHAP| over all rows (in trips-per-hour units when
    USE_HOURLY_RATE is True).

Synchronized with citibike_ferry_integration_final_2.py matching logic.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import shap
import xgboost as xgb
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================================
# CONFIGURATION
# =====================================================================
WINDOWS = [8, 10, 12, 15, 20]        # transfer windows (minutes)
BASELINE_WINDOW = 12                  # window used in the manuscript
EXPECTED_BASELINE_TRIPS = 231_902     # integrated trip records at the baseline window (manuscript)

# Commuting hours (weekdays only). End hours are exclusive.
AM_PEAK = (7, 10)
PM_PEAK = (16, 20)

# True: model trips per hour, weight rows by block length (recommended, see docstring).
# False: model raw trip counts per block.
USE_HOURLY_RATE = True

SINGLE_PASS = True                    # match once at the widest window, then filter by time gap

HOLDOUT_DAYS = 14                     # last N days are the temporal test set
STUDY_START = "2024-06-01"
STUDY_END = "2024-08-31"
N_NOISE_SEEDS = 5
TOP_K_LIST = (5, 10)
DIRECTION_MIN = 0.05
SEED = 42

# NOTE: these values were tuned for the OLD target (trip counts, separate AM/PM indicators).
# Re-run the comparison script with the new specification and update them if they change.
XGB_PARAMS = dict(
    n_estimators=300, max_depth=8, learning_rate=0.03, subsample=0.7,
    colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=10.0, n_jobs=-1,
)

MATCHING_MODULE_DIR = None

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "window_shap_sensitivity"

for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))
if MATCHING_MODULE_DIR is not None:
    sys.path.insert(0, str(MATCHING_MODULE_DIR))

try:
    import paths
except ImportError as exc:
    raise SystemExit(
        "Cannot import `paths`. Run this from the same place as your other model scripts."
    ) from exc

try:
    import citibike_ferry_integration_final_2 as cf
except ImportError as exc:
    raise SystemExit(
        "Cannot import citibike_ferry_integration_final_2.py. Ensure it is in the same directory."
    ) from exc

assert BASELINE_WINDOW in WINDOWS, "BASELINE_WINDOW must be one of WINDOWS"

COMMUTE_HOURS = (AM_PEAK[1] - AM_PEAK[0]) + (PM_PEAK[1] - PM_PEAK[0])

# =====================================================================
# FEATURES, GROUPS, LABELS
# =====================================================================
FEATURES = [
    'is_first_mile', 'is_commuting_time',
    'is_weekend', 'temp_max_c', 'precipitation_mm',
    'max_wind_speed_kmh', 'mean_humidity_pct', 'dock_capacity', 'ferry_distance_m', 'landuse_entropy',
    'job_density', 'IntDensity', 'TransitAccess', 'road_length_m',
    'residential_pct', 'commercial_pct', 'industrial_pct', 'institutional_pct',
    'population_density', 'zero_vehicle_pct', 'age18_24_pct', 'age24_30_pct', 'higher_edu_pct', 'median_income',
]

FEATURE_GROUPS = {
    'Station & trip characteristics': ['dock_capacity', 'ferry_distance_m', 'is_first_mile'],
    'Built environment': ['residential_pct', 'commercial_pct', 'industrial_pct', 'institutional_pct',
                          'landuse_entropy', 'TransitAccess', 'road_length_m', 'IntDensity'],
    'Socio-demographic': ['job_density', 'population_density', 'median_income', 'higher_edu_pct',
                          'zero_vehicle_pct', 'age18_24_pct', 'age24_30_pct'],
    'Temporal & weather': ['is_commuting_time', 'is_weekend', 'precipitation_mm', 'temp_max_c',
                           'mean_humidity_pct', 'max_wind_speed_kmh'],
}

DISPLAY = {
    'job_density': 'Employment Density', 'dock_capacity': 'Dock Capacity',
    'TransitAccess': 'Transit Access', 'IntDensity': 'Intersection Density',
    'is_commuting_time': 'Commuting Time', 'is_first_mile': 'Trip Stage',
    'is_weekend': 'Weekend', 'age18_24_pct': 'Age 18–24 (%)', 'age24_30_pct': 'Age 25–30 (%)',
    'ferry_distance_m': 'Ferry Distance (m)', 'road_length_m': 'Road Length (m)',
    'temp_max_c': 'Max Temperature (°C)', 'precipitation_mm': 'Precipitation (mm)',
    'max_wind_speed_kmh': 'Max Wind Speed (km/h)', 'mean_humidity_pct': 'Mean Humidity (%)',
    'landuse_entropy': 'Land-use Entropy', 'population_density': 'Population Density',
    'zero_vehicle_pct': 'Zero-vehicle HH (%)', 'higher_edu_pct': 'Higher Education (%)',
    'median_income': 'Median Income', 'residential_pct': 'Residential (%)',
    'commercial_pct': 'Commercial (%)', 'industrial_pct': 'Industrial (%)',
    'institutional_pct': 'Institutional (%)',
}


def clean_name(f):
    return DISPLAY.get(f, f.replace('_', ' ').title())


# =====================================================================
# 1. INTEGRATED-TRIP IDENTIFICATION FOR EVERY WINDOW (via final_2 logic)
# =====================================================================
def load_matching_inputs():
    mapping_raw, _ = cf.load_bike_ferry_mapping(cf.BIKE_MAP_PATH)
    target_ids = set(mapping_raw["bike_station_id"].dropna())

    schedule_files = cf.discover_ferry_schedule_files(cf.FERRY_SCHEDULE_DIR)
    all_events = []
    for _, row in schedule_files.iterrows():
        events, _ = cf.parse_timetable_file(Path(row["source_file"]), row["route"], row["day_type"])
        if not events.empty:
            all_events.append(events)
    if not all_events:
        raise ValueError("No valid ferry schedule events were parsed.")

    ferry_events, _ = cf.normalize_ferry_events(pd.concat(all_events, ignore_index=True))
    schedule_terminals = sorted(ferry_events["ferry_terminal"].dropna().unique().tolist())
    alias_map, _, _ = cf.map_bike_terminals_to_schedule(
        mapping_raw["ferry_terminal"].unique(), schedule_terminals, minimum_score=0.80
    )
    matching_map = cf.build_matching_map(mapping_raw, alias_map)

    print("Loading Citi Bike trip data...")
    trips, _ = cf.load_citibike_data(cf.TRIP_DATA_DIR, target_ids)

    relevant_dates = pd.concat(
        [trips["started_at"].dropna(), trips["ended_at"].dropna()], ignore_index=True
    )
    events_dated = cf.expand_ferry_events_to_dates(ferry_events, relevant_dates)
    events_dated["ferry_terminal_canon"] = events_dated["ferry_terminal_canon"].astype(str)
    events_dated = events_dated.rename(columns={"ferry_terminal": "matched_schedule_terminal"})
    return trips, matching_map, events_dated


def match_for_window(trips, matching_map, events_dated, buffer_min):
    first_raw = cf.match_first_mile(trips, matching_map, events_dated, buffer_min)
    last_raw = cf.match_last_mile(trips, matching_map, events_dated, buffer_min)

    first_mile, _ = cf.deduplicate_rides(first_raw, subset=["ride_id", "trip_type"])
    last_mile, _ = cf.deduplicate_rides(last_raw, subset=["ride_id", "trip_type"])

    first_mile = cf.add_output_fields(first_mile)
    last_mile = cf.add_output_fields(last_mile)

    combined = pd.concat([first_mile, last_mile], ignore_index=True)

    if not combined.empty and "ride_id" in combined.columns:
        combined = (
            combined.sort_values(["ride_id", "trip_type", "time_gap_minutes"], kind="stable")
            .drop_duplicates(subset=["ride_id", "trip_type"], keep="first")
            .reset_index(drop=True)
        )
    return combined


def identify_trips_all_windows():
    trips, matching_map, events_dated = load_matching_inputs()
    integrated = {}
    if SINGLE_PASS:
        widest = max(WINDOWS)
        print(f"\nMatching once at {widest} min, then filtering by time gap...")
        widest_df = match_for_window(trips, matching_map, events_dated, widest)
        for w in WINDOWS:
            integrated[w] = widest_df.loc[widest_df["time_gap_minutes"] <= w].copy()
    else:
        for w in WINDOWS:
            print(f"\nMatching with {w}-minute window using final_2 rules...")
            integrated[w] = match_for_window(trips, matching_map, events_dated, w)
    return integrated


# =====================================================================
# 2. MODEL DATASET
# =====================================================================
def is_commute_hour(hour):
    return (AM_PEAK[0] <= hour < AM_PEAK[1]) or (PM_PEAK[0] <= hour < PM_PEAK[1])


def to_float(series):
    return pd.to_numeric(series.astype(str), errors="coerce").astype("float64")


def load_static_tables():
    sld_df = pd.read_csv(paths.SLD_STATION_CSV)
    landuse_df = pd.read_csv(paths.LANDUSE_500M_CSV)
    weather_df = pd.read_csv(paths.WEATHER_CSV)
    station_acs_df = pd.read_csv(paths.ACS_STATION_CSV)
    station_osm_df = pd.read_csv(paths.OSM_STATION_CSV)

    weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date
    for df in (sld_df, landuse_df, station_acs_df, station_osm_df):
        df['Bike Station ID'] = pd.to_numeric(df['Bike Station ID'], errors='coerce').round(2)

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
    return spatial_complete, weather_df


def prepare_events(integrated):
    """Matched trips -> one row per trip with date, time block and anchor station."""
    t = integrated[["started_at", "start_station_id", "end_station_id", "trip_type"]].copy()
    t["started_at"] = pd.to_datetime(t["started_at"], errors="coerce")
    t = t.dropna(subset=["started_at"])
    t["date"] = t["started_at"].dt.date
    t["hour"] = t["started_at"].dt.hour
    weekend = t["started_at"].dt.weekday >= 5
    commuting = (~weekend) & t["hour"].apply(is_commute_hour)
    t["time_block"] = np.where(commuting, "Commute", "Non_Commute")
    t["end_station_id"] = to_float(t["end_station_id"])
    t["start_station_id"] = to_float(t["start_station_id"])
    t["anchor_station_id"] = np.where(
        t["trip_type"] == "first_mile", t["end_station_id"], t["start_station_id"]
    )
    return t.dropna(subset=["anchor_station_id", "date"])


def build_dataset(integrated, spatial_complete, weather_df, stations=None, dates=None):
    """Zero-filled station x day x time block x stage panel.

    `stations` / `dates` fix the grid (the baseline window's), so every window is evaluated
    on exactly the same rows and only the demand counts differ.
    """
    t = prepare_events(integrated)

    keys = ['date', 'anchor_station_id', 'time_block', 'trip_type']
    observed = t.groupby(keys).size().reset_index(name='trip_demand')

    if dates is None:
        dates = pd.date_range(start=STUDY_START, end=STUDY_END).date
    if stations is None:
        stations = observed['anchor_station_id'].unique()
    observed = observed[observed['anchor_station_id'].isin(stations)]

    grid_index = pd.MultiIndex.from_product(
        [dates, stations, ['Commute', 'Non_Commute'], ['first_mile', 'last_mile']], names=keys,
    )
    grid = pd.DataFrame(index=grid_index).reset_index()
    grid['is_weekend'] = grid['date'].apply(lambda x: 1 if x.weekday() >= 5 else 0)
    # no commuting block on weekends
    grid = grid[~((grid['is_weekend'] == 1) & (grid['time_block'] == 'Commute'))]

    panel = pd.merge(grid, observed, on=keys, how='left').fillna({'trip_demand': 0})
    panel['is_first_mile'] = (panel['trip_type'] == 'first_mile').astype(int)
    panel['is_commuting_time'] = (panel['time_block'] == 'Commute').astype(int)
    panel['period_hours'] = np.where(
        panel['is_commuting_time'] == 1, COMMUTE_HOURS,
        np.where(panel['is_weekend'] == 1, 24, 24 - COMMUTE_HOURS),
    ).astype(float)
    panel['trip_rate'] = panel['trip_demand'] / panel['period_hours']
    panel['anchor_station_id'] = pd.to_numeric(panel['anchor_station_id'], errors='coerce').round(2)

    mid = pd.merge(panel, spatial_complete, left_on='anchor_station_id',
                   right_on='Bike Station ID', how='inner')
    return pd.merge(mid, weather_df, on='date', how='inner').reset_index(drop=True)


# =====================================================================
# 3. XGBOOST (TEMPORAL SPLIT) + SHAP
# =====================================================================
def feature_direction(x, s):
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)
    if np.nanstd(x) == 0 or np.nanstd(s) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return spearmanr(x, s)[0]


def fit_and_explain(ds, seed):
    X = ds[FEATURES]
    hours = ds['period_hours'].to_numpy(dtype=float)
    y_count = ds['trip_demand'].to_numpy(dtype=float)
    y_fit = (y_count / hours) if USE_HOURLY_RATE else y_count
    weights = hours if USE_HOURLY_RATE else None

    dates = np.sort(ds['date'].unique())
    if len(dates) <= HOLDOUT_DAYS:
        raise ValueError(f"Only {len(dates)} days in the panel; need more than HOLDOUT_DAYS={HOLDOUT_DAYS}.")
    cutoff = dates[-HOLDOUT_DAYS]
    train = (ds['date'] < cutoff).to_numpy()
    test = ~train

    model = xgb.XGBRegressor(**XGB_PARAMS, random_state=seed)
    model.fit(X.loc[train], y_fit[train],
              sample_weight=None if weights is None else weights[train])
    pred = model.predict(X.loc[test])
    pred_count = pred * hours[test] if USE_HOURLY_RATE else pred

    sv = np.asarray(shap.TreeExplainer(model).shap_values(X))   # SHAP on all rows
    if sv.ndim == 3:
        sv = sv[:, :, 0]

    return {
        'imp': np.abs(sv).mean(axis=0),
        'dir': np.array([feature_direction(X[c], sv[:, i]) for i, c in enumerate(FEATURES)]),
        'r2': r2_score(y_count[test], pred_count),
        'rmse': float(np.sqrt(mean_squared_error(y_count[test], pred_count))),
        'mae': float(mean_absolute_error(y_count[test], pred_count)),
        'n_rows': len(ds),
        'n_test_rows': int(test.sum()),
        'n_stations': int(ds['anchor_station_id'].nunique()),
        'demand_total': int(ds['trip_demand'].sum()),
        'holdout_start': cutoff,
    }


def compare_to_baseline(imp, base_imp):
    imp_s = pd.Series(imp, index=FEATURES)
    base_s = pd.Series(base_imp, index=FEATURES)
    r_w = imp_s.rank(ascending=False, method="min")
    r_b = base_s.rank(ascending=False, method="min")
    shift = (r_w - r_b).abs()
    out = {
        'spearman_rho': spearmanr(base_s, imp_s)[0],
        'kendall_tau': kendalltau(base_s, imp_s)[0],
        'max_rank_shift': int(shift.max()),
        'max_shift_feature': clean_name(shift.idxmax()),
    }
    for k in TOP_K_LIST:
        out[f'top{k}_overlap'] = len(set(base_s.nlargest(k).index) & set(imp_s.nlargest(k).index))
    out['top5_same_order'] = list(base_s.nlargest(5).index) == list(imp_s.nlargest(5).index)
    top10 = base_s.nlargest(10).index
    out['max_rank_shift_top10'] = int(shift.loc[top10].max())
    return out


def plot_rank_heatmap(rank_df, path_stem):
    plt.rcParams['font.family'] = 'serif'
    order = rank_df[BASELINE_WINDOW].sort_values().index
    mat = rank_df.loc[order, WINDOWS].to_numpy()
    n = len(order)

    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    ax.imshow(mat, cmap="Blues_r", aspect="auto", vmin=1, vmax=n)
    ax.set_xticks(range(len(WINDOWS)))
    ax.set_xticklabels([f"{w}\n(baseline)" if w == BASELINE_WINDOW else str(w) for w in WINDOWS])
    ax.set_yticks(range(n))
    ax.set_yticklabels([clean_name(f) for f in order], fontsize=10)
    for i in range(n):
        for j in range(len(WINDOWS)):
            v = int(mat[i, j])
            ax.text(j, i, v, ha="center", va="center", fontsize=9,
                    color="white" if v <= n * 0.45 else "black")
    base_idx = WINDOWS.index(BASELINE_WINDOW)
    ax.add_patch(Rectangle((base_idx - 0.5, -0.5), 1, n, fill=False, ec="black", lw=2))
    ax.set_xlabel("Transfer window (minutes)")          # no title: the caption carries it
    plt.tight_layout()
    plt.savefig(f"{path_stem}.png", dpi=600)
    plt.savefig(f"{path_stem}.svg")
    plt.close()


# =====================================================================
# 4. MAIN
# =====================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    integrated = identify_trips_all_windows()

    print("\n" + "=" * 72)
    print("INTEGRATED TRIPS PER WINDOW")
    print("=" * 72)
    print(f"  {'window':>8} {'trips':>10}")
    for w in WINDOWS:
        print(f"  {w:>5} min {len(integrated[w]):>10,}")
    n_base = len(integrated[BASELINE_WINDOW])
    if n_base != EXPECTED_BASELINE_TRIPS:
        print(f"\nNOTE: {BASELINE_WINDOW}-min trips = {n_base:,} "
              f"(manuscript reference: {EXPECTED_BASELINE_TRIPS:,}).")
    else:
        print(f"\nBaseline trip count matches the manuscript ({EXPECTED_BASELINE_TRIPS:,}).")

    spatial_complete, weather_df = load_static_tables()

    # Fix the grid (stations and dates) from the baseline window
    base_obs = prepare_events(integrated[BASELINE_WINDOW])
    stations = np.sort(base_obs['anchor_station_id'].unique())
    dates = pd.date_range(start=STUDY_START, end=STUDY_END).date
    print(f"\nPanel grid fixed from the {BASELINE_WINDOW}-min window: "
          f"{len(stations)} stations x {len(dates)} days")
    print(f"Target: {'trips per hour, rows weighted by block hours' if USE_HOURLY_RATE else 'trip counts per block'}"
          f" | commuting hours: {AM_PEAK[0]}-{AM_PEAK[1]} and {PM_PEAK[0]}-{PM_PEAK[1]} (weekdays)")

    results = {}
    base_ds = None
    print("\nFitting XGBoost (temporal holdout) and computing SHAP for each window...")
    for w in WINDOWS:
        ds = build_dataset(integrated[w], spatial_complete, weather_df, stations, dates)
        results[w] = fit_and_explain(ds, SEED)
        if w == BASELINE_WINDOW:
            base_ds = ds
        r = results[w]
        lost = len(integrated[w]) - r['demand_total']
        print(f"  {w:>2} min: {r['n_rows']:,} rows ({r['n_test_rows']:,} test), "
              f"{r['n_stations']} stations, {r['demand_total']:,} trips in panel "
              f"({lost:,} not in panel), holdout R2 = {r['r2']:.3f}, RMSE = {r['rmse']:.3f}")

    # Diagnostic: block length vs. trips (this is why the target is a rate)
    diag = (base_ds.groupby(['time_block', 'is_weekend'])
            .agg(rows=('trip_demand', 'size'), hours=('period_hours', 'first'),
                 mean_trips_per_row=('trip_demand', 'mean'),
                 mean_trips_per_hour=('trip_rate', 'mean')))
    print("\n" + "=" * 72)
    print(f"DIAGNOSTIC: demand by time block at the {BASELINE_WINDOW}-min window")
    print("=" * 72)
    print(diag.to_string(float_format=lambda x: f"{x:.3f}"))

    base = results[BASELINE_WINDOW]

    imp_df = pd.DataFrame({w: results[w]['imp'] for w in WINDOWS}, index=FEATURES)
    share_df = imp_df / imp_df.sum() * 100.0
    rank_df = imp_df.rank(ascending=False, method="min").astype(int)
    dir_df = pd.DataFrame({w: results[w]['dir'] for w in WINDOWS}, index=FEATURES)
    group_df = pd.DataFrame(
        {w: {g: imp_df.loc[m, w].sum() / imp_df[w].sum() * 100.0 for g, m in FEATURE_GROUPS.items()}
         for w in WINDOWS}
    )

    top10 = pd.Series(base['imp'], index=FEATURES).nlargest(10).index
    d_b = pd.Series(base['dir'], index=FEATURES)
    rows = []
    for w in WINDOWS:
        r = results[w]
        d_w = pd.Series(r['dir'], index=FEATURES)
        flip = ((d_b.abs() >= DIRECTION_MIN) & (d_w.abs() >= DIRECTION_MIN)
                & (np.sign(d_b) != np.sign(d_w)))
        row = {
            'window_min': w, 'integrated_trips': len(integrated[w]),
            'panel_rows': r['n_rows'], 'stations': r['n_stations'],
            'holdout_R2': r['r2'], 'holdout_RMSE': r['rmse'], 'holdout_MAE': r['mae'],
        }
        row.update(compare_to_baseline(r['imp'], base['imp']))
        row['direction_flips_top10'] = int(flip.loc[top10].sum())
        rows.append(row)
    summary = pd.DataFrame(rows)

    # Seed-to-seed noise floor: same data and window, only the model seed changes
    noise_rows = []
    for s in range(1, N_NOISE_SEEDS + 1):
        rr = fit_and_explain(base_ds, SEED + s)
        c = compare_to_baseline(rr['imp'], base['imp'])
        noise_rows.append({'seed': SEED + s, **{k: c[k] for k in
                           ['spearman_rho', 'kendall_tau', 'top5_overlap', 'top10_overlap',
                            'max_rank_shift_top10']}})
    noise = pd.DataFrame(noise_rows)

    order = rank_df[BASELINE_WINDOW].sort_values().index
    disp_rank = rank_df.loc[order].copy()
    disp_rank.index = [clean_name(f) for f in order]
    disp_rank.columns = [f"{w} min" for w in WINDOWS]

    print("\n" + "=" * 72)
    print(f"IMPORTANCE RANK BY WINDOW (1 = most important; baseline = {BASELINE_WINDOW} min)")
    print("=" * 72)
    print(disp_rank.to_string())

    print("\n" + "=" * 72)
    print("AGREEMENT WITH THE BASELINE RANKING")
    print("=" * 72)
    show = ['window_min', 'integrated_trips', 'holdout_R2', 'spearman_rho', 'kendall_tau',
            'top5_overlap', 'top10_overlap', 'top5_same_order', 'max_rank_shift_top10',
            'max_rank_shift', 'max_shift_feature', 'direction_flips_top10']
    print(summary[show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 72)
    print(f"SEED NOISE FLOOR ({N_NOISE_SEEDS} reseeded fits at the {BASELINE_WINDOW}-min window)")
    print("=" * 72)
    print(noise.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n  mean Spearman rho = {noise['spearman_rho'].mean():.3f}, "
          f"min = {noise['spearman_rho'].min():.3f}")
    other = summary[summary['window_min'] != BASELINE_WINDOW]
    print(f"  across other windows: Spearman rho {other['spearman_rho'].min():.3f}"
          f"-{other['spearman_rho'].max():.3f}")

    print("\n" + "=" * 72)
    print("GROUP IMPORTANCE SHARE (% of total mean |SHAP|)")
    print("=" * 72)
    print(group_df.round(1).rename(columns=lambda w: f"{w} min").to_string())

    print("\n" + "=" * 72)
    print(f"DIRECTION OF EFFECT AT THE BASELINE (Spearman of feature value vs SHAP), top 10")
    print("=" * 72)
    print(pd.Series(base['dir'], index=FEATURES).loc[top10].rename(index=clean_name)
          .round(3).to_string())

    imp_df.rename(columns=lambda w: f"{w}min").to_csv(OUT_DIR / "window_shap_importance.csv")
    share_df.rename(columns=lambda w: f"{w}min").to_csv(OUT_DIR / "window_shap_share_pct.csv")
    rank_df.rename(columns=lambda w: f"{w}min").to_csv(OUT_DIR / "window_shap_ranks.csv")
    dir_df.rename(columns=lambda w: f"{w}min").to_csv(OUT_DIR / "window_shap_direction.csv")
    group_df.rename(columns=lambda w: f"{w}min").to_csv(OUT_DIR / "window_shap_group_share.csv")
    summary.to_csv(OUT_DIR / "window_shap_summary.csv", index=False)
    noise.to_csv(OUT_DIR / "window_shap_seed_noise.csv", index=False)
    diag.to_csv(OUT_DIR / "time_block_diagnostic.csv")
    plot_rank_heatmap(rank_df, str(OUT_DIR / "window_shap_rank_heatmap"))

    print(f"\nSaved all outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
