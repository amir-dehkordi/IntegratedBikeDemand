"""
hourly_demand_weekday_vs_weekend.py
-----------------------------------
Hourly demand profiles for the manuscript (June-August 2024).

Figure A: hourly_demand_all_trips.png / .svg
    Average number of trips per hour of day over ALL Citi Bike trips in the raw trip
    files (the whole system, not only the ferry-linked stations). Peak hours shaded.

Figure B: hourly_demand_weekday_vs_weekend_12min.png / .svg
    Average number of integrated bike-ferry trip records per hour of day, weekdays vs
    weekends, identified with the 12-minute transfer window.

CSV: hourly_demand_profiles_12min.csv (one row per hour; the numbers behind both figures)

Neither figure has a title. Put the caption in the manuscript.

Usage
    python integrated_trips/integrated_trip_diagnostics.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "integrated_trips") not in sys.path:
    sys.path.insert(0, str(_ROOT / "integrated_trips"))

from identify_integrated_trips import (
    BIKE_MAP_PATH,
    FERRY_SCHEDULE_DIR,
    OUTPUT_DIR,
    TRIP_DATA_DIR,
    add_output_fields,
    build_matching_map,
    deduplicate_rides,
    discover_ferry_schedule_files,
    expand_ferry_events_to_dates,
    load_bike_ferry_mapping,
    load_citibike_data,
    map_bike_terminals_to_schedule,
    match_first_mile,
    match_last_mile,
    normalize_ferry_events,
    parse_timetable_file,
)

# =====================================================================
# SETTINGS (same as the original figure scripts)
# =====================================================================
REFERENCE_BUFFER_MIN = 12

STUDY_START = "2024-06-01"
STUDY_END = "2024-08-31"

AM_PEAK = (7, 10)
PM_PEAK = (16, 19)

MIN_TRIP_MIN = None
MAX_TRIP_MIN = None

CHUNK_ROWS = 1_000_000   # rows per chunk when counting raw trip files

# Exact Color Palette Matching Your Reference Figure
BLUE = "#1f77b4"      # Weekday solid blue line with 'o' markers
RED = "#d62728"       # Weekend dashed red line with 's' markers
NEUTRAL = "#4a4a4a"   # All-days average line for system-wide plot
BAND = "#fccbbf"      # Soft peach/coral for peak hours shading (alpha=0.4)
GRID = "#dcdcdc"      # Light grey grid lines

_PANDAS_MAJOR = int(pd.__version__.split(".")[0])


# =====================================================================
# 1. ALL TRIPS: hourly counts from raw Citi Bike files
# =====================================================================
def _parse_times(series):
    """Parse Citi Bike timestamps whether or not they carry fractional seconds."""
    if _PANDAS_MAJOR >= 2:
        return pd.to_datetime(series, errors="coerce", format="ISO8601")
    return pd.to_datetime(series, errors="coerce")


def count_all_trips_by_hour(trip_dir):
    """Count every Citi Bike trip by start hour. Returns a Series indexed by hour timestamp."""
    trip_dir = Path(trip_dir)
    csv_files = sorted(trip_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No Citi Bike CSV files found under {trip_dir}")

    need_end = MIN_TRIP_MIN is not None or MAX_TRIP_MIN is not None
    usecols = ["started_at", "ended_at"] if need_end else ["started_at"]

    parts = []
    total_rows = total_bad_time = total_dropped_duration = 0

    print("\nCounting ALL Citi Bike trips by start hour (raw files)...")
    for csv_file in csv_files:
        header = pd.read_csv(csv_file, nrows=0).columns
        if not set(usecols) <= set(header):
            print(f"  skipping {csv_file.name}: missing columns {usecols}")
            continue

        file_rows = 0
        for chunk in pd.read_csv(csv_file, usecols=usecols, chunksize=CHUNK_ROWS, low_memory=False):
            file_rows += len(chunk)
            start = _parse_times(chunk["started_at"])
            valid = start.notna()
            keep = valid.copy()

            if need_end:
                end = _parse_times(chunk["ended_at"])
                minutes = (end - start).dt.total_seconds() / 60.0
                keep &= minutes.notna()
                if MIN_TRIP_MIN is not None:
                    keep &= minutes >= MIN_TRIP_MIN
                if MAX_TRIP_MIN is not None:
                    keep &= minutes <= MAX_TRIP_MIN

            total_bad_time += int((~valid).sum())
            total_dropped_duration += int(valid.sum() - keep.sum())
            parts.append(start[keep].dt.floor("60min").value_counts())

        total_rows += file_rows
        print(f"  {csv_file.name}: {file_rows:,} rows")

    if not parts:
        raise RuntimeError("No trips could be counted. Check the trip files and their column names.")

    counts = pd.concat(parts).groupby(level=0).sum().sort_index().astype("int64")
    print(f"  total rows read: {total_rows:,} | unparseable start time: {total_bad_time:,}"
          + (f" | removed by duration filter: {total_dropped_duration:,}" if need_end else ""))
    return counts


# =====================================================================
# 2. INTEGRATED TRIPS (12-minute window)
# =====================================================================
def build_integrated_trips(buffer_min):
    """Integrated bike-ferry trip records, built exactly as in pipeline."""
    mapping_raw, _ = load_bike_ferry_mapping(BIKE_MAP_PATH)
    target_ids = set(mapping_raw["bike_station_id"].dropna())

    schedule_files = discover_ferry_schedule_files(FERRY_SCHEDULE_DIR)
    all_events = []
    for _, file_row in schedule_files.iterrows():
        events, _ = parse_timetable_file(
            Path(file_row["source_file"]), file_row["route"], file_row["day_type"]
        )
        if not events.empty:
            all_events.append(events)
    if not all_events:
        raise ValueError("No valid ferry schedule events were parsed.")

    ferry_events, _ = normalize_ferry_events(pd.concat(all_events, ignore_index=True))
    schedule_terminals = sorted(ferry_events["ferry_terminal"].dropna().unique().tolist())
    alias_map, _, _ = map_bike_terminals_to_schedule(
        mapping_raw["ferry_terminal"].unique(), schedule_terminals, minimum_score=0.80
    )
    matching_map = build_matching_map(mapping_raw, alias_map)

    print("\nLoading Citi Bike trips at the ferry-linked stations...")
    trips, _ = load_citibike_data(TRIP_DATA_DIR, target_ids)

    relevant_dates = pd.concat(
        [trips["started_at"].dropna(), trips["ended_at"].dropna()], ignore_index=True
    )
    ferry_events_dated = expand_ferry_events_to_dates(ferry_events, relevant_dates)
    ferry_events_dated["ferry_terminal_canon"] = ferry_events_dated["ferry_terminal_canon"].astype(str)
    ferry_events_dated = ferry_events_dated.rename(columns={"ferry_terminal": "matched_schedule_terminal"})

    print(f"Matching integrated trips with buffer = {buffer_min} minutes...")
    first_raw = match_first_mile(trips, matching_map, ferry_events_dated, buffer_min)
    last_raw = match_last_mile(trips, matching_map, ferry_events_dated, buffer_min)
    first_mile, _ = deduplicate_rides(first_raw, subset=["ride_id", "trip_type"])
    last_mile, _ = deduplicate_rides(last_raw, subset=["ride_id", "trip_type"])

    combined = pd.concat(
        [add_output_fields(first_mile), add_output_fields(last_mile)], ignore_index=True
    )
    if combined.empty:
        raise RuntimeError("No integrated trips were identified.")

    return (
        combined.sort_values(["ride_id", "trip_type", "time_gap_minutes"], kind="stable")
        .drop_duplicates(subset=["ride_id", "trip_type"], keep="first")
        .reset_index(drop=True)
    )


def apply_duration_filter(trips):
    if MIN_TRIP_MIN is None and MAX_TRIP_MIN is None:
        return trips
    minutes = (pd.to_datetime(trips["ended_at"]) - pd.to_datetime(trips["started_at"])).dt.total_seconds() / 60.0
    keep = minutes.notna()
    if MIN_TRIP_MIN is not None:
        keep &= minutes >= MIN_TRIP_MIN
    if MAX_TRIP_MIN is not None:
        keep &= minutes <= MAX_TRIP_MIN
    return trips.loc[keep].reset_index(drop=True)


def hourly_counts(trips, time_col="started_at"):
    """Trip records per (date, hour), by start time, as a Series indexed by hour timestamp."""
    ts = pd.to_datetime(trips[time_col], errors="coerce").dropna()
    return ts.dt.floor("60min").value_counts().sort_index()


# =====================================================================
# 3. HOURLY AVERAGES
# =====================================================================
def hourly_profile(counts, days):
    """Average trips per hour of day over `days`: all days, weekdays only, weekends only."""
    df = counts.rename("trips").rename_axis("ts").reset_index()
    df["date"] = df["ts"].dt.normalize()
    df = df[df["date"].isin(days)].copy()
    df["hour"] = df["ts"].dt.hour
    is_weekend = df["date"].dt.dayofweek >= 5

    n_all = len(days)
    n_weekday = int((days.dayofweek < 5).sum())
    n_weekend = int((days.dayofweek >= 5).sum())

    def avg(frame, n_days):
        return frame.groupby("hour")["trips"].sum().reindex(range(24), fill_value=0) / n_days

    return pd.DataFrame({
        "all_days": avg(df, n_all),
        "weekday": avg(df[~is_weekend], n_weekday),
        "weekend": avg(df[is_weekend], n_weekend),
    })


def peak_summary(avg):
    """Share of the daily total in the AM and PM peaks, and the busiest hour."""
    total = avg.sum()
    am = avg.loc[AM_PEAK[0]:AM_PEAK[1] - 1].sum() / total * 100.0
    pm = avg.loc[PM_PEAK[0]:PM_PEAK[1] - 1].sum() / total * 100.0
    return am, pm, int(avg.idxmax())


# =====================================================================
# 4. FIGURES (original manuscript format)
# =====================================================================
def _style_axes(ax, y_max):
    hours = np.arange(24)
    ax.set_xticks(hours)
    ax.set_xlim(-0.6, 23.6)
    ax.set_ylim(0, y_max)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.set_xlabel("Hour of Day", fontsize=12, family="serif", labelpad=8)


def _shade_peaks(ax, hours, envelope, label="Peak Hours"):
    ax.fill_between(hours, 0, envelope, where=(hours >= AM_PEAK[0]) & (hours <= AM_PEAK[1]),
                    color=BAND, alpha=0.4, lw=0, zorder=2, label=label)
    ax.fill_between(hours, 0, envelope, where=(hours >= PM_PEAK[0]) & (hours <= PM_PEAK[1]),
                    color=BAND, alpha=0.4, lw=0, zorder=2)


def _save(fig, stem):
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=500, bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.linewidth": 0.8,
}


def plot_all_trips(profile, stem):
    """Figure A: average trips per hour, all Citi Bike trips."""
    hours = np.arange(24)
    y_all = profile["all_days"].reindex(range(24)).to_numpy(dtype=float)
    y_wd = profile["weekday"].reindex(range(24)).to_numpy(dtype=float)
    y_we = profile["weekend"].reindex(range(24)).to_numpy(dtype=float)
    envelope = np.maximum.reduce([y_all, y_wd, y_we])
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(12, 6.5))
        _shade_peaks(ax, hours, envelope)
        ax.plot(hours, y_all, color=NEUTRAL, marker="o", ms=4.5, lw=1.3, zorder=3, label="Average Trips (all days)")
        ax.plot(hours, y_wd, color=BLUE, marker="o", ms=5, lw=1.4, zorder=4, label="Weekday Avg Demand")
        ax.plot(hours, y_we, color=RED, marker="s", ms=5, lw=1.4, ls="--", zorder=4, label="Weekend Avg Demand")
        _style_axes(ax, envelope.max() * 1.10)
        ax.set_ylabel("Average Number of Trips", fontsize=12, family="serif", labelpad=8)
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index("Average Trips (all days)"), labels.index("Weekday Avg Demand"),
                 labels.index("Weekend Avg Demand"), labels.index("Peak Hours")]
        ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper right",
                  frameon=False, fontsize=11, prop={"family": "serif", "size": 11})
        _save(fig, stem)


def plot_weekday_vs_weekend(weekday, weekend, stem):
    """Figure B: integrated trips per hour, weekday vs weekend matching your reference layout."""
    hours = np.arange(24)
    wd = weekday.reindex(range(24)).to_numpy(dtype=float)
    we = weekend.reindex(range(24)).to_numpy(dtype=float)
    max_envelope = np.maximum(wd, we)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(12, 6.5))
        _shade_peaks(ax, hours, max_envelope)
        ax.plot(hours, wd, color=BLUE, marker="o", ms=6, lw=2.0, zorder=4, label="Weekday Avg Demand")
        ax.plot(hours, we, color=RED, marker="s", ms=6, lw=2.0, ls="--", zorder=4, label="Weekend Avg Demand")
        _style_axes(ax, max(max_envelope.max() * 1.10, 10))
        ax.set_ylabel("Average Number of Trips", fontsize=12, family="serif", labelpad=8)
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index("Weekday Avg Demand"), labels.index("Weekend Avg Demand"), labels.index("Peak Hours")]
        ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper right",
                  frameon=False, fontsize=11, prop={"family": "serif", "size": 11})
        _save(fig, stem)



# =====================================================================
# 5. MAIN
# =====================================================================
def run_hourly_demand():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(STUDY_START, STUDY_END, freq="D")

    print("=" * 72)
    print(f"HOURLY DEMAND PROFILES ({REFERENCE_BUFFER_MIN}-MIN TRANSFER WINDOW)")
    print(f"Study period {STUDY_START} to {STUDY_END}: {len(days)} days "
          f"({int((days.dayofweek < 5).sum())} weekdays, {int((days.dayofweek >= 5).sum())} weekend days)")
    print("=" * 72)

    # A. All trips
    all_counts = count_all_trips_by_hour(TRIP_DATA_DIR)
    in_window = all_counts[all_counts.index.normalize().isin(days)]
    print(f"  trips inside the study period: {int(in_window.sum()):,} "
          f"(outside it, ignored: {int(all_counts.sum() - in_window.sum()):,})")
    all_prof = hourly_profile(all_counts, days)

    # B. Integrated trips
    integrated = apply_duration_filter(build_integrated_trips(REFERENCE_BUFFER_MIN))
    int_counts = hourly_counts(integrated)
    int_in_window = int_counts[int_counts.index.normalize().isin(days)]
    int_prof = hourly_profile(int_counts, days)

    # Outputs — same names, dpi, and layout as the original figure script
    plot_all_trips(all_prof, OUTPUT_DIR / "hourly_demand_all_trips")
    plot_weekday_vs_weekend(int_prof["weekday"], int_prof["weekend"],
                            OUTPUT_DIR / "hourly_demand_weekday_vs_weekend_12min")

    share = 100.0 * int_prof["all_days"] / all_prof["all_days"].replace(0, np.nan)
    table = pd.DataFrame({
        "hour": range(24),
        "all_trips_avg": all_prof["all_days"].to_numpy(),
        "all_trips_weekday_avg": all_prof["weekday"].to_numpy(),
        "all_trips_weekend_avg": all_prof["weekend"].to_numpy(),
        "integrated_avg": int_prof["all_days"].to_numpy(),
        "integrated_weekday_avg": int_prof["weekday"].to_numpy(),
        "integrated_weekend_avg": int_prof["weekend"].to_numpy(),
        "integrated_share_of_all_trips_pct": share.to_numpy(),
    }).round(2)
    csv_path = OUTPUT_DIR / "hourly_demand_profiles_12min.csv"
    table.to_csv(csv_path, index=False)

    print(f"\nOutputs saved to {OUTPUT_DIR}:")
    print("  - hourly_demand_all_trips.png / .svg")
    print("  - hourly_demand_weekday_vs_weekend_12min.png / .svg")
    print(f"  - {csv_path.name}")



def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate hourly demand figures and the corresponding CSV."
    )
    parser.parse_args()
    run_hourly_demand()


if __name__ == "__main__":
    main()

