"""
Identify potentially integrated Citi Bike–NYC Ferry trips for June–August 2024.

The matching is based on temporal association only. A matched Citi Bike ride and
ferry event are NOT claimed to belong to the same passenger.

First mile:
    Citi Bike ride ENDS at a target bike station, and the next usable ferry
    event at that station's mapped nearest ferry terminal occurs within the
    configured buffer (default: 12 minutes).

Last mile:
    A usable ferry event at the mapped nearest ferry terminal occurs, and the
    Citi Bike ride STARTS at the target bike station within the configured
    buffer after that event.

Important:
    A ferry terminal may be served by multiple NYC Ferry routes. Matching is
    therefore done by ferry terminal + day type, across ALL routes serving that
    terminal. The matched route and direction are retained in the output.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paths


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

BUFFER_MINUTES = 12

PROJECT_ROOT = paths.TRIPS_DIR
BIKE_MAP_PATH = paths.BIKE_FERRY_MAP_CSV
FERRY_SCHEDULE_DIR = paths.FERRY_SCHEDULE_DIR
TRIP_DATA_DIR = paths.CITIBIKE_TRIP_DATA_DIR
OUTPUT_DIR = paths.INTEGRATED_TRIPS_OUTPUT_DIR

KNOWN_ROUTES = {
    "south brooklyn": "South Brooklyn",
    "south_brooklyn": "South Brooklyn",
    "soundview": "Soundview",
    "st george": "St. George",
    "st_george": "St. George",
    "george": "St. George",
    "astoria": "Astoria",
    "east river": "East River",
    "east_river": "East River",
}

CITIBIKE_COLUMNS = [
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
]

OUTPUT_COLUMNS = [
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
    "bike_station_id",
    "bike_station_name",
    "ferry_terminal",
    "matched_ferry_terminal",
    "matched_route",
    "matched_direction",
    "ferry_time",
    "event_type",
    "time_gap_minutes",
    "day_type",
    "trip_type",
    "source_file",
    "source_timetable_section",
    "origin_terminal",
    "destination_terminal",
    "ferry_datetime",
]

TIME_RE = re.compile(
    r"^\s*(\d{1,2}):(\d{2})(?:\s*:\s*\d{2})?\s*(a\.?m\.?|p\.?m\.?)?\s*$",
    re.IGNORECASE,
)

# Remove phrases such as "Pier 11" without removing meaningful street
# numbers such as "E. 34th St." or "E. 90th St.".
PIER_RE = re.compile(r"\bpier\s*\d+[a-z]?\b", re.IGNORECASE)
ORDINAL_RE = re.compile(r"(?<=\d)(st|nd|rd|th)\b", re.IGNORECASE)

# Words that are safe to remove from terminal names for comparison.
DROP_TOKENS = {"the", "of"}

TOKEN_EXPAND = {
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "rd": "road",
    "ln": "lane",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
}


# -----------------------------------------------------------------------------
# NORMALIZATION HELPERS
# -----------------------------------------------------------------------------


def normalize_station_id(value) -> str | None:
    """Normalize Citi Bike station IDs for reliable joins."""
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None

    try:
        # Preserve IDs such as 5458.02 in a stable representation.
        return f"{float(text):.2f}"
    except (TypeError, ValueError):
        return text


def canonicalize_terminal_name(name: str) -> str:
    """
    Normalize ferry terminal names for matching.

    Important: retain meaningful street numbers (e.g. E. 34th vs E. 90th).
    "Pier 11", "Pier 6", etc. are removed as terminal qualifiers.
    "St. George" is handled before expanding "st" to "street".
    """
    if name is None or pd.isna(name):
        return ""

    text = str(name).strip().lower()

    # Common spelling/wording corrections & abbreviations.
    text = text.replace("roosvelet", "roosevelt")
    text = re.sub(r"\bst\.?\s+george\b", "saint george", text)
    text = re.sub(r"\bbk\b", "brooklyn", text)
    text = text.replace("&", " and ")

    # Remove pier qualifiers, but keep meaningful station numbers.
    text = PIER_RE.sub(" ", text)

    # Remove punctuation except alphanumeric characters/spaces.
    text = re.sub(r"[^a-z0-9]+", " ", text)

    # Normalize ordinal suffixes: 34th -> 34, 1st -> 1.
    text = ORDINAL_RE.sub("", text)

    tokens: list[str] = []
    for token in text.split():
        if token in DROP_TOKENS:
            continue
        if token == "st":
            token = "street"
        elif token == "saint":
            token = "saint"
        else:
            token = TOKEN_EXPAND.get(token, token)
        tokens.append(token)

    return " ".join(tokens)


def terminal_match_score(bike_terminal: str, schedule_terminal: str) -> float:
    """Return a conservative similarity score between terminal names."""
    a = canonicalize_terminal_name(bike_terminal)
    b = canonicalize_terminal_name(schedule_terminal)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    if a in b or b in a:
        return 0.95

    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return 0.0

    overlap = a_tokens & b_tokens
    if not overlap:
        return 0.0

    return len(overlap) / len(a_tokens | b_tokens)


def normalize_route_key(value: str) -> str:
    """Normalize route folder names for route identification."""
    return re.sub(r"\s+", " ", str(value).strip().lower().replace("-", " ").replace("_", " "))


# -----------------------------------------------------------------------------
# FILE DISCOVERY
# -----------------------------------------------------------------------------


def infer_route_from_path(path: Path) -> str:
    """Infer route from a parent path component."""
    candidates = [path.parent.name] + [p.name for p in path.parents]

    for candidate in candidates:
        key = normalize_route_key(candidate)
        if key in KNOWN_ROUTES:
            return KNOWN_ROUTES[key]

    return path.parent.name.replace("_", " ").replace("-", " ").title()


def infer_day_type(path: Path) -> str | None:
    """Infer weekday/weekend from a schedule filename or path."""
    text = path.name.lower()
    if "weekday" in text:
        return "weekday"
    if "weekend" in text:
        return "weekend"
    return None


def discover_ferry_schedule_files(schedule_dir: Path) -> pd.DataFrame:
    """Discover ferry timetable CSVs recursively."""
    rows = []
    extra_routes: list[str] = []

    if not schedule_dir.exists():
        raise FileNotFoundError(
            f"Ferry schedule directory does not exist: {schedule_dir}"
        )

    for path in sorted(schedule_dir.rglob("*.csv")):
        # Do not accidentally read generated files placed inside the schedule dir.
        if path.name.lower() in {
            "ferry_events.csv",
            "validation_report.txt",
        }:
            continue

        route = infer_route_from_path(path)
        day_type = infer_day_type(path)

        parent_key = normalize_route_key(path.parent.name)
        if parent_key not in KNOWN_ROUTES:
            extra_routes.append(str(path.relative_to(schedule_dir)))

        rows.append(
            {
                "source_file": str(path.resolve()),
                "relative_path": str(path.relative_to(schedule_dir)),
                "route": route,
                "day_type": day_type,
            }
        )

    files = pd.DataFrame(rows)
    if not files.empty:
        files.attrs["extra_routes"] = sorted(set(extra_routes))
    else:
        files.attrs["extra_routes"] = []

    return files


# -----------------------------------------------------------------------------
# FERRY TIME PARSING
# -----------------------------------------------------------------------------


def parse_clock_token(
    raw,
) -> tuple[int | None, int | None, str | None, str | None, bool]:
    """
    Parse a timetable clock value.

    Returns:
        hour, minute, raw_text, parse_kind, malformed
    """
    if raw is None or pd.isna(raw):
        return None, None, None, None, False

    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "-", "—"}:
        return None, None, None, None, False

    match = TIME_RE.match(text)
    if not match:
        return None, None, text, None, True

    hour = int(match.group(1))
    minute = int(match.group(2))
    ampm = match.group(3)

    if minute > 59:
        return None, None, text, None, True

    if ampm:
        # Explicit AM/PM must use a 1–12 clock.
        if hour < 1 or hour > 12:
            return None, None, text, None, True
        suffix = ampm.lower().replace(".", "")
        hour24 = hour % 12
        if suffix.startswith("p"):
            hour24 += 12
        return hour24, minute, text, "explicit_ampm", False

    # No AM/PM: accept 00:00–23:59 and the special 24:00 notation.
    if hour == 24 and minute == 0:
        return 24, 0, text, "24h", False

    if hour > 23:
        return None, None, text, None, True

    if hour > 12:
        return hour, minute, text, "24h", False

    return hour, minute, text, "clock", False


def chronological_minutes(parsed_rows: list[tuple]) -> list[tuple]:
    """
    Resolve timetable times that omit AM/PM by following chronological order.
    """
    prev = -1
    out = []

    for hour, minute, raw, kind, malformed in parsed_rows:
        if malformed or hour is None:
            out.append((None, raw, kind, True))
            continue

        candidate = hour * 60 + minute

        # Only apply 12-hour rollover to ambiguous clock values without AM/PM.
        if kind == "clock":
            while candidate < prev:
                candidate += 12 * 60

        if candidate > 24 * 60:
            out.append((None, raw, kind, True))
            continue

        prev = candidate
        out.append((candidate, raw, kind, False))

    return out


def parse_timetable_file(
    path: Path,
    route: str,
    day_type: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Parse an extracted timetable CSV where each terminal is represented by a
    column and each row contains scheduled times.
    """
    issues: list[str] = []
    path = Path(path).resolve()

    df = pd.read_csv(path, dtype=str)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    df = df.dropna(axis=1, how="all")

    if df.empty:
        issues.append(f"{path.name}: empty timetable")
        return pd.DataFrame(), issues

    # Only treat columns with at least one parseable time as terminal columns.
    terminal_candidates = []
    for col in df.columns:
        valid_count = 0
        for raw in df[col].tolist():
            hour, minute, _, _, malformed = parse_clock_token(raw)
            if hour is not None and minute is not None and not malformed:
                valid_count += 1
        if valid_count > 0:
            terminal_candidates.append(col)

    terminals = [str(c).strip() for c in terminal_candidates]

    if len(terminals) < 2:
        issues.append(
            f"{path.name}: fewer than 2 terminal columns detected ({terminals})"
        )
        return pd.DataFrame(), issues

    origin = terminals[0]
    destination = terminals[-1]
    direction = f"{origin} -> {destination}"

    events: list[dict] = []

    for col_idx, terminal in enumerate(terminals):
        parsed: list[tuple] = []

        for raw in df[terminal_candidates[col_idx]].tolist():
            hour, minute, raw_text, kind, malformed = parse_clock_token(raw)

            if raw_text is None:
                continue

            if malformed:
                issues.append(
                    f"{path.name}: malformed time '{raw_text}' in column '{terminal}'"
                )

            parsed.append((hour, minute, raw_text, kind, malformed))

        resolved = chronological_minutes(parsed)

        n_cols = len(terminals)
        if col_idx == 0:
            event_type = "departure"
        elif col_idx == n_cols - 1:
            event_type = "arrival"
        else:
            event_type = "scheduled_stop"

        for minutes, raw_text, kind, malformed in resolved:
            if minutes is None:
                continue

            if minutes >= 24 * 60:
                issues.append(
                    f"{path.name}: time '{raw_text}' in column '{terminal}' "
                    f"resolved to {minutes // 60:02d}:{minutes % 60:02d}; excluded"
                )
                continue

            try:
                source_rel = str(path.relative_to(PROJECT_ROOT.resolve()))
            except ValueError:
                source_rel = str(path)

            events.append(
                {
                    "route": route,
                    "ferry_terminal": terminal,
                    "ferry_terminal_canon": canonicalize_terminal_name(terminal),
                    "direction": direction,
                    "day_type": day_type,
                    "scheduled_minutes": int(minutes),
                    "scheduled_time": f"{minutes // 60:02d}:{minutes % 60:02d}",
                    "event_type": event_type,
                    "source_file": source_rel,
                    "source_timetable_section": path.stem,
                    "origin_terminal": origin,
                    "destination_terminal": destination,
                    "time_parse_kind": kind,
                    "raw_time": raw_text,
                }
            )

    return pd.DataFrame(events), issues


# -----------------------------------------------------------------------------
# BIKE/FERRY MAPPING
# -----------------------------------------------------------------------------


def load_bike_ferry_mapping(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load and validate bike station -> nearest ferry mapping."""
    mapping = pd.read_csv(path)
    mapping.columns = [str(c).strip() for c in mapping.columns]

    required = {
        "Ferry terminal",
        "Bike Station ID",
        "name bike station",
    }
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(
            f"bike.csv is missing required columns: {sorted(missing)}"
        )

    mapping["bike_station_id"] = mapping["Bike Station ID"].map(normalize_station_id)
    mapping["bike_station_name"] = mapping["name bike station"].astype(str).str.strip()
    mapping["ferry_terminal"] = mapping["Ferry terminal"].astype(str).str.strip()
    mapping["ferry_terminal_canon"] = mapping["ferry_terminal"].map(
        canonicalize_terminal_name
    )

    issues: list[str] = []

    duplicate_station_rows = (
        mapping.groupby("bike_station_id")["ferry_terminal"]
        .nunique(dropna=False)
    )
    conflicting = duplicate_station_rows[duplicate_station_rows > 1]
    if not conflicting.empty:
        issues.append(
            "Bike station IDs mapped to multiple different ferry terminals: "
            + ", ".join(conflicting.index.astype(str).tolist())
        )

    mapping = mapping.dropna(
        subset=["bike_station_id", "ferry_terminal_canon"]
    ).drop_duplicates(
        subset=["bike_station_id"], keep="first"
    )

    return mapping, issues


def map_bike_terminals_to_schedule(
    bike_terminals: Iterable[str],
    schedule_terminals: Iterable[str],
    minimum_score: float = 0.80,
) -> tuple[dict[str, list[str]], list[str], dict[str, list[tuple[float, str]]]]:
    """Map bike.csv terminal names to one or more schedule terminal names."""
    mapping: dict[str, list[str]] = {}
    unmatched: list[str] = []
    diagnostics: dict[str, list[tuple[float, str]]] = {}

    schedule_terminals = sorted(set(str(x).strip() for x in schedule_terminals if str(x).strip()))

    for bike_terminal in sorted(set(str(x).strip() for x in bike_terminals if str(x).strip())):
        scored = sorted(
            [
                (terminal_match_score(bike_terminal, schedule_terminal), schedule_terminal)
                for schedule_terminal in schedule_terminals
            ],
            key=lambda x: (-x[0], x[1]),
        )

        diagnostics[bike_terminal] = scored[:10]

        if not scored or scored[0][0] < minimum_score:
            unmatched.append(bike_terminal)
            mapping[bike_terminal] = []
            continue

        best_score = scored[0][0]
        aliases = sorted({name for score, name in scored if score >= minimum_score})
        mapping[bike_terminal] = aliases

    return mapping, unmatched, diagnostics


def build_matching_map(
    mapping_raw: pd.DataFrame,
    terminal_alias_map: dict[str, list[str]],
) -> pd.DataFrame:
    """Expand each bike station into all accepted schedule-terminal aliases."""
    rows: list[dict] = []

    for _, row in mapping_raw.iterrows():
        aliases = terminal_alias_map.get(row["ferry_terminal"], [])
        for alias in aliases:
            rows.append(
                {
                    "bike_station_id": row["bike_station_id"],
                    "bike_station_name": row["bike_station_name"],
                    "ferry_terminal": row["ferry_terminal"],
                    "ferry_terminal_canon": canonicalize_terminal_name(alias),
                    "schedule_terminal_name": alias,
                }
            )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# CITI BIKE DATA
# -----------------------------------------------------------------------------


def load_citibike_data(
    trip_dir: Path,
    target_ids: set[str],
) -> tuple[pd.DataFrame, dict]:
    """Load Citi Bike data in chunks, retaining only rows touching target stations."""
    diagnostics = {
        "files_loaded": [],
        "rows_read": 0,
        "missing_started_at": 0,
        "missing_ended_at": 0,
        "missing_ride_id": 0,
    }

    csv_files = sorted(trip_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No Citi Bike CSV files found under {trip_dir}"
        )

    frames: list[pd.DataFrame] = []

    for csv_file in csv_files:
        file_label = str(csv_file.relative_to(PROJECT_ROOT.resolve())) if csv_file.is_relative_to(PROJECT_ROOT.resolve()) else str(csv_file)
        print(f"Loading {file_label} ...")

        file_rows = 0

        for chunk in pd.read_csv(
            csv_file,
            usecols=lambda c: c in CITIBIKE_COLUMNS,
            dtype={
                "start_station_id": "string",
                "end_station_id": "string",
                "ride_id": "string",
            },
            chunksize=250_000,
            low_memory=False,
        ):
            file_rows += len(chunk)

            for required_col in ["started_at", "ended_at", "start_station_id", "end_station_id"]:
                if required_col not in chunk.columns:
                    raise ValueError(
                        f"Required Citi Bike column '{required_col}' missing in {csv_file}"
                    )

            diagnostics["missing_started_at"] += int(chunk["started_at"].isna().sum())
            diagnostics["missing_ended_at"] += int(chunk["ended_at"].isna().sum())
            diagnostics["missing_ride_id"] += int(chunk["ride_id"].isna().sum()) if "ride_id" in chunk else 0

            chunk["start_station_id"] = chunk["start_station_id"].map(normalize_station_id)
            chunk["end_station_id"] = chunk["end_station_id"].map(normalize_station_id)

            keep = (
                chunk["start_station_id"].isin(target_ids)
                | chunk["end_station_id"].isin(target_ids)
            )
            filtered = chunk.loc[keep].copy()

            if not filtered.empty:
                frames.append(filtered)

        diagnostics["files_loaded"].append(
            {
                "file": file_label,
                "rows": file_rows,
            }
        )
        diagnostics["rows_read"] += file_rows
        print(f"  {file_rows:,} rows")

    if not frames:
        return pd.DataFrame(columns=CITIBIKE_COLUMNS), diagnostics

    trips = pd.concat(frames, ignore_index=True)
    trips["started_at"] = pd.to_datetime(trips["started_at"], errors="coerce")
    trips["ended_at"] = pd.to_datetime(trips["ended_at"], errors="coerce")

    return trips, diagnostics


# -----------------------------------------------------------------------------
# FERRY EVENTS
# -----------------------------------------------------------------------------


def normalize_ferry_events(raw_events: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate schedule records without merging legitimate routes."""
    if raw_events.empty:
        return raw_events, 0

    events = raw_events.copy()

    subset_cols = [
        "route",
        "ferry_terminal_canon",
        "direction",
        "day_type",
        "scheduled_minutes",
        "event_type",
        "source_file",
    ]

    before = len(events)
    events = events.drop_duplicates(subset=subset_cols, keep="first")

    events["usable_for_first_mile"] = events["event_type"].isin(
        ["departure", "scheduled_stop"]
    )
    events["usable_for_last_mile"] = events["event_type"].isin(
        ["arrival", "scheduled_stop"]
    )

    return events, before - len(events)


def expand_ferry_events_to_dates(
    ferry_events: pd.DataFrame,
    relevant_dates: pd.Series,
) -> pd.DataFrame:
    """Expand recurring timetable events onto the actual Citi Bike calendar dates."""
    if ferry_events.empty or relevant_dates.empty:
        return pd.DataFrame()

    dates = pd.Series(pd.to_datetime(relevant_dates, errors="coerce").dropna().dt.normalize().unique())
    dates = pd.DatetimeIndex(dates).sort_values()

    date_frame = pd.DataFrame({"service_date": dates})
    date_frame["day_type"] = np.where(
        date_frame["service_date"].dt.weekday < 5,
        "weekday",
        "weekend",
    )

    events = ferry_events.merge(
        date_frame,
        on="day_type",
        how="inner",
    )

    events["ferry_datetime"] = events["service_date"] + pd.to_timedelta(
        events["scheduled_minutes"], unit="m"
    )

    return events


# -----------------------------------------------------------------------------
# MATCHING
# -----------------------------------------------------------------------------


def _grouped_asof_match(
    trips: pd.DataFrame,
    events: pd.DataFrame,
    trip_datetime_col: str,
    event_direction: str,
    buffer_minutes: int,
    trip_type: str,
) -> pd.DataFrame:
    """Match each trip to the nearest ferry event at its mapped terminal/day type."""
    if trips.empty or events.empty:
        return pd.DataFrame()

    results: list[pd.DataFrame] = []
    group_cols = ["ferry_terminal_canon", "day_type"]

    left = trips.copy()
    right = events.copy()

    right = right[
        [
            "ferry_terminal_canon",
            "day_type",
            "ferry_datetime",
            "scheduled_time",
            "route",
            "direction",
            "event_type",
            "source_file",
            "origin_terminal",
            "destination_terminal",
            "matched_schedule_terminal",
        ]
    ].copy()

    for (terminal_canon, day_type), left_group in left.groupby(group_cols, sort=False):
        right_group = right[
            (right["ferry_terminal_canon"] == terminal_canon)
            & (right["day_type"] == day_type)
        ].copy()

        if right_group.empty:
            continue

        left_group = left_group.sort_values(trip_datetime_col).copy()
        right_group = right_group.sort_values(
            ["ferry_datetime", "route", "direction"]
        ).copy()

        matched = pd.merge_asof(
            left_group,
            right_group,
            left_on=trip_datetime_col,
            right_on="ferry_datetime",
            direction=event_direction,
            allow_exact_matches=True,
        )

        if event_direction == "forward":
            matched["time_gap_minutes"] = (
                matched["ferry_datetime"] - matched[trip_datetime_col]
            ).dt.total_seconds() / 60.0
        else:
            matched["time_gap_minutes"] = (
                matched[trip_datetime_col] - matched["ferry_datetime"]
            ).dt.total_seconds() / 60.0

        matched = matched.loc[
            matched["time_gap_minutes"].between(0, buffer_minutes, inclusive="both")
        ].copy()

        if matched.empty:
            continue

        matched["trip_type"] = trip_type
        matched["matched_route"] = matched["route"]
        matched["matched_direction"] = matched["direction"]
        matched["ferry_time"] = matched["scheduled_time"]
        matched["matched_ferry_terminal"] = matched[
            "matched_schedule_terminal"
        ]

        results.append(matched)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def match_first_mile(
    trips: pd.DataFrame,
    matching_map: pd.DataFrame,
    ferry_events_dated: pd.DataFrame,
    buffer_minutes: int,
) -> pd.DataFrame:
    """First mile matching logic."""
    target_ids = set(matching_map["bike_station_id"])

    end_trips = trips.loc[
        trips["end_station_id"].isin(target_ids)
    ].copy()
    end_trips = end_trips.dropna(subset=["ended_at"])

    if end_trips.empty:
        return pd.DataFrame()

    end_trips = end_trips.merge(
        matching_map[
            [
                "bike_station_id",
                "bike_station_name",
                "ferry_terminal",
                "ferry_terminal_canon",
            ]
        ].drop_duplicates(),
        left_on="end_station_id",
        right_on="bike_station_id",
        how="inner",
    )

    end_trips["day_type"] = np.where(
        end_trips["ended_at"].dt.weekday < 5,
        "weekday",
        "weekend",
    )

    events = ferry_events_dated.loc[
        ferry_events_dated["usable_for_first_mile"]
    ].copy()

    return _grouped_asof_match(
        end_trips,
        events,
        trip_datetime_col="ended_at",
        event_direction="forward",
        buffer_minutes=buffer_minutes,
        trip_type="first_mile",
    )


def match_last_mile(
    trips: pd.DataFrame,
    matching_map: pd.DataFrame,
    ferry_events_dated: pd.DataFrame,
    buffer_minutes: int,
) -> pd.DataFrame:
    """Last mile matching logic."""
    target_ids = set(matching_map["bike_station_id"])

    start_trips = trips.loc[
        trips["start_station_id"].isin(target_ids)
    ].copy()
    start_trips = start_trips.dropna(subset=["started_at"])

    if start_trips.empty:
        return pd.DataFrame()

    start_trips = start_trips.merge(
        matching_map[
            [
                "bike_station_id",
                "bike_station_name",
                "ferry_terminal",
                "ferry_terminal_canon",
            ]
        ].drop_duplicates(),
        left_on="start_station_id",
        right_on="bike_station_id",
        how="inner",
    )

    start_trips["day_type"] = np.where(
        start_trips["started_at"].dt.weekday < 5,
        "weekday",
        "weekend",
    )

    events = ferry_events_dated.loc[
        ferry_events_dated["usable_for_last_mile"]
    ].copy()

    return _grouped_asof_match(
        start_trips,
        events,
        trip_datetime_col="started_at",
        event_direction="backward",
        buffer_minutes=buffer_minutes,
        trip_type="last_mile",
    )


# -----------------------------------------------------------------------------
# RESULT CLEANING / SUMMARIES
# -----------------------------------------------------------------------------


def deduplicate_rides(
    matched: pd.DataFrame,
    subset: list[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Keep the closest matched ferry event for each ride."""
    if matched.empty:
        return matched, 0

    subset = subset or ["ride_id", "trip_type"]

    before = len(matched)

    sort_cols = [c for c in [*subset, "time_gap_minutes", "matched_route", "matched_direction"] if c in matched.columns]
    matched = matched.sort_values(sort_cols, kind="stable")
    matched = matched.drop_duplicates(subset=subset, keep="first")

    return matched.reset_index(drop=True), before - len(matched)


def add_output_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Keep a clean, consistent output schema."""
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    existing = [c for c in OUTPUT_COLUMNS if c in df.columns]
    return df[existing].copy()


def build_station_summary(
    mapping: pd.DataFrame,
    first_mile: pd.DataFrame,
    last_mile: pd.DataFrame,
) -> pd.DataFrame:
    """Create one summary row per target bike station."""
    keys = [
        "bike_station_id",
        "bike_station_name",
        "ferry_terminal",
    ]

    base = mapping[keys].drop_duplicates()

    if first_mile.empty:
        first_counts = pd.DataFrame(columns=keys + ["first_mile_trips"])
    else:
        first_counts = (
            first_mile.groupby(keys)
            .size()
            .rename("first_mile_trips")
            .reset_index()
        )

    if last_mile.empty:
        last_counts = pd.DataFrame(columns=keys + ["last_mile_trips"])
    else:
        last_counts = (
            last_mile.groupby(keys)
            .size()
            .rename("last_mile_trips")
            .reset_index()
        )

    summary = base.merge(first_counts, how="left").merge(last_counts, how="left")
    summary["first_mile_trips"] = summary["first_mile_trips"].fillna(0).astype(int)
    summary["last_mile_trips"] = summary["last_mile_trips"].fillna(0).astype(int)
    summary["total_trips"] = (
        summary["first_mile_trips"] + summary["last_mile_trips"]
    )

    return summary.sort_values("bike_station_id").reset_index(drop=True)


def build_terminal_route_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize schedule events count per route at each terminal."""
    if events.empty:
        return pd.DataFrame()

    grouped = events.groupby(
        ["ferry_terminal", "ferry_terminal_canon", "route"],
        dropna=False,
    )

    summary = grouped.agg(
        number_of_schedule_events=("scheduled_minutes", "size"),
        weekday_events=("day_type", lambda s: int((s == "weekday").sum())),
        weekend_events=("day_type", lambda s: int((s == "weekend").sum())),
    ).reset_index()

    return summary.sort_values(
        ["ferry_terminal", "route"]
    ).reset_index(drop=True)


def build_bike_route_mapping(
    mapping: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Show all routes serving each bike station's nearest ferry terminal."""
    if events.empty:
        out = mapping[
            ["bike_station_id", "bike_station_name", "ferry_terminal"]
        ].drop_duplicates()
        out["all_routes_serving_that_terminal"] = ""
        return out

    route_lookup = (
        events.groupby("ferry_terminal_canon")["route"]
        .agg(lambda s: " | ".join(sorted(set(s.dropna()))))
        .rename("all_routes_serving_that_terminal")
        .reset_index()
    )

    out = mapping.merge(
        route_lookup,
        how="left",
        on="ferry_terminal_canon",
    )

    out = out[
        [
            "bike_station_id",
            "bike_station_name",
            "ferry_terminal",
            "all_routes_serving_that_terminal",
        ]
    ].drop_duplicates()

    return out.sort_values("bike_station_id").reset_index(drop=True)


def describe_gaps(series: pd.Series) -> dict:
    """Describe time gap metrics."""
    if series.empty:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
            "distribution": {},
        }

    return {
        "count": int(series.size),
        "min": float(series.min()),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "max": float(series.max()),
        "distribution": series.value_counts().sort_index().to_dict(),
    }


def write_validation_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def load_matching_context() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load station mapping, Citi Bike trips, and dated ferry events for matching."""
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

    print("Loading Citi Bike trip data...")
    trips, _ = load_citibike_data(TRIP_DATA_DIR, target_ids)

    relevant_dates = pd.concat(
        [trips["started_at"].dropna(), trips["ended_at"].dropna()], ignore_index=True
    )
    ferry_events_dated = expand_ferry_events_to_dates(ferry_events, relevant_dates)
    ferry_events_dated["ferry_terminal_canon"] = ferry_events_dated["ferry_terminal_canon"].astype(str)
    ferry_events_dated = ferry_events_dated.rename(
        columns={"ferry_terminal": "matched_schedule_terminal"}
    )
    return mapping_raw, trips, matching_map, ferry_events_dated


def match_integrated_trips(
    trips: pd.DataFrame,
    matching_map: pd.DataFrame,
    ferry_events_dated: pd.DataFrame,
    buffer_minutes: int,
) -> pd.DataFrame:
    """Return unique first- and last-mile matches for a transfer window."""
    first_raw = match_first_mile(trips, matching_map, ferry_events_dated, buffer_minutes)
    last_raw = match_last_mile(trips, matching_map, ferry_events_dated, buffer_minutes)
    first_mile, _ = deduplicate_rides(first_raw, subset=["ride_id", "trip_type"])
    last_mile, _ = deduplicate_rides(last_raw, subset=["ride_id", "trip_type"])
    combined = pd.concat(
        [add_output_fields(first_mile), add_output_fields(last_mile)],
        ignore_index=True,
    )
    if combined.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return (
        combined.sort_values(["ride_id", "trip_type", "time_gap_minutes"], kind="stable")
        .drop_duplicates(subset=["ride_id", "trip_type"], keep="first")
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    issues: list[str] = []

    print("=" * 72)
    print("INSPECTING INPUTS")
    print("=" * 72)

    # 1. BIKE/FERRY MAPPING
    mapping_raw, mapping_issues = load_bike_ferry_mapping(BIKE_MAP_PATH)
    issues.extend(mapping_issues)

    n_stations = mapping_raw["bike_station_id"].nunique()
    print(f"Unique target bike stations in bike.csv: {n_stations}")

    print("\nBike station -> nearest ferry terminal")
    print(
        mapping_raw[
            ["bike_station_id", "bike_station_name", "ferry_terminal"]
        ].to_string(index=False)
    )

    # 2. DISCOVER SCHEDULE FILES
    schedule_files = discover_ferry_schedule_files(FERRY_SCHEDULE_DIR)

    if schedule_files.empty:
        raise FileNotFoundError(
            f"No ferry timetable CSV files found under {FERRY_SCHEDULE_DIR}"
        )

    print("\nSchedule files discovered:")
    print(
        schedule_files[
            ["relative_path", "route", "day_type"]
        ].to_string(index=False)
    )

    extra_routes = schedule_files.attrs.get("extra_routes", [])
    if extra_routes:
        issues.append(
            "Additional/unrecognized schedule files or route folders: "
            + ", ".join(extra_routes)
        )

    unknown_day = schedule_files[schedule_files["day_type"].isna()]
    if not unknown_day.empty:
        issues.append(
            "Could not infer weekday/weekend from: "
            + ", ".join(unknown_day["relative_path"].tolist())
        )

    # 3. PARSE ALL FERRY SCHEDULES
    all_events: list[pd.DataFrame] = []
    parse_issues: list[str] = []

    for _, file_row in schedule_files.iterrows():
        events, file_issues = parse_timetable_file(
            Path(file_row["source_file"]),
            file_row["route"],
            file_row["day_type"],
        )
        parse_issues.extend(file_issues)
        if not events.empty:
            all_events.append(events)

    raw_events = (
        pd.concat(all_events, ignore_index=True)
        if all_events
        else pd.DataFrame()
    )

    ferry_events, n_duplicate_schedule = normalize_ferry_events(raw_events)

    if ferry_events.empty:
        raise RuntimeError("No ferry events could be parsed from the timetable CSVs.")

    # 4. MAP bike.csv TERMINALS TO SCHEDULE TERMINALS
    schedule_terminals = sorted(
        ferry_events["ferry_terminal"].dropna().unique().tolist()
    )

    alias_map, unmatched_terminals, alias_diagnostics = map_bike_terminals_to_schedule(
        mapping_raw["ferry_terminal"].unique(),
        schedule_terminals,
        minimum_score=0.80,
    )

    if unmatched_terminals:
        issues.append(
            "Ferry terminals in bike.csv with no sufficiently close schedule match: "
            + ", ".join(unmatched_terminals)
        )

    matching_map = build_matching_map(mapping_raw, alias_map)

    if matching_map.empty:
        raise RuntimeError(
            "No bike station could be mapped to a ferry schedule terminal."
        )

    print("\nBike station -> bike.csv ferry terminal -> schedule terminal aliases")
    for _, row in mapping_raw.iterrows():
        aliases = alias_map.get(row["ferry_terminal"], [])
        print(
            f"  {row['bike_station_id']} | "
            f"{row['bike_station_name']} -> "
            f"{row['ferry_terminal']} -> {aliases}"
        )

    # 5. LOAD CITI BIKE TRIPS
    print("\nLoading Citi Bike trips filtered to target stations...")
    target_ids = set(mapping_raw["bike_station_id"].dropna())

    trips, citibike_diag = load_citibike_data(
        TRIP_DATA_DIR,
        target_ids,
    )

    print(f"Target-station Citi Bike rows: {len(trips):,}")

    # 6. EXPAND FERRY EVENTS TO ACTUAL CITIBIKE DATES
    relevant_dates = pd.concat(
        [
            trips["started_at"].dropna(),
            trips["ended_at"].dropna(),
        ],
        ignore_index=True,
    )

    ferry_events_dated = expand_ferry_events_to_dates(
        ferry_events,
        relevant_dates,
    )

    ferry_events_dated["ferry_terminal_canon"] = ferry_events_dated[
        "ferry_terminal_canon"
    ].astype(str)

    ferry_events_dated = ferry_events_dated.rename(
        columns={"ferry_terminal": "matched_schedule_terminal"}
    )

    # 7. MATCH FIRST/ LAST MILE
    first_raw = match_first_mile(
        trips,
        matching_map,
        ferry_events_dated,
        BUFFER_MINUTES,
    )

    last_raw = match_last_mile(
        trips,
        matching_map,
        ferry_events_dated,
        BUFFER_MINUTES,
    )

    print(f"Raw first-mile matches before deduplication: {len(first_raw):,}")
    print(f"Raw last-mile matches before deduplication: {len(last_raw):,}")

    if first_raw.empty and last_raw.empty:
        print("WARNING: No Citi Bike trips matched a ferry event within the configured buffer.")

    first_mile, first_dups = deduplicate_rides(
        first_raw,
        subset=["ride_id", "trip_type"],
    )
    last_mile, last_dups = deduplicate_rides(
        last_raw,
        subset=["ride_id", "trip_type"],
    )

    first_mile = add_output_fields(first_mile)
    last_mile = add_output_fields(last_mile)

    combined = pd.concat(
        [first_mile, last_mile],
        ignore_index=True,
    )

    if combined.empty:
        combined = pd.DataFrame(columns=OUTPUT_COLUMNS)

    if not combined.empty and "ride_id" in combined.columns:
        combined = (
            combined.sort_values(
                ["ride_id", "trip_type", "time_gap_minutes"],
                kind="stable",
            )
            .drop_duplicates(
                subset=["ride_id", "trip_type"],
                keep="first",
            )
            .reset_index(drop=True)
        )

    unique_rides = (
        combined["ride_id"].nunique(dropna=True)
        if not combined.empty and "ride_id" in combined.columns
        else 0
    )

    # 8. SUMMARIES
    station_summary = build_station_summary(
        mapping_raw,
        first_mile,
        last_mile,
    )

    terminal_route_summary = build_terminal_route_summary(
        ferry_events,
    )

    bike_route_mapping = build_bike_route_mapping(
        matching_map,
        ferry_events,
    )

    ferry_events_out = ferry_events.copy()
    if not ferry_events_out.empty:
        ferry_events_out = ferry_events_out.sort_values(
            [
                "route",
                "day_type",
                "direction",
                "scheduled_minutes",
                "ferry_terminal",
            ]
        ).reset_index(drop=True)

    print("\nMatch summary")
    print("-" * 72)
    print(f"First-mile trips: {len(first_mile):,}")
    print(f"Last-mile trips:  {len(last_mile):,}")
    print(f"Combined trips:   {len(combined):,}")

    # 9. SAVE OUTPUTS
    first_path = OUTPUT_DIR / "first_mile_integrated_trips.csv"
    last_path = OUTPUT_DIR / "last_mile_integrated_trips.csv"
    all_path = OUTPUT_DIR / "integrated_trips_all.csv"
    station_path = OUTPUT_DIR / "integrated_trip_summary_by_station.csv"
    events_path = OUTPUT_DIR / "ferry_events.csv"
    terminal_path = OUTPUT_DIR / "ferry_terminal_route_summary.csv"
    bike_routes_path = OUTPUT_DIR / "bike_station_ferry_route_mapping.csv"

    first_mile.to_csv(first_path, index=False)
    last_mile.to_csv(last_path, index=False)
    combined.to_csv(all_path, index=False)
    station_summary.to_csv(station_path, index=False)
    ferry_events_out.to_csv(events_path, index=False)
    terminal_route_summary.to_csv(terminal_path, index=False)
    bike_route_mapping.to_csv(bike_routes_path, index=False)

    # 10. VALIDATION REPORT
    all_gaps = (
        pd.concat(
            [
                first_mile["time_gap_minutes"],
                last_mile["time_gap_minutes"],
            ],
            ignore_index=True,
        )
        if not first_mile.empty or not last_mile.empty
        else pd.Series(dtype="float64")
    )
    gap_stats = describe_gaps(all_gaps)

    present_start = (
        set(trips["start_station_id"].dropna()) if not trips.empty else set()
    )
    present_end = (
        set(trips["end_station_id"].dropna()) if not trips.empty else set()
    )
    present_ids = present_start | present_end
    missing_bike_in_trips = sorted(target_ids - present_ids)

    mapped_terminals = sorted(
        mapping_raw["ferry_terminal"].dropna().unique().tolist()
    )

    routes_by_mapped_terminal: dict[str, list[str]] = {}
    for bike_terminal, aliases in alias_map.items():
        alias_canons = {
            canonicalize_terminal_name(a)
            for a in aliases
        }
        routes = sorted(
            ferry_events.loc[
                ferry_events["ferry_terminal_canon"].isin(alias_canons),
                "route",
            ]
            .dropna()
            .unique()
            .tolist()
        )
        routes_by_mapped_terminal[bike_terminal] = routes

    events_by_terminal = (
        ferry_events.groupby("ferry_terminal").size().to_dict()
        if not ferry_events.empty
        else {}
    )

    events_by_route_terminal = (
        ferry_events.groupby(["route", "ferry_terminal"]).size().to_dict()
        if not ferry_events.empty
        else {}
    )

    first_by_station = (
        first_mile.groupby(
            ["bike_station_id", "bike_station_name"]
        )
        .size()
        .to_dict()
        if not first_mile.empty
        else {}
    )

    last_by_station = (
        last_mile.groupby(
            ["bike_station_id", "bike_station_name"]
        )
        .size()
        .to_dict()
        if not last_mile.empty
        else {}
    )

    report: list[str] = []
    report.append("=" * 72)
    report.append("VALIDATION REPORT - potentially integrated Citi Bike–NYC Ferry trips")
    report.append("Temporal association only; records are not claimed to be the same passenger.")
    report.append("=" * 72)
    report.append("")
    report.append(f"1. Unique bike stations in bike.csv: {n_stations}")
    report.append(f"2. Time buffer: {BUFFER_MINUTES} minutes")
    report.append("")

    report.append("3. Bike station -> nearest ferry terminal")
    for _, row in mapping_raw.iterrows():
        report.append(
            f"   {row['bike_station_id']} | {row['bike_station_name']} -> {row['ferry_terminal']}"
        )

    report.append("")
    report.append("4. Routes serving each mapped ferry terminal")
    for terminal in mapped_terminals:
        routes = routes_by_mapped_terminal.get(terminal, [])
        report.append(
            f"   {terminal}: {', '.join(routes) if routes else '(none)'}"
        )

    report.append("")
    report.append("5. Ferry events per schedule terminal")
    for terminal, count in sorted(events_by_terminal.items()):
        report.append(f"   {terminal}: {count}")

    report.append("")
    report.append("6. Ferry events per route and terminal")
    for (route, terminal), count in sorted(events_by_route_terminal.items()):
        report.append(f"   {route} | {terminal}: {count}")

    report.append("")
    report.append("7. First-mile matches per bike station")
    for key, count in sorted(first_by_station.items()):
        report.append(f"   {key[0]} {key[1]}: {count}")

    report.append("")
    report.append("8. Last-mile matches per bike station")
    for key, count in sorted(last_by_station.items()):
        report.append(f"   {key[0]} {key[1]}: {count}")

    report.append("")
    report.append(f"9. Total first-mile trips: {len(first_mile):,}")
    report.append(f"10. Total last-mile trips: {len(last_mile):,}")
    report.append(
        f"11. Total unique Citi Bike rides across first/last-mile outputs: {unique_rides:,}"
    )
    report.append(
        f"12. Duplicate first-mile matches removed: {first_dups:,}"
    )
    report.append(
        f"13. Duplicate last-mile matches removed: {last_dups:,}"
    )
    report.append("")

    report.append("14. Time-gap distribution (minutes)")
    for gap, count in gap_stats["distribution"].items():
        report.append(f"    {gap}: {count}")

    report.append(
        "15. Time-gap min/median/mean/max: "
        f"{gap_stats['min']} / {gap_stats['median']} / "
        f"{None if gap_stats['mean'] is None else round(gap_stats['mean'], 3)} / "
        f"{gap_stats['max']}"
    )

    report.append("")
    report.append("16. Matched trips by route")
    if not combined.empty:
        for route, count in combined["matched_route"].value_counts().sort_index().items():
            report.append(f"    {route}: {count}")

    report.append("17. Matched trips by ferry terminal (bike.csv terminal name)")
    if not combined.empty:
        for terminal, count in combined["ferry_terminal"].value_counts().sort_index().items():
            report.append(f"    {terminal}: {count}")

    report.append("18. Matched trips by day type")
    if not combined.empty and "day_type" in combined.columns:
        for day, count in combined["day_type"].value_counts().sort_index().items():
            report.append(f"    {day}: {count}")
    elif combined.empty:
        report.append("    no matched trips")
    else:
        report.append("    day_type column unavailable")

    report.append("")
    report.append("ISSUES AND LIMITATIONS")
    report.append("-" * 72)

    if missing_bike_in_trips:
        report.append("Bike stations in bike.csv absent from Citi Bike data:")
        for sid in missing_bike_in_trips:
            names = mapping_raw.loc[
                mapping_raw["bike_station_id"] == sid,
                "bike_station_name",
            ]
            name = names.iloc[0] if not names.empty else "(unknown)"
            report.append(f"    {sid} {name}")
    else:
        report.append("Bike stations in bike.csv absent from Citi Bike data: none")

    if unmatched_terminals:
        report.append(
            "Ferry terminals in bike.csv absent from schedules: "
            + ", ".join(unmatched_terminals)
        )
    else:
        report.append("Ferry terminals in bike.csv absent from schedules: none")

    all_routes = set(ferry_events["route"].dropna())
    used_routes = (
        set(combined["matched_route"].dropna())
        if not combined.empty
        else set()
    )
    unused_routes = sorted(all_routes - used_routes)
    report.append(
        "Routes present in schedules but unused in matches: "
        + (", ".join(unused_routes) if unused_routes else "none")
    )

    report.append(f"Malformed/invalid timetable values flagged: {len(parse_issues)}")
    for item in parse_issues[:50]:
        report.append(f"    {item}")
    if len(parse_issues) > 50:
        report.append(f"    ... {len(parse_issues) - 50} more")

    report.append(
        "Missing Citi Bike timestamps: "
        f"started_at={citibike_diag['missing_started_at']:,}, "
        f"ended_at={citibike_diag['missing_ended_at']:,}"
    )
    report.append(
        f"Missing Citi Bike ride_id values: {citibike_diag['missing_ride_id']:,}"
    )
    report.append(
        f"Possible duplicate ferry schedule records removed: {n_duplicate_schedule:,}"
    )

    for item in issues:
        report.append(f"Other: {item}")

    report.append("")
    report.append("OUTPUT FILES")
    for p in [
        first_path,
        last_path,
        all_path,
        station_path,
        events_path,
        terminal_path,
        bike_routes_path,
    ]:
        report.append(f"    {p}")

    write_validation_report(
        OUTPUT_DIR / "validation_report.txt",
        report,
    )


if __name__ == "__main__":
    main()