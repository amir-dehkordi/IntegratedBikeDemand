import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import paths

# =====================================================
# SETTINGS
# =====================================================

BUFFER_DISTANCE = 500
METRIC_CRS = "EPSG:32618"

# =====================================================
# 1. LOAD BIKE STATIONS
# =====================================================

bike = pd.read_csv(paths.BIKE_STATIONS_CSV)

print("Bike columns:")
print(bike.columns.tolist())

bike_gdf = gpd.GeoDataFrame(
    bike,
    geometry=gpd.points_from_xy(
        bike["Long"],
        bike["Lat"]
    ),
    crs="EPSG:4326"
)

bike_gdf = bike_gdf.to_crs(METRIC_CRS)

# dock capacity
bike_gdf["DockCap"] = (
    bike["capacity of bike stations(docks)"]
)

print("Bike stations loaded!")

# =====================================================
# 2. FERRY DISTANCE
# =====================================================

ferry_gdf = gpd.GeoDataFrame(
    bike,
    geometry=gpd.points_from_xy(
        bike["Long ferry terminal"],
        bike["Lat ferry terminal"]
    ),
    crs="EPSG:4326"
)

ferry_gdf = ferry_gdf.to_crs(METRIC_CRS)

bike_gdf["FerryDist"] = (
    bike_gdf.geometry.distance(
        ferry_gdf.geometry
    )
)

print("Ferry distance calculated!")

# =====================================================
# 3. CREATE 500m BUFFER
# =====================================================

buffers = bike_gdf.copy()

buffers["geometry"] = (
    buffers.buffer(BUFFER_DISTANCE)
)

print("500m buffers created!")

# =====================================================
# 4a. LOAD SMART LOCATION DATABASE (For Road Intersection Density & Built Environment)
# =====================================================

print("Loading SLD...")

sld = gpd.read_file(paths.SLD_GDB)

print("SLD loaded!")

print("SLD CRS:")
print(sld.crs)

# project
sld = sld.to_crs(METRIC_CRS)

# GEOID
sld["TRACT_GEOID"] = (
    sld["GEOID20"]
    .astype(str)
)

# keep only needed variables (D3B is Street Intersection Density)
sld = sld[
    [
        "TRACT_GEOID",
        "D1A",
        "D1C",
        "D2B_E5MIX",
        "D3B",  # <--- Road Intersection Density is here!
        "D4B025",
        "Pct_AO0",
        "geometry"
    ]
]

print("SLD variables selected!")

# =====================================================
# 4b. LOAD AND CALCULATE BIKE LANE DENSITY
# =====================================================

# =====================================================
# 4b. LOAD AND CALCULATE BIKE LANE DENSITY
# =====================================================
print("Calculating Bike Lane Density...")

try:
    # NOTE: paths.BIKE_LANES_SHP is not present in the current data tree.
    # Falls back to a zero-filled placeholder column below if missing.
    bike_lanes = gpd.read_file(paths.BIKE_LANES_SHP)
    bike_lanes = bike_lanes.to_crs(METRIC_CRS)

    # Spatial join/clip to find bike lanes within the 500m buffers
    lanes_in_buffers = gpd.overlay(buffers, bike_lanes, how="intersection")

    # Calculate length of lanes inside the intersections
    lanes_in_buffers["lane_len"] = lanes_in_buffers.geometry.length

    # Aggregate total bike lane length per station buffer
    lane_agg = lanes_in_buffers.groupby("Bike Station ID")["lane_len"].sum().reset_index()

    # Calculate buffer area (pi * r^2) for a 500m radius buffer (~785,398 sq meters)
    buffer_area = np.pi * (BUFFER_DISTANCE ** 2)

    # Density = meters of bike lane per square meter of buffer area
    lane_agg["BikeLaneDensity"] = lane_agg["lane_len"] / buffer_area
    lane_agg = lane_agg[["Bike Station ID", "BikeLaneDensity"]]

except Exception as e:
    # Catches pyogrio.errors.DataSourceError, FileNotFoundError, etc.
    print(f"\n[NOTE] Bike lane file not processed or found. Reason: {e}")
    print("Creating a placeholder 'BikeLaneDensity' column with 0s for now.\n")
    lane_agg = pd.DataFrame({
        "Bike Station ID": bike_gdf["Bike Station ID"], 
        "BikeLaneDensity": 0.0
    })
# =====================================================
# 5. INTERSECT (SLD & Buffers)
# =====================================================

print("Overlaying buffers with SLD polygons...")

intersections = gpd.overlay(
    buffers,
    sld,
    how="intersection"
)

print(
    "Intersections:",
    len(intersections)
)

# =====================================================
# 6. AREA WEIGHTS
# =====================================================

intersections["intersect_area"] = (
    intersections.geometry.area
)

tract_area_lookup = (
    sld.set_index(
        "TRACT_GEOID"
    )
    .geometry.area
)

intersections["tract_area"] = (
    intersections["TRACT_GEOID"]
    .map(
        tract_area_lookup
    )
)

intersections["weight"] = (
    intersections["intersect_area"]
    /
    intersections["tract_area"]
)

# =====================================================
# 7. WEIGHTED VARIABLES
# =====================================================

variables = [
    "D1A",
    "D1C",
    "D2B_E5MIX",
    "D3B",
    "D4B025",
    "Pct_AO0"
]

for var in variables:

    intersections[var] = pd.to_numeric(
        intersections[var],
        errors="coerce"
    )

    intersections[
        f"{var}_weighted"
    ] = (
        intersections[var]
        *
        intersections["weight"]
    )

print("Weighting complete!")

# =====================================================
# 8. AGGREGATE TO STATION
# =====================================================

weighted_cols = [
    f"{v}_weighted"
    for v in variables
]

agg = (
    intersections.groupby(
        "Bike Station ID"
    )[weighted_cols]
    .sum()
    .reset_index()
)

agg.columns = [
    c.replace(
        "_weighted",
        ""
    )
    for c in agg.columns
]

print("Aggregation complete!")

# =====================================================
# 9. MERGE BACK TO STATIONS
# =====================================================

# Merge SLD aggregate metrics
bike_gdf = bike_gdf.merge(
    agg,
    on="Bike Station ID",
    how="left"
)

# Merge Bike Lane Density metrics
bike_gdf = bike_gdf.merge(
    lane_agg,
    on="Bike Station ID",
    how="left"
).fillna({"BikeLaneDensity": 0}) # fill stations with no bike lanes as 0

print("Merged back to stations!")

# =====================================================
# 10. FINAL DATASET
# =====================================================

final_cols = [
    "Bike Station ID",
    "name bike station",

    "DockCap",
    "FerryDist",

    "D1A",          # population density
    "D1C",          # employment density
    "D2B_E5MIX",    # land-use mix
    "D3B",          # intersection density (Road intersection density)
    "D4B025",       # transit accessibility
    "Pct_AO0",      # zero-car households
    "BikeLaneDensity" # Bike lane density (meters/sq meter)
]

final = bike_gdf[
    final_cols
].copy()

# rename for paper
final = final.rename(
    columns={
        "D1A": "PopDensity",
        "D1C": "EmpDensity",
        "D2B_E5MIX": "LandMix",
        "D3B": "IntDensity",  # Road Intersection Density - kept as IntDensity to match model.py's feature_cols
        "D4B025": "TransitAccess",
        "Pct_AO0": "ZeroVehicle"
    }
)

# =====================================================
# 11. EXPORT
# =====================================================

final.to_csv(
    paths.SLD_STATION_CSV,
    index=False
)

print("\n===================================")
print("DONE")
print("===================================")

print(final.head())

print(
    "\nSaved as:"
)

print(paths.SLD_STATION_CSV)