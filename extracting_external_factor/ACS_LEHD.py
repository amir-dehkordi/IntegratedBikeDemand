import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.stats import entropy

# Make the project-root paths.py importable regardless of cwd
sys.path.append(str(Path(__file__).resolve().parents[1]))
import paths

# =====================================================
# PARAMETERS
# =====================================================

BUFFER_DISTANCE = 500
METRIC_CRS = "EPSG:32618"

# =====================================================
# 1. LOAD BIKE STATIONS
# =====================================================

bike = pd.read_csv(paths.BIKE_STATIONS_CSV)

bike_gdf = gpd.GeoDataFrame(
    bike,
    geometry=gpd.points_from_xy(
        bike["Long"],
        bike["Lat"]
    ),
    crs="EPSG:4326"
)

bike_gdf = bike_gdf.to_crs(METRIC_CRS)

bike_gdf["dock_capacity"] = (
    bike[
        "capacity of bike stations(docks)"
    ]
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

bike_gdf["ferry_distance_m"] = (
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

# Calculate buffer area in square miles for intersection density calculations
# 500m buffer area ≈ 785,398 m² ≈ 0.303 square miles
buffers["buffer_area_sqmi"] = buffers.geometry.area / 2589988.11

# =====================================================
# 4. LOAD ACS FILES (EXPANDED FOR ECONOMICS)
# =====================================================

def load_acs(filepath):
    df = pd.read_csv(filepath)
    df = df.iloc[1:].copy()
    df["TRACT_GEOID"] = (
        df["GEO_ID"]
        .str[-11:]
    )
    return df

print("Loading ACS tables...")
# NOTE: paths.ACS_B25044 needs the .zip in data/acs data/ extracted to a -Data.csv
# NOTE: paths.ACS_S1701 is not present in data/acs data/ yet - download it from
#       data.census.gov (table S1701) and place it in that folder before running.
b01001 = load_acs(paths.ACS_B01001)
b15003 = load_acs(paths.ACS_B15003)
b25044 = load_acs(paths.ACS_B25044)
b19013 = load_acs(paths.ACS_B19013)
s1701 = load_acs(paths.ACS_S1701)

# =====================================================
# 5. CREATE ACS VARIABLES
# =====================================================

acs = pd.DataFrame()
acs["TRACT_GEOID"] = b01001["TRACT_GEOID"]

acs["population"] = pd.to_numeric(b01001["B01001_001E"], errors="coerce")

age18_24_cols = [
    "B01001_007E", "B01001_008E", "B01001_009E", "B01001_010E",  
    "B01001_031E", "B01001_032E", "B01001_033E", "B01001_034E"   
]
acs["age18_24"] = b01001[age18_24_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
acs["age18_24_pct"] = np.where(acs["population"] > 0, (acs["age18_24"] / acs["population"]) * 100, 0)

age24_30_cols = [
    "B01001_010E", "B01001_011E", "B01001_012E",
    "B01001_034E", "B01001_035E", "B01001_036E"
]
acs["age24_30"] = b01001[age24_30_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
acs["age24_30_pct"] = np.where(acs["population"] > 0, (acs["age24_30"] / acs["population"]) * 100, 0)

edu_cols = ["B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"]
acs["higher_edu"] = b15003[edu_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
acs["higher_edu_pct"] = np.where(acs["population"] > 0, (acs["higher_edu"] / acs["population"]) * 100, 0)

acs["zero_vehicle"] = (
    pd.to_numeric(b25044["B25044_003E"], errors="coerce") +
    pd.to_numeric(b25044["B25044_010E"], errors="coerce")
)
acs["zero_vehicle_pct"] = np.where(acs["population"] > 0, (acs["zero_vehicle"] / acs["population"]) * 100, 0)

acs["median_income"] = pd.to_numeric(b19013["B19013_001E"], errors="coerce")
acs["poverty_pct"] = pd.to_numeric(s1701["S1701_C03_001E"], errors="coerce")

print("ACS variables created!")

# =====================================================
# 6. LOAD TRACTS
# =====================================================

tracts = gpd.read_file(paths.CENSUS_TRACTS_SHP)
tracts = tracts.to_crs(METRIC_CRS)
tracts["TRACT_GEOID"] = tracts["FIPS"].astype(str)

nyc_counties = ["36005", "36047", "36061", "36081", "36085"]
tracts = tracts[tracts["TRACT_GEOID"].str[:5].isin(nyc_counties)]
print("NYC tracts filtered!")

# =====================================================
# 7. MERGE ACS
# =====================================================

tracts = tracts.merge(acs, on="TRACT_GEOID", how="left")
tracts["land_area_km2"] = tracts.geometry.area / 1_000_000
tracts["population_density"] = tracts["population"] / tracts["land_area_km2"]

print("ACS merged!")

# =====================================================
# 8. LEHD JOBS   
# =====================================================

jobs = pd.read_csv(paths.LEHD_JOBS_CSV)
jobs["TRACT_GEOID"] = jobs["w_geocode"].astype(str).str[:11]

jobs_tract = jobs.groupby("TRACT_GEOID")["C000"].sum().reset_index()
jobs_tract.columns = ["TRACT_GEOID", "job_density"]

tracts = tracts.merge(jobs_tract, on="TRACT_GEOID", how="left")

print("LEHD jobs merged!")

# =====================================================
# 9. WEIGHTED VARIABLES
# =====================================================

intersections = gpd.overlay(buffers, tracts, how="intersection")
intersections["intersect_area"] = intersections.geometry.area

# BUGFIX: The weight must reflect each tract's SHARE OF THE BUFFER, not the
# fraction of the tract's own area that happens to fall inside the buffer.
# Dividing by tract_area causes weights to NOT sum to 1 across the tracts
# intersecting a given station's buffer -- if a buffer fully contains several
# small tracts (common near dense areas with tiny commercial-heavy tracts),
# each contributes a weight near 1.0, so the weighted "sum" can exceed the
# true blended value by 2x-3x or more. This is what produced the impossible
# $520,000+ median_income outlier.
#
# Fix: normalize weight per station so the weights of all tracts touching
# that station's buffer sum to exactly 1.0, based on each tract's share of
# the TOTAL intersected area for that station (not the tract's own area).
station_total_intersect = (
    intersections.groupby("Bike Station ID")["intersect_area"].sum()
)
intersections["station_total_intersect_area"] = intersections["Bike Station ID"].map(
    station_total_intersect
)
intersections["weight"] = (
    intersections["intersect_area"] / intersections["station_total_intersect_area"]
)

variables = [
    "population_density",
    "job_density",
    "higher_edu_pct",
    "zero_vehicle_pct",
    "median_income",       
    "poverty_pct",
    "age18_24_pct",
    "age24_30_pct",
]


for var in variables:
    intersections[f"{var}_weighted"] = intersections[var] * intersections["weight"]

agg = (
    intersections.groupby("Bike Station ID")[
        [f"{v}_weighted" for v in variables]
    ]
    .sum()
    .reset_index()
)

agg.columns = [c.replace("_weighted", "") for c in agg.columns]
bike_gdf = bike_gdf.merge(agg, on="Bike Station ID", how="left")

print("Weighted ACS aggregations complete!")

# =====================================================
# 10. LAND USE ENTROPY
# =====================================================

pluto = gpd.read_file(paths.LANDUSE_SHP)
pluto = pluto.to_crs(METRIC_CRS)

entropy_values = []

for _, row in buffers.iterrows():
    clip = pluto[pluto.intersects(row.geometry)].copy()
    if len(clip) == 0:
        entropy_values.append(np.nan)
        continue

    clip["area"] = clip.geometry.area
    landuse_area = clip.groupby("LandUse")["area"].sum()
    proportions = landuse_area / landuse_area.sum()
    ent = entropy(proportions)

    if len(proportions) > 1:
        ent = ent / np.log(len(proportions))

    entropy_values.append(ent)

bike_gdf["landuse_entropy"] = entropy_values
print("Land-use entropy done!")

# =====================================================
# 11. EXPORT
# =====================================================

final_cols = [
    "Bike Station ID",
    "name bike station",
    "dock_capacity",
    "ferry_distance_m",
    "population_density",
    "job_density",
    "age18_24_pct",
    "age24_30_pct",
    "higher_edu_pct",
    "zero_vehicle_pct",
    "median_income",
    "poverty_pct",
    "landuse_entropy"
]

final = bike_gdf[final_cols].copy()
final.to_csv(paths.ACS_STATION_CSV, index=False)

print("\nDONE!")
print(final.head())
print(f"\n{paths.ACS_STATION_CSV} updated!.")