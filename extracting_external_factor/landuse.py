import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import paths

# ==========================================================
# 1. FILE PATHS
# ==========================================================
station_path = paths.BIKE_STATIONS_CSV
pluto_path = paths.LANDUSE_SHP

# ==========================================================
# 2. LOAD DATA & CREATE GEODATAFRAME
# ==========================================================
# Load the CSV file
bike_df = pd.read_csv(station_path)

# Convert coordinates to geometry points (Assuming WGS84 / EPSG:4326)
stations = gpd.GeoDataFrame(
    bike_df,
    geometry=gpd.points_from_xy(bike_df["Long"], bike_df["Lat"]),
    crs="EPSG:4326"
)

# Load PLUTO shapefile
pluto = gpd.read_file(pluto_path)

# ==========================================================
# 3. PROJECT TO METRIC CRS (METERS)
# UTM Zone 18N for NYC
# ==========================================================
metric_crs = "EPSG:32618"

stations = stations.to_crs(metric_crs)
pluto = pluto.to_crs(metric_crs)

# ==========================================================
# 4. DEFINE COLUMNS (Updated for bike.csv schema)
# ==========================================================
station_id_col = "Bike Station ID"
station_name_col = "name bike station"

# ==========================================================
# 5. CREATE 500 m BUFFER
# ==========================================================
stations_buffer = stations.copy()

stations_buffer["geometry"] = (
    stations_buffer.buffer(500)
)

stations_buffer["buffer_area"] = (
    stations_buffer.geometry.area
)

# ==========================================================
# 6. CLEAN LAND USE FIELD
# ==========================================================
pluto["LandUse"] = (
    pluto["LandUse"]
    .astype(str)
    .str.zfill(2)
)

# ==========================================================
# 7. RECLASSIFY LAND USE
# ==========================================================
landuse_mapping = {
    "01": "residential",
    "02": "residential",
    "03": "residential",
    "04": "residential",
    "05": "commercial",
    "06": "industrial",
    "07": "industrial",
    "08": "institutional",
    "09": "open_space",
    "10": "transportation",
    "11": "recreation"
}

pluto["lu_class"] = (
    pluto["LandUse"]
    .map(landuse_mapping)
)

# Remove missing land-use classes
pluto = pluto.dropna(
    subset=["lu_class"]
)

# ==========================================================
# 8. SPATIAL INTERSECTION
# ==========================================================
intersection = gpd.overlay(
    stations_buffer,
    pluto,
    how="intersection"
)

# Area of overlap
intersection["land_area"] = (
    intersection.geometry.area
)

# ==========================================================
# 9. SUM AREA BY LAND USE
# ==========================================================
summary = (
    intersection.groupby(
        [station_id_col, "lu_class"]
    )["land_area"]
    .sum()
    .unstack(fill_value=0)
    .reset_index()  # BUGFIX: station_id_col was left as the index after
                     # unstack(), so merge(on=station_id_col) below would
                     # raise a KeyError since it needs a real column.
)

# ==========================================================
# 10. MERGE BACK TO STATIONS
# ==========================================================
stations_buffer = (
    stations_buffer.merge(
        summary,
        on=station_id_col,
        how="left"
    )
)

stations_buffer = (
    stations_buffer.fillna(0)
)

# ==========================================================
# 11. CALCULATE LAND-USE PERCENTAGES
# ==========================================================
land_classes = [
    "residential",
    "commercial",
    "industrial",
    "institutional",
    "open_space",
    "transportation",
    "recreation"
]

# Add missing columns if absent
for cls in land_classes:
    
    if cls not in stations_buffer.columns:
        stations_buffer[cls] = 0

    stations_buffer[f"{cls}_pct"] = (
        stations_buffer[cls]
        / stations_buffer["buffer_area"]
    ) * 100

# ==========================================================
# 12. CALCULATE LAND-USE ENTROPY
# ==========================================================
proportion_df = (
    stations_buffer[
        [f"{cls}_pct"
         for cls in land_classes]
    ] / 100
)

stations_buffer[
    "landuse_entropy"
] = -(
    proportion_df *
    np.log(
        proportion_df.replace(
            0,
            np.nan
        )
    )
).sum(axis=1)

stations_buffer[
    "landuse_entropy"
] = (
    stations_buffer[
        "landuse_entropy"
    ] / np.log(
        len(land_classes)
    )
)

stations_buffer[
    "landuse_entropy"
] = (
    stations_buffer[
        "landuse_entropy"
    ].fillna(0)
)

# ==========================================================
# 13. FINAL OUTPUT TABLE
# ==========================================================
output_columns = [
    station_id_col,
    station_name_col,
    "landuse_entropy"
]

output_columns += [
    f"{cls}_pct"
    for cls in land_classes
]

result = (
    stations_buffer[
        output_columns
    ]
)

# ==========================================================
# 14. SAVE CSV
# ==========================================================
output_file = paths.LANDUSE_500M_CSV

result.to_csv(
    output_file,
    index=False
)

print(result.head())

print("\nDone!")
print(
    f"Saved as: {output_file}"
)