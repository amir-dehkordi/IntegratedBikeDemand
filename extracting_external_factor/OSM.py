import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
import osmnx as ox
import numpy as np
import time

sys.path.append(str(Path(__file__).resolve().parents[1]))
import paths

# Set projection parameters matching your script
METRIC_CRS = "EPSG:32618"  # UTM Zone 18N (New York City)
BUFFER_DISTANCE = 500       # 500 meters

# Configure OSMnx settings
ox.settings.use_cache = True
ox.settings.log_console = False

# =====================================================
# 1. LOAD BIKE STATIONS
# =====================================================
print("Loading bike stations...")
bike = pd.read_csv(paths.BIKE_STATIONS_CSV)

# Create GeoDataFrame from station lat/long
bike_gdf = gpd.GeoDataFrame(
    bike,
    geometry=gpd.points_from_xy(bike["Long"], bike["Lat"]),
    crs="EPSG:4326"
)

# Initialize feature collectors
road_lengths = []
intersection_counts = []
bike_lane_lengths = []

# =====================================================
# 2. ITERATE OVER STATIONS AND QUERY OSM DATA
# =====================================================
print("Querying OpenStreetMap via OSMnx using point radius...")

for idx, row in bike_gdf.iterrows():
    station_id = row["Bike Station ID"]
    lat = row["Lat"]
    lon = row["Long"]
    
    print(f" Processing Station ID: {station_id}...")
    
    # Add a short delay to respect Overpass API rate limits and avoid timeouts
    time.sleep(2)
    
    try:
        # Querying by point + distance radius is significantly faster and uses less server overhead
        G = ox.graph_from_point((lat, lon), dist=BUFFER_DISTANCE, network_type='all', retain_all=True)
        
        # Project graph to the local metric UTM zone to get accurate meter measurements
        G_projected = ox.project_graph(G, to_crs=METRIC_CRS)
        
        # Convert graph elements to GeoDataFrames
        nodes, edges = ox.graph_to_gdfs(G_projected)
        
        # --- A. Measure Total Road Length ---
        total_road_len = edges['length'].sum()
        road_lengths.append(total_road_len)
        
        # --- B. Count Intersections ---
        intersections = nodes[nodes['street_count'] >= 3]
        intersection_counts.append(len(intersections))
        
        # --- C. Measure Bike Lane Infrastructure Length (Safely check missing keys) ---
        # Using .get() prevents KeyError if the column is completely missing in that buffer area
        cycleway = edges.get('cycleway', pd.Series(index=edges.index, dtype=str))
        cycleway_left = edges.get('cycleway:left', pd.Series(index=edges.index, dtype=str))
        cycleway_right = edges.get('cycleway:right', pd.Series(index=edges.index, dtype=str))
        highway = edges.get('highway', pd.Series(index=edges.index, dtype=str))
        bicycle = edges.get('bicycle', pd.Series(index=edges.index, dtype=str))

        bike_mask = (
            cycleway.notna() | 
            cycleway_left.notna() | 
            cycleway_right.notna() | 
            (highway == 'cycleway') |
            (bicycle.isin(['yes', 'designated']))
        )
        bike_edges = edges[bike_mask]
        
        bike_len = bike_edges['length'].sum() if len(bike_edges) > 0 else 0.0
        bike_lane_lengths.append(bike_len)
        
        print(f"      -> Success: {len(intersections)} intersections, {total_road_len:.1f}m roads, {bike_len:.1f}m bike lanes.")
        
    except Exception as e:
        print(f"      Warning for Station {station_id}: {e}")
        road_lengths.append(0.0)
        intersection_counts.append(0)
        bike_lane_lengths.append(0.0)

# =====================================================
# 3. MERGE METRICS BACK AND EXPORT
# =====================================================
bike_gdf = bike_gdf.to_crs(METRIC_CRS)
bike_gdf["road_length_m"] = road_lengths
bike_gdf["intersection_count"] = intersection_counts
bike_gdf["bike_lane_length_m"] = bike_lane_lengths

# Drop geometry to save as standard structural attributes dataframe
output_df = pd.DataFrame(bike_gdf.drop(columns='geometry'))

# Export explicitly to update your feature database
output_df.to_csv(paths.OSM_STATION_CSV, index=False)
print(f"\nSuccess! '{paths.OSM_STATION_CSV}' generated with actual values.")