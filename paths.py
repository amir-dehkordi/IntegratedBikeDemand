"""
Central path configuration for the bike-share demand project.

Every script in the repo imports from this module instead of using
hardcoded relative filenames. This means:
  - Scripts can be run from any working directory (paths are absolute,
    resolved from this file's location).
  - If you move/rename a data folder, you only update it here once.

Project layout (relative to this file):

final model/
├── paths.py
├── extracting_external_factor/
├── integrated_trips/
├── model_inputs/
├── scripts/
└── shap_pub_plots/
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------
# extracting_external_factor/ and its data/ subtree
# ---------------------------------------------------------------------
EXTRACT_DIR = ROOT / "extracting_external_factor"
DATA_DIR = EXTRACT_DIR / "data"

BIKE_STATIONS_CSV = EXTRACT_DIR / "Bike_Stations.csv"   # used by ACS_LEHD.py

ACS_DIR = DATA_DIR / "acs data"
ACS_B01001 = ACS_DIR / "ACSDT5Y2024.B01001-Data.csv"
ACS_B15003 = ACS_DIR / "ACSDT5Y2024.B15003-Data.csv"
ACS_B25044 = ACS_DIR / "ACSDT5Y2024.B25044-Data.csv"    # NOTE: only the .zip currently exists here, extract it first
ACS_B19013 = ACS_DIR / "ACSDT5Y2024.B19013-Data.csv"
ACS_S1701 = ACS_DIR / "ACSST5Y2024.S1701-Data.csv"      # NOTE: not present in tree yet, needs to be downloaded

CENSUS_TRACTS_SHP = DATA_DIR / "census_tracts_shapefile" / "census_tracts.shp"

LANDUSE_SHP = DATA_DIR / "landuse" / "MapPLUTO.shp"

LEHD_JOBS_CSV = DATA_DIR / "LEHD LODES" / "ny_wac_S000_JT00_2023.csv"

SLD_DIR = DATA_DIR / "SmartLocationDatabaseV3"
SLD_GDB = SLD_DIR / "SmartLocationDatabase.gdb"
# NOTE: SmartLocationDatabaseV3/bike.csv and extracting_external_factor/Bike_Stations.csv
# are the same station data (just renamed at different points). Standardizing on
# BIKE_STATIONS_CSV everywhere below avoids the two copies drifting out of sync.
BIKE_LANES_SHP = SLD_DIR / "bike_lanes.shp"             # NOTE: not present in tree; SLD.py falls back to zeros if missing

# ---------------------------------------------------------------------
# integrated_trips/
# ---------------------------------------------------------------------
TRIPS_DIR = ROOT / "integrated_trips"
TRIPS_CSV = TRIPS_DIR / "all_integrated_trips_16_ferry_terminal.csv"

# ---------------------------------------------------------------------
# model_inputs/ - shared handoff folder between extraction and modeling
# scripts. Each extraction script WRITES its output csv here; each
# modeling script READS its input csvs from here.
# ---------------------------------------------------------------------
MODEL_INPUTS_DIR = ROOT / "model_inputs"
MODEL_INPUTS_DIR.mkdir(exist_ok=True)

WEATHER_CSV = MODEL_INPUTS_DIR / "nyc_summer_weather_2024.csv"                  # written by Open-Meteo.py
SLD_STATION_CSV = MODEL_INPUTS_DIR / "station_model_dataset.csv"                # written by SLD.py
OSM_STATION_CSV = MODEL_INPUTS_DIR / "station_osm_network_features.csv"        # written by OSM.py
ACS_STATION_CSV = MODEL_INPUTS_DIR / "station_final_model_dataset.csv"         # written by ACS_LEHD.py
LANDUSE_500M_CSV = MODEL_INPUTS_DIR / "bike_station_landuse_500m.csv"          # written by landuse.py

# ---------------------------------------------------------------------
# scripts/ and shap_pub_plots/
# ---------------------------------------------------------------------
SCRIPTS_DIR = ROOT / "scripts"
SHAP_PLOTS_DIR = ROOT / "shap_pub_plots"
SHAP_PLOTS_DIR.mkdir(exist_ok=True)

MODEL_COMPARISON_RESULTS_CSV = SCRIPTS_DIR / "model_comparison_results.csv"
