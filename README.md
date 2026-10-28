# IntegratedBikeDemand

Code for the manuscript **"Exploring the Determinants of Docked Bike-Share Demand in Connection with Ferry Transit Integration"** (New York City, summer 2024).

The pipeline builds a station × day × period × trip-stage demand panel from integrated Citi Bike–NYC Ferry trips, benchmarks XGBoost against OLS, a Poisson GLM and LightGBM under a temporal holdout, and interprets the fitted models with SHAP.

## Repository structure

```
.
├── paths.py                         # all input/output locations (edit this first)
├── extracting_external_factor/
│   ├── ACS_LEHD.py                  # ACS and LEHD features per station
│   ├── SLD.py                       # Smart Location Database features
│   ├── OSM.py                       # street-network features (OSMnx)
│   ├── landuse.py                   # MapPLUTO land-use mix in a 500 m buffer
│   └── Open-Meteo.py                # daily weather for Central Park
├── integrated_trips/
│   └── all_integrated_trips_16_ferry_terminal.csv
├── model_inputs/                    # station-level features used by the models
├── scripts/
│   ├── Model_comparison.py          # model benchmark (Table 2)
│   ├── shap_publication_plots.py    # SHAP importance and dependence plots
│   └── window_shap_sensitivity.py   # transfer-window sensitivity
├── shap_pub_plots/                  # figures written by shap_publication_plots.py
├── requirements.txt
└── LICENSE
```

Place `Bike_Stations.csv` in `extracting_external_factor/` if you need to re-run the feature-extraction scripts. Raw GIS extracts (ACS tables, LEHD LODES, MapPLUTO, SLD geodatabase, census tract shapefile) belong under `extracting_external_factor/data/` and are not tracked.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.13. Paths are resolved from `paths.py`, so scripts can be run from any working directory.

## Data

This repository includes the processed integrated-trip file and the station-level model inputs. Raw Citi Bike dumps, GTFS, ACS, LEHD, MapPLUTO and the Smart Location Database are not included (size and source terms). All of those sources are public:

| Data | Source |
|---|---|
| Citi Bike trip records, station information | [Citi Bike System Data](https://citibikenyc.com/system-data) |
| NYC Ferry schedule (GTFS) | NYC Ferry GTFS feed |
| Smart Location Database v3 | [U.S. EPA](https://www.epa.gov/smartgrowth/smart-location-mapping) |
| ACS 5-year estimates | U.S. Census Bureau |
| LEHD LODES (Workplace Area Characteristics) | U.S. Census Bureau |
| MapPLUTO land use | NYC Department of City Planning |
| Road network | OpenStreetMap via OSMnx |
| Daily weather | [Open-Meteo archive API](https://open-meteo.com/) (Central Park) |

The trip-matching scripts that build `integrated_trips/all_integrated_trips_16_ferry_terminal.csv` from raw Citi Bike and GTFS files are not in this snapshot. Modeling and SHAP scripts start from that CSV plus `model_inputs/`.

## Reproducing the results

1. Confirm `paths.py` points at your local folders (defaults match this layout).
2. Optional: re-build station features with the scripts in `extracting_external_factor/` (requires `Bike_Stations.csv` and the raw files under `extracting_external_factor/data/`).
3. `python scripts/Model_comparison.py` — temporal holdout benchmark.
4. `python scripts/shap_publication_plots.py` — SHAP figures (written to `shap_pub_plots/`).
5. `python scripts/window_shap_sensitivity.py` — transfer-window sensitivity (needs the extra window-specific trip files if you use that analysis).

## Key settings

- Study period: June–August 2024 (92 days); test set: the last 14 days (18–31 August).
- Panel: 16 stations × 92 days × 3 periods (morning peak 07:00–10:00, evening peak 16:00–19:00, off-peak) × 2 trip stages = 8,832 observations.
- XGBoost (tuned): 300 trees, maximum depth 8, learning rate 0.03, subsample 0.7, column subsample 0.6, L1 = 0.1, L2 = 10.
- Station-level SLD values are area-weighted from the block groups overlapping the 500 m catchment (see `extracting_external_factor/SLD.py`).

## Citation

If you use this code, please cite the manuscript (reference to be added after publication).

## License

MIT (see `LICENSE`).
