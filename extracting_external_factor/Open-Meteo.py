import sys
from pathlib import Path

import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

sys.path.append(str(Path(__file__).resolve().parents[1]))
import paths

# =====================================================================
# CONFIGURATION & CONTEXT SETTINGS
# =====================================================================
# study window: May 31, 2024 to August 31, 2024
start_date_str = "2024-05-31"
end_date_str = "2024-08-31"
output_csv_name = paths.WEATHER_CSV

print(f"Initializing Open-Meteo download wrapper for window: {start_date_str} to {end_date_str}")

# Setup standard cache and robust urllib3 retry rules
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retries = Retry(
    total=5, 
    backoff_factor=0.2, 
    status_forcelist=[500, 502, 503, 504]
)
cache_session.mount('https://', HTTPAdapter(max_retries=retries))
openmeteo = openmeteo_requests.Client(session=cache_session)

# Coordinates for Central Park, New York City
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 40.7829,
    "longitude": -73.9654,
    "start_date": start_date_str,
    "end_date": end_date_str,
    "daily": [
        "temperature_2m_max", 
        "precipitation_sum",
        "wind_speed_10m_max",        
        "relative_humidity_2m_mean"  
    ],
    "timezone": "America/New_York"
}

# Execute API Request
print("Querying historical atmospheric vectors...")
responses = openmeteo.weather_api(url, params=params)
response = responses[0]

# Process daily timelines
daily = response.Daily()
weather_dates = pd.date_range(
    start=pd.to_datetime(daily.Time(), unit="s", utc=True).tz_convert("America/New_York"),
    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True).tz_convert("America/New_York"),
    freq=pd.Timedelta(seconds=daily.Interval()),
    inclusive="left"
).date

# Build DataFrame
weather_df = pd.DataFrame({
    "date": weather_dates,
    "temp_max_c": daily.Variables(0).ValuesAsNumpy(),
    "precipitation_mm": daily.Variables(1).ValuesAsNumpy(),
    "max_wind_speed_kmh": daily.Variables(2).ValuesAsNumpy(),
    "mean_humidity_pct": daily.Variables(3).ValuesAsNumpy()
})

# Create the binary indicator threshold (>2.5 mm precipitation)
weather_df['is_raining'] = np.where(weather_df['precipitation_mm'] > 2.5, 1, 0)

# Save to local directory
weather_df.to_csv(output_csv_name, index=False)
print(f"Success! Weather file compiled and written to: {output_csv_name}")
print(weather_df.head())