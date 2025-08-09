import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import branca.colormap as cm
from typing import Dict, Tuple, List, Union, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from datetime import datetime
import numpy as np
from functools import lru_cache
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


# Configuration
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"

# Expanded default coverage across South Africa (coastal and inland)
DISTRICT_COORDS: Dict[str, Tuple[float, float]] = {
    # KwaZulu-Natal
    "eThekwini (Durban)": (-29.8587, 31.0218),
    "King Cetshwayo (Richards Bay)": (-28.7807, 32.0383),
    "Ugu (Port Shepstone)": (-30.7414, 30.4540),
    "iLembe (Ballito)": (-29.5389, 31.2140),
    "uThukela (Ladysmith)": (-28.5539, 29.7784),
    "uMgungundlovu (Pietermaritzburg)": (-29.6006, 30.3794),
    "Amajuba (Newcastle)": (-27.7577, 29.9318),
    "Uthungulu (Empangeni)": (-28.7489, 31.8933),
    "UMkhanyakude (Mtubatuba)": (-28.4176, 32.1822),
    "Zululand (Vryheid)": (-27.7695, 30.7916),

    # Western Cape
    "Cape Town": (-33.9249, 18.4241),
    "George": (-33.9630, 22.4617),
    "Mossel Bay": (-34.1830, 22.1460),
    "Hermanus": (-34.4187, 19.2345),
    "Saldanha": (-33.0117, 17.9440),
    "Knysna": (-34.0363, 23.0479),

    # Eastern Cape
    "Gqeberha (Port Elizabeth)": (-33.9608, 25.6022),
    "East London": (-33.0153, 27.9116),
    "Mthatha": (-31.5889, 28.7844),

    # Gauteng
    "Johannesburg": (-26.2041, 28.0473),
    "Pretoria": (-25.7479, 28.2293),

    # Free State
    "Bloemfontein": (-29.0852, 26.1596),

    # Limpopo
    "Polokwane": (-23.9045, 29.4689),

    # Mpumalanga
    "Mbombela (Nelspruit)": (-25.4658, 30.9853),

    # North West
    "Rustenburg": (-25.6676, 27.2421),

    # Northern Cape
    "Kimberley": (-28.7282, 24.7499),
}
FEATURE_COLS = ['lat', 'lon', 'temp_c', 'humidity', 'wind_kph', 'pressure_mb', 'precip_mm', 'cloud', 'wave_height']

# Setup Open-Meteo client
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Geocoder (rate-limited to be safe)
_geolocator = Nominatim(user_agent="crisis-connect")
_geocode = RateLimiter(_geolocator.geocode, min_delay_seconds=1)

def fetch_weather_and_marine_data(lat, lon, is_coastal=False):
    """Fetch weather and marine data for a given location."""
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m,pressure_msl,cloud_cover",
        "timezone": "auto"
    }
    marine_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wave_height",
        "timezone": "auto"
    }
    
    try:
        weather_response = openmeteo.weather_api(WEATHER_URL, params=weather_params)[0]
        weather_hourly = weather_response.Hourly()
        marine_response = openmeteo.weather_api(MARINE_URL, params=marine_params)[0] if is_coastal else None
        marine_hourly = marine_response.Hourly() if marine_response else None

        # Debug print for marine data
        if marine_hourly:
            print(f"Marine data for lat={lat}, lon={lon}:")
            for i in range(marine_hourly.VariablesLength()):
                var = marine_hourly.Variables(i)
                print(f"  Variable {i}: {var.Variable()} - Values: {var.ValuesAsNumpy()[-5:]}")
        else:
            print(f"No marine data for lat={lat}, lon={lon}")

        return weather_hourly, marine_hourly
    except Exception as e:
        print(f"⚠️ Error fetching data for lat={lat}, lon={lon}: {e}")
        return None, None

def extract_features(district, lat, lon, weather_hourly, marine_hourly=None):
    """Extract features from weather and marine data."""
    if not weather_hourly:
        return None
    
    try:
        temp_c = weather_hourly.Variables(0).ValuesAsNumpy()[-24:].mean()
        precip_mm = weather_hourly.Variables(1).ValuesAsNumpy()[-24:].sum()
        wind_kph = weather_hourly.Variables(2).ValuesAsNumpy()[-24:].max() * 3.6
        humidity = weather_hourly.Variables(3).ValuesAsNumpy()[-24:].mean()
        pressure_mb = weather_hourly.Variables(4).ValuesAsNumpy()[-24:].mean()
        cloud = weather_hourly.Variables(5).ValuesAsNumpy()[-24:].mean()
        wave_height = marine_hourly.Variables(0).ValuesAsNumpy()[-24:].mean() if marine_hourly else 0
    except Exception as e:
        print(f"⚠️ Error extracting features for {district}: {e}")
        return None

    # Stricter conditions for is_severe
    is_severe = int(
        (precip_mm > 25 and wind_kph > 50) or  # Severe rain + wind
        (wave_height > 2 and precip_mm > 15) or  # Coastal surge + rain
        (humidity > 90 and temp_c < 10 and precip_mm > 10)  # Extreme humidity + cold + rain
    )

    return {
        'location': district,
        'lat': lat,
        'lon': lon,
        'temp_c': temp_c,
        'humidity': humidity,
        'wind_kph': wind_kph,
        'pressure_mb': pressure_mb,
        'precip_mm': precip_mm,
        'cloud': cloud,
        'wave_height': wave_height,
        'is_severe': is_severe
    }

def collect_all_data(locations: Optional[Union[Dict[str, Tuple[float, float]], List[Union[str, Tuple[float, float], Dict[str, float]]]]] = None):
    """Collect weather and marine data for a set of locations.

    Accepts a variety of inputs for flexibility (names, coords, dicts). If not provided,
    a broad default set is used.
    """
    data: List[Dict[str, float]] = []
    # Normalize user-provided shapes (names, tuples, dicts)
    if isinstance(locations, dict) or isinstance(locations, list) or locations is None:
        target_locations: Dict[str, Tuple[float, float]] = (
            {str(k): (float(v[0]), float(v[1])) for k, v in locations.items()} if isinstance(locations, dict)
            else DISTRICT_COORDS if locations is None
            else {
                (str(item) if isinstance(item, str) else item.get('name', f'place_{idx}') if isinstance(item, dict) else f'place_{idx}'): (
                    float(item[0]), float(item[1])
                ) if isinstance(item, tuple) else (
                    float(item.get('lat')), float(item.get('lon'))
                ) if isinstance(item, dict) and item.get('lat') is not None and item.get('lon') is not None else None
                for idx, item in enumerate(locations)
            }
        )
        # Remove Nones possibly added above
        target_locations = {k: v for k, v in target_locations.items() if v is not None}
    else:
        target_locations = DISTRICT_COORDS

    # Approximate list of coastal places for marine data
    coastal_keywords = [
        "Durban", "Richards Bay", "Port Shepstone", "Ballito", "Cape Town",
        "George", "Mossel Bay", "Hermanus", "Saldanha", "Knysna",
        "Gqeberha", "Port Elizabeth", "East London"
    ]

    for district, (lat, lon) in target_locations.items():
        is_coastal = any(k.lower() in district.lower() for k in coastal_keywords)
        weather_hourly, marine_hourly = fetch_weather_and_marine_data(lat, lon, is_coastal=is_coastal)
        if weather_hourly:
            features = extract_features(district, lat, lon, weather_hourly, marine_hourly)
            if features:
                data.append(features)
                print(f"✅ Collected data for {district}: {features}")
        else:
            print(f"❌ Failed to fetch data for {district}")
    
    df = pd.DataFrame(data)
    
    # Add synthetic negative samples if needed
    if len(df) > 0 and df['is_severe'].nunique() < 2:
        print("⚠️ Only one class detected, adding synthetic negative samples")
        synthetic_data = []
        for district, (lat, lon) in DISTRICT_COORDS.items():
            synthetic_data.append({
                'location': f"{district}_synthetic",
                'lat': lat + 0.01,  # Slight offset
                'lon': lon + 0.01,
                'temp_c': 20.0,  # Normal conditions
                'humidity': 50.0,
                'wind_kph': 10.0,
                'pressure_mb': 1013.0,
                'precip_mm': 0.0,
                'cloud': 20.0,
                'wave_height': 0.0,
                'is_severe': 0
            })
        df_synthetic = pd.DataFrame(synthetic_data)
        df = pd.concat([df, df_synthetic], ignore_index=True)
    
    return df

def train_model(df):
    """Train Random Forest and Logistic Regression models."""
    print("Class distribution before training:\n", df['is_severe'].value_counts())
    if df['is_severe'].nunique() < 2:
        raise ValueError("Cannot train model: Data contains only one class")

    X = df[FEATURE_COLS]
    y = df['is_severe']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print("\n🌲 Random Forest Report:\n", classification_report(y_test, y_pred_rf, zero_division=0))

    cv_scores = cross_val_score(rf_model, X, y, cv=5)
    print(f"📊 Cross-validation scores: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    importances = rf_model.feature_importances_
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances, y=FEATURE_COLS)
    plt.title("Feature Importance in Flood Prediction")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    print("\n📉 Logistic Regression Report:\n", classification_report(y_test, y_pred_log, zero_division=0))

    joblib.dump(rf_model, '../rf_model.pkl')
    print("💾 Random Forest model saved as 'rf_model.pkl'")
    return rf_model

def generate_risk_scores(model, df):
    """Generate risk scores and categories."""
    X = df[FEATURE_COLS]
    df['risk_score'] = model.predict_proba(X)[:, 1] * 100
    df['risk_category'] = df['risk_score'].apply(lambda score: "High" if score > 80 else "Moderate" if score > 50 else "Low")
    return df

def visualize_data(df):
    """Visualize data distributions and correlations."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x='is_severe', data=df, palette='coolwarm')
    plt.title("Class Distribution: Severe vs Normal")
    plt.xlabel("Severe (1 = Yes, 0 = No)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    df[FEATURE_COLS].hist(bins=20, figsize=(14, 8), color='skyblue', edgecolor='black')
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[FEATURE_COLS + ['is_severe']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def visualize_map_with_scores(df):
    """Create Folium map with risk scores and wave heights."""
    map = folium.Map(location=[-30.5595, 22.9375], zoom_start=5)
    colormap = cm.LinearColormap(['green', 'orange', 'red'], vmin=0, vmax=100, caption="Flood Risk (%)")

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=8,
            popup=f"{row['location']}:\n{row['risk_category']} Risk ({row['risk_score']:.1f}%)\nWave Height: {row['wave_height']:.1f}m",
            color=colormap(row['risk_score']),
            fill=True,
            fill_color=colormap(row['risk_score']),
            fill_opacity=0.7
        ).add_to(map)

    colormap.add_to(map)
    map.save("weather_risk_map.html")
    print("🗺️ Map with risk scores saved to 'weather_risk_map.html'")

def main():
    """Main pipeline for Crisis Connect."""
    df = collect_all_data()
    if df.empty:
        print("❌ No data collected. Exiting.")
        return

    try:
        df_disasters = pd.read_excel("data_disaster.xlsx", sheet_name=0)
        weather_disasters = df_disasters[
            df_disasters['Disaster Type'].str.lower().isin([
                'flood', 'storm', 'drought', 'cold wave', 'heat wave'
            ])
        ].dropna(subset=['Latitude', 'Longitude'])

        historical_data = []
        for _, row in weather_disasters.iterrows():
            historical_data.append({
                'location': row.get('Location', 'Unknown'),
                'lat': row['Latitude'],
                'lon': row['Longitude'],
                'temp_c': None,
                'humidity': None,
                'wind_kph': None,
                'pressure_mb': None,
                'precip_mm': None,
                'cloud': None,
                'wave_height': None,
                'is_severe': 1
            })

        df_hist = pd.DataFrame(historical_data)
        df = pd.concat([df, df_hist], ignore_index=True)
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
        print("✅ Merged historical disaster data")

    except Exception as e:
        print(f"⚠️ Error loading disaster data: {e}")

    print("Class distribution:\n", df['is_severe'].value_counts())
    if df['is_severe'].nunique() < 2:
        print("❌ Not enough classes to train a classifier. Adding more negative samples.")
        synthetic_data = [{
            'location': 'synthetic_normal',
            'lat': -29.0,
            'lon': 30.0,
            'temp_c': 20.0,
            'humidity': 50.0,
            'wind_kph': 10.0,
            'pressure_mb': 1013.0,
            'precip_mm': 0.0,
            'cloud': 20.0,
            'wave_height': 0.0,
            'is_severe': 0
        }]
        df = pd.concat([df, pd.DataFrame(synthetic_data)], ignore_index=True)

    model = train_model(df)
    df = generate_risk_scores(model, df)
    df.to_csv("weather_data_scored.csv", index=False)
    print("💾 Data with risk scores saved to 'weather_data_scored.csv'")

    visualize_data(df)
    visualize_map_with_scores(df)

if __name__ == "__main__":
    main()