from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import pandas as pd
import numpy as np
import joblib
from typing import List, Optional
from services.predict import collect_all_data, generate_risk_scores, fetch_weather_and_marine_data, extract_features
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import os
import logging
from models.model import AlertModel, LocationRequest
from collections import defaultdict
from datetime import datetime
from utils.db import get_mongo_client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB setup
MONGODB_URL = os.getenv("MONGO_CONNECTION_STRING", "mongodb+srv://utshuma6:zpyBvSV2LbdMbvhU@cluster0.fyrnhd0.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
client = AsyncIOMotorClient(MONGODB_URL)
db = client.crisis_connect

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-deployed-app-url"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize historical data from XLSX if collection is empty
HISTORICAL_XLSX = "data_disaster.xlsx"

async def initialize_historical_data():
    count = await db.historicaldata.count_documents({})
    if count == 0 and os.path.exists(HISTORICAL_XLSX):
        try:
            df = pd.read_excel(HISTORICAL_XLSX)
            df.columns = df.columns.str.strip().str.lower()
            logger.info(f"📄 Excel columns: {df.columns.tolist()}")

            # Rename for easier reference
            df.rename(columns={
                'location': 'location',
                'latitude': 'latitude',
                'longitude': 'longitude',
                'total deaths': 'total_deaths'
            }, inplace=True)

            # Define severity from total deaths
            def classify_severity(row):
                deaths = row.get('total_deaths') or 0
                try:
                    deaths = int(deaths)
                except:
                    deaths = 0
                if deaths > 100:
                    return 'High'
                elif deaths > 10:
                    return 'Medium'
                elif deaths > 0:
                    return 'Low'
                return 'Unknown'

            df['severity'] = df.apply(classify_severity, axis=1)

            df = df.fillna({'location': 'Unknown', 'severity': 'Unknown', 'latitude': np.nan, 'longitude': np.nan})

            records = [
                {
                    "location": record["location"],
                    "severity": record["severity"],
                    "latitude": record.get("latitude"),
                    "longitude": record.get("longitude")
                }
                for record in df.to_dict(orient="records")
            ]

            await db.historicaldata.insert_many(records)
            logger.info("✅ Initialized historical data in MongoDB")
        except Exception as e:
            logger.error(f"❌ Failed to initialize historical data: {e}")

@app.on_event("startup")
async def startup():
    await initialize_historical_data()
    logger.info("✅ MongoDB connected")

@app.on_event("shutdown")
async def shutdown():
    client.close()
    logger.info("✅ MongoDB disconnected")

@app.get("/weather", summary="Retrieve all weather data")
async def get_weather():
    try:
        records = await db.weatherdata.find().to_list(None)
        logger.info(f"✅ Loaded {len(records)} weather records")
        return {"count": len(records), "records": records}
    except Exception as e:
        logger.error(f"❌ Error loading weather data: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading weather data: {e}")

@app.post("/predict", summary="Score weather data using ML model")
async def predict_risk():
    try:
        records = await db.weatherdata.find().to_list(None)
        if not records:
            logger.warning("❌ No weather data to score")
            raise HTTPException(status_code=404, detail="No weather data to score")
        df = pd.DataFrame(records)
        df = generate_risk_scores(model, df)
        df = df.replace([float('inf'), float('-inf')], pd.NA)
        df = df.where(pd.notnull(df), None)
        for record in df.to_dict(orient="records"):
            await db.weatherdata.update_one(
                {"_id": record["_id"]},
                {"$set": {
                    "risk_score": record.get("risk_score"),
                    "risk_category": record.get("risk_category")
                }}
            )
        logger.info(f"✅ Scored {len(df)} weather records")
        return {"message": f"{len(df)} records scored.", "predictions": df.to_dict(orient="records")}
    except Exception as e:
        logger.error(f"❌ Error generating risk scores: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating risk scores: {e}")

@app.post("/alerts", summary="Create a new flood alert")
async def create_alert(alert: AlertModel):
    try:
        new_alert = {
            "location": alert.location,
            "risk_level": alert.risk_level,
            "message": alert.message,
            "language": alert.language,
            "timestamp": alert.timestamp
        }
        result = await db.alerts.insert_one(new_alert)
        new_alert["_id"] = str(result.inserted_id)
        logger.info(f"✅ Created alert for {alert.location}")
        return {"message": "Alert created", "alert": new_alert}
    except Exception as e:
        logger.error(f"❌ Error creating alert: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating alert: {e}")

@app.get("/alerts/history", summary="Retrieve alert history with optional filters")
async def get_alerts(
    location: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    try:
        filters = {}
        if location:
            filters["location"] = location
        if level:
            filters["risk_level"] = level.upper()
        alerts = await db.alerts.find(filters).sort("timestamp", -1).limit(limit).to_list(None)
        logger.info(f"✅ Loaded {len(alerts)} alerts")
        return {"count": len(alerts), "alerts": alerts}
    except Exception as e:
        logger.error(f"❌ Error loading alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading alerts: {e}")

async def load_historical_data() -> List[dict]:
    try:
        data = await db.historicaldata.find().to_list(None)
        logger.info(f"✅ Loaded {len(data)} historical records")
        return data
    except Exception as e:
        logger.error(f"❌ Error loading historical data: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading historical data: {e}")

@app.get("/api/historical", response_model=List[dict], description="Retrieve historical disaster data")
async def get_historical_data():
    return await load_historical_data()

@app.get("/api/locations", response_model=List[str], description="Get all unique locations from historical data")
async def get_all_locations():
    rows = await load_historical_data()
    locations = list({r["location"] for r in rows if "location" in r})
    logger.info(f"✅ Loaded {len(locations)} unique locations")
    return locations

@app.get("/api/risk/{location}", description="Assess historical risk profile for a location")
async def assess_risk_by_location(location: str):
    rows = await load_historical_data()
    filtered = [r for r in rows if str(r.get("location", "")).lower() == location.lower()]
    if not filtered:
        logger.warning(f"❌ No historical data for location: {location}")
        raise HTTPException(status_code=404, detail="No historical data for this location")
    severity_count = defaultdict(int)
    for r in filtered:
        severity = str(r.get("severity", "Unknown")).capitalize()
        severity_count[severity] += 1
    total = sum(severity_count.values())
    logger.info(f"✅ Assessed risk for {location}: {total} events")
    return {
        "location": location,
        "total_events": total,
        "risk_profile": dict(severity_count)
    }

try:
    model = joblib.load("rf_model.pkl")
    logger.info("✅ Loaded Random Forest model")
except Exception:
    logger.error("❌ Model not found")
    raise HTTPException(status_code=500, detail="Model not found, please train the model first")

@app.get("/", description="API root endpoint")
async def root():
    return {"message": "Welcome to the Crisis Connect API"}

@app.get("/collect", description="Collect weather and marine data for predefined districts")
async def collect_data():
    try:
        df = collect_all_data()
        records = df.to_dict(orient="records")
        await db.weatherdata.insert_many(records)
        logger.info(f"✅ Collected and saved {len(df)} weather records")
        return {"message": "Data collected", "count": len(df)}
    except Exception as e:
        logger.error(f"❌ Error collecting data: {e}")
        raise HTTPException(status_code=500, detail=f"Error collecting data: {e}")

@app.get("/risk-assessment", description="Generate risk scores for collected data")
async def assess_risk():
    try:
        records = await db.weatherdata.find().to_list(None)
        if not records:
            logger.warning("❌ No weather data available")
            raise HTTPException(status_code=404, detail="No weather data available")
        df = pd.DataFrame(records)
        df = generate_risk_scores(model, df)
        df = df.replace([float('inf'), float('-inf')], pd.NA)
        df = df.where(pd.notnull(df), None)
        for record in df.to_dict(orient="records"):
            await db.weatherdata.update_one(
                {"_id": record["_id"]},
                {"$set": {
                    "risk_score": record.get("risk_score"),
                    "risk_category": record.get("risk_category")
                }}
            )
        logger.info(f"✅ Generated risk scores for {len(df)} records")
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"❌ Error generating risk scores: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating risk scores: {e}")

@app.post("/risk/location", description="Assess risk for a specific location")
async def location_risk(data: LocationRequest):
    geolocator = Nominatim(user_agent="crisis-connect")
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=2)

    # Get coordinates if not provided
    if (data.lat is None or data.lon is None) and (data.place_name or data.district):
        location_query = data.place_name if data.place_name else data.district
        try:
            location = geolocator.geocode(location_query)
            if not location:
                logger.warning(f"❌ Could not geocode location: {location_query}")
                raise HTTPException(status_code=400, detail=f"Could not geocode location: {location_query}")
            data.lat = location.latitude
            data.lon = location.longitude
            if not data.district or data.district == "Unknown":
                data.district = location.address.split(",")[0].strip()
            logger.info(f"✅ Geocoded {location_query} to ({data.lat}, {data.lon})")
        except Exception as e:
            logger.error(f"❌ Geocoding failed: {e}")
            raise HTTPException(status_code=429, detail=f"Geocoding failed: {e}")

    if data.lat is None or data.lon is None:
        logger.warning("❌ Latitude and longitude required")
        raise HTTPException(status_code=400, detail="Latitude and longitude are required")

    # Fetch weather/marine data
    try:
        weather_hourly, marine_hourly = fetch_weather_and_marine_data(data.lat, data.lon, is_coastal=data.is_coastal)
        if not weather_hourly:
            logger.error("❌ Failed to fetch weather data")
            raise HTTPException(status_code=500, detail="Failed to fetch weather data")
    except Exception as e:
        logger.error(f"❌ Weather fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Weather fetch failed: {e}")

    # Extract features and predict risk
    try:
        features = extract_features(data.district or "Unknown", data.lat, data.lon, weather_hourly, marine_hourly)
        if not features:
            logger.error("❌ Feature extraction failed")
            raise HTTPException(status_code=500, detail="Feature extraction failed")
        df = pd.DataFrame([features])
        df = generate_risk_scores(model, df)
        df = df.replace([float("inf"), float("-inf")], pd.NA).where(pd.notnull(df), None)
        location_risk = df.to_dict(orient="records")[0]
        logger.info(f"✅ Generated risk for {data.district}: {location_risk['risk_score']}%")
        
        # Store in weatherdata
        location_risk["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = await db.weatherdata.insert_one(location_risk)
        location_risk["_id"] = str(result.inserted_id)
        logger.info(f"✅ Stored risk data for {data.district}")
    except Exception as e:
        logger.error(f"❌ Model prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # Analyze nearby locations
    nearby_predictions = []
    try:
        result = reverse((data.lat, data.lon), exactly_one=True, language="en")
        address = result.raw.get("address", {})
        possible_places = []
        for key in ["suburb", "town", "village", "city", "county"]:
            if address.get(key) and address.get(key) != data.district:
                possible_places.append(address[key])
        possible_places = list(dict.fromkeys(possible_places))[:3]

        for place in possible_places:
            try:
                place_location = geolocator.geocode(place)
                if place_location:
                    weather_hourly_n, marine_hourly_n = fetch_weather_and_marine_data(
                        place_location.latitude, place_location.longitude, is_coastal=data.is_coastal
                    )
                    features_n = extract_features(
                        place, place_location.latitude, place_location.longitude, weather_hourly_n, marine_hourly_n
                    )
                    if features_n:
                        df_n = pd.DataFrame([features_n])
                        df_n = generate_risk_scores(model, df_n)
                        df_n = df_n.replace([float("inf"), float("-inf")], pd.NA).where(pd.notnull(df_n), None)
                        risk_n = df_n.to_dict(orient="records")[0]
                        risk_n["analyzed_place"] = place
                        risk_n["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        await db.weatherdata.insert_one(risk_n)
                        risk_n["_id"] = str(result.inserted_id)
                        nearby_predictions.append(risk_n)
                        logger.info(f"✅ Generated risk for nearby {place}: {risk_n['risk_score']}%")
            except Exception as e:
                logger.warning(f"❌ Failed to analyze nearby {place}: {e}")
                nearby_predictions.append({"analyzed_place": place, "error": f"Could not analyze: {e}"})
    except Exception as e:
        logger.warning(f"❌ Failed to analyze nearby locations: {e}")
        nearby_predictions = [{"error": f"Could not analyze nearby locations: {e}"}]

    # Get nearest location metadata
    try:
        nearest_location = {
            "town": address.get("town"),
            "suburb": address.get("suburb"),
            "county": address.get("county"),
            "district": address.get("state_district"),
            "province": address.get("state"),
            "country": address.get("country"),
        }
        nearest_location = {k: v for k, v in nearest_location.items() if v is not None}
    except Exception as e:
        logger.warning(f"❌ Could not reverse geocode: {e}")
        nearest_location = {"error": f"Could not reverse geocode: {e}"}

    return {
        "location_input": {
            "district": data.district,
            "lat": data.lat,
            "lon": data.lon,
            "is_coastal": data.is_coastal
        },
        "location_risk": location_risk,
        "nearest_location": nearest_location,
        "nearby_locations_risk": nearby_predictions
    }