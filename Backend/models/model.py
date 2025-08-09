from pydantic import BaseModel
from typing import List, Optional

class AlertModel(BaseModel):
    location: str
    risk_level: str
    message: str
    language: str
    timestamp: str  # Changed from datetime to str to match Prisma schema

class LocationRequest(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    district: str = "Unknown"
    place_name: Optional[str] = None
    is_coastal: bool = False

class WeatherEntry(BaseModel):
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    rainfall: Optional[float] = None
    wind_speed: Optional[float] = None
    wave_height: Optional[float] = None
    location: str
    timestamp: str  # Changed to str to match Prisma schema
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class WeatherBatch(BaseModel):
    data: List[WeatherEntry]