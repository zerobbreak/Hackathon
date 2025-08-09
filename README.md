# Crisis Connect

## Overview
Crisis Connect is an AI-powered early flood detection and community alert system designed for high-risk regions in South Africa, such as KwaZulu-Natal. It leverages real-time weather and marine data from the Open-Meteo API to predict flood risks using machine learning, generates multilingual alerts (English, isiZulu, isiXhosa), and provides a user-friendly React dashboard for visualization. The backend is built with FastAPI and uses MongoDB for data storage, ensuring scalability and reliability for disaster response.

The system fetches data for predefined districts (e.g., eThekwini, Ugu), scores risk using a Random Forest model, and supports location-based risk assessment with geocoding. It's ideal for local governments, NGOs, and communities to act proactively against floods.

## Features
- **Real-Time Data Collection**: Fetches weather (temperature, humidity, rainfall, wind speed) and marine (wave height) data from Open-Meteo API for predefined districts.
- **Flood Risk Prediction**: Uses a pre-trained Random Forest model to generate risk scores and categories (Low, Moderate, High).
- **Multilingual Alerts**: Generates alerts in English, isiZulu, and isiXhosa, stored in MongoDB for history and filtering.
- **Location-Based Risk Assessment**: Supports custom locations with geocoding (via geopy) and nearby location analysis.
- **Historical Data**: Loads and queries historical disaster data from MongoDB for risk profiling.
- **Dashboard**: React frontend with shadcn/ui for visualizing alerts, flood risk map (Mapbox), and wave height chart (Chart.js).
- **Twilio Integration**: (Optional) Send alerts via WhatsApp for community outreach.

## Tech Stack
- **Backend**: FastAPI (Python), MongoDB (via pymongo)
- **Data Fetching**: Open-Meteo API, geopy for geocoding
- **Machine Learning**: scikit-learn (Random Forest), joblib for model serialization
- **Frontend**: React with Vite, shadcn/ui, Mapbox GL JS, Chart.js
- **Other**: Requests-cache, Retry-requests for API reliability, Logging for debugging
- **Deployment**: Render (backend), Vercel (frontend)

## Installation
1. **Prerequisites**:
   - Python 3.8+
   - MongoDB (local or Atlas)
   - Node.js 18+ for frontend

2. **Backend Setup**:
   - Clone the repository:
     ```
     git clone <repository-url>
     cd backend
     ```
   - Install dependencies:
     ```
     pip install -r requirements.txt
     ```
   - Set environment variables in `.env`:
     ```
     MONGO_CONNECTION_STRING=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
     ```
   - Run the app:
     ```
     uvicorn main:app --reload
     ```
   - Access Swagger UI: `http://localhost:8000/docs`

3. **Frontend Setup**:
   - Navigate to frontend directory:
     ```
     cd frontend/crisis-connect-dashboard
     ```
   - Install dependencies:
     ```
     npm install
     ```
   - Run the app:
     ```
     npm run dev
     ```
   - Access dashboard: `http://localhost:3000`

## Usage
1. **Collect Data**:
   - Call `/collect` to fetch and store weather data for districts (e.g., eThekwini).
   - Response: `{"message": "Data collected", "count": 5}`

2. **Generate Risk Scores**:
   - Call `/predict` to score stored weather data and update MongoDB with `risk_score` and `risk_category`.
   - Response: Scored records.

3. **Create Alert**:
   - POST to `/alerts` with body:
     ```json
     {
       "location": "eThekwini",
       "risk_level": "HIGH",
       "message": "Severe flood warning",
       "language": "English",
       "timestamp": "2025-08-07 12:00:00"
     }
     ```
   - Response: `{"message": "Alert created", "alert": {...}}`

4. **Retrieve Alerts**:
   - GET `/alerts/history?location=eThekwini&language=English&limit=10`
   - Response: Filtered alerts.

5. **Location Risk Assessment**:
   - POST to `/risk/location` with body:
     ```json
     {
       "place_name": "eThekwini",
       "is_coastal": true
     }
     ```
   - Response: Risk score, wave height, and nearby locations.

6. **Historical Data**:
   - GET `/api/historical` for all historical records.
   - GET `/api/risk/eThekwini` for risk profile.

## Testing
1. **Insert Dummy Data**:
   - Run `insert_dummy_data.py` to populate `alerts`, `weatherdata`, and `historicaldata`.

2. **Run Tests**:
   - Use Swagger UI (`http://localhost:8000/docs`) for endpoint testing.
   - Verify data in MongoDB Compass or `mongo` shell.
   - Run frontend and check alerts, map, and chart with dummy data.

3. **Edge Cases**:
   - Empty database: Call `/collect` to populate.
   - Invalid location: Test `/risk/location` with invalid `place_name` (expect 400 error).
   - Duplicate alert: Test `/alerts` with existing timestamp/location (expect 400 error).

## Deployment
1. **Backend (Render)**:
   - Deploy `main.py`, `services/predict.py`, `models/model.py`, and `utils/db.py`.
   - Set `MONGO_CONNECTION_STRING` in Render’s environment variables.
   - Upload `rf_model.pkl` and `data_disaster.xlsx`.
   - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

2. **Frontend (Vercel)**:
   - Deploy the React app.
   - Update `API_URL` in `App.jsx` to your Render URL.

3. **MongoDB Atlas**:
   - Use the provided connection string.
   - Whitelist Render’s IP for access.

## Contributing
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/new-feature`.
3. Commit changes: `git commit -m "Add new feature"`.
4. Push branch: `git push origin feature/new-feature`.
5. Submit a pull request.

## License
MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments
- Open-Meteo API for weather and marine data.
- scikit-learn for ML model.
- FastAPI and MongoDB for backend.
- React and shadcn/ui for frontend.
