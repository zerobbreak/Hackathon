from typing import Any

import pandas as pd
import os
from datetime import datetime
from google import genai

HIGH_RISK_THRESHOLD = 70
MODERATE_RISK_THRESHOLD = 40
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

TEMPLATES = {
    "high": "⚠️ [HIGH RISK] Severe weather in {location}.\nRisk Score: {risk_score}%.\nWave Height: {wave_height}m.\nMove to higher ground and follow local updates.",
    "moderate": "⚠️ [MODERATE RISK] Unstable weather in {location}.\nRisk Score: {risk_score}%.\nWave Height: {wave_height}m.\nMonitor updates and stay cautious."
}

def translate_with_gemini(text, target_language):
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"Translate the following English text to {target_language}:\n{text}"

    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=prompt,
    )

    # Extract the text from the Content object
    content_obj = response.candidates[0].content
    if hasattr(content_obj, "parts") and len(content_obj.parts) > 0:
        return content_obj.parts[0].text.strip()
    elif hasattr(content_obj, "text"):
        return content_obj.text.strip()
    else:
        return str(content_obj)

def generate_alerts(data: pd.DataFrame) -> list[Any]:
    alerts = []
    translation_cache = {}
    for _, row in data.iterrows():
        risk_score = row['risk_score']
        location = row['location']
        wave_height = row['wave_height']
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if risk_score >= HIGH_RISK_THRESHOLD:
            level = "high"
        elif risk_score >= MODERATE_RISK_THRESHOLD:
            level = "moderate"
        else:
            continue

        message_en = TEMPLATES[level].format(location=location, risk_score=int(risk_score), wave_height=round(wave_height, 1))

        # Cache translations
        if message_en not in translation_cache:
            message_zu = translate_with_gemini(message_en, "isiZulu")
            message_xh = translate_with_gemini(message_en, "isiXhosa")
            translation_cache[message_en] = (message_zu, message_xh)
        else:
            message_zu, message_xh = translation_cache[message_en]

        alert = {
            "timestamp": timestamp,
            "location": location,
            "risk_score": risk_score,
            "risk_level": level.upper(),
            "message_en": message_en,
            "message_zu": message_zu,
            "message_xh": message_xh
        }

        alerts.append(alert)

    if alerts:

    # alerts_df = pd.DataFrame(alerts)
    # alerts_df.to_csv("alerts_log.csv", index=False)
    print("💾 Alerts saved to 'alerts_log.csv'")
    return alerts

print("🔍 Analyzing data for severe weather alerts...")
def main():
    try:
        df = pd.read_csv("../weather_data_scored.csv")
    except FileNotFoundError:
        print("❌ 'weather_data_scored.csv' not found. Please ensure the file exists.")
        return

    if df.empty:
        print("❌ No data available in 'weather_data_scored.csv'. Exiting.")
        return

    print("📊 Data loaded successfully. Generating alerts...")
    alerts_df = generate_alerts(df)
    if not alerts_df.empty:
        print("✅ Alerts generated successfully.")
    else:
        print("❌ No alerts generated. Check your data.")

main()