from typing import Any, List, Tuple

import os
from datetime import datetime

import pandas as pd
try:
    from google import genai  # type: ignore
except Exception:
    genai = None  # Fallback if google-genai is not installed
from motor.motor_asyncio import AsyncIOMotorDatabase


# Thresholds for generating alerts from predictions
HIGH_RISK_THRESHOLD = 70
MODERATE_RISK_THRESHOLD = 40

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

TEMPLATES = {
    "high": "⚠️ [HIGH RISK] Severe weather in {location}.\nRisk Score: {risk_score}%.\nWave Height: {wave_height}m.\nMove to higher ground and follow local updates.",
    "moderate": "⚠️ [MODERATE RISK] Unstable weather in {location}.\nRisk Score: {risk_score}%.\nWave Height: {wave_height}m.\nMonitor updates and stay cautious.",
}


def translate_with_gemini(text: str, target_language: str) -> str:
    if not GEMINI_API_KEY or genai is None:
        # Fallback: return original text when translator is unavailable
        return text

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"Translate the following English text to {target_language}:\n{text}"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    content_obj = response.candidates[0].content
    if hasattr(content_obj, "parts") and len(content_obj.parts) > 0:
        return content_obj.parts[0].text.strip()
    if hasattr(content_obj, "text"):
        return content_obj.text.strip()
    return str(content_obj)


def generate_alerts(data: pd.DataFrame) -> List[dict[str, Any]]:
    alerts: List[dict[str, Any]] = []
    translation_cache: dict[str, Tuple[str, str]] = {}

    if data.empty:
        return alerts

    for _, row in data.iterrows():
        risk_score = float(row.get("risk_score", 0))
        location = str(row.get("location", "Unknown"))
        wave_height = float(row.get("wave_height", 0) or 0)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if risk_score >= HIGH_RISK_THRESHOLD:
            level = "high"
        elif risk_score >= MODERATE_RISK_THRESHOLD:
            level = "moderate"
        else:
            continue

        message_en = TEMPLATES[level].format(
            location=location,
            risk_score=int(risk_score),
            wave_height=round(wave_height, 1),
        )

        if message_en not in translation_cache:
            message_zu = translate_with_gemini(message_en, "isiZulu")
            message_xh = translate_with_gemini(message_en, "isiXhosa")
            translation_cache[message_en] = (message_zu, message_xh)
        else:
            message_zu, message_xh = translation_cache[message_en]

        alerts.append(
            {
                "timestamp": timestamp,
                "location": location,
                "risk_score": risk_score,
                "risk_level": level.upper(),
                "message_en": message_en,
                "message_zu": message_zu,
                "message_xh": message_xh,
            }
        )

    return alerts


async def generate_alerts_from_db(db: AsyncIOMotorDatabase, limit: int = 1000) -> List[dict[str, Any]]:
    """Fetch recent predictions from MongoDB, generate alerts, and store them.

    Returns the list of generated alert documents.
    """
    # Pull latest predictions
    cursor = (
        db["predictions"]
        .find({}, {"location": 1, "risk_score": 1, "wave_height": 1, "timestamp": 1})
        .sort("timestamp", -1)
        .limit(limit)
    )
    predictions = await cursor.to_list(length=limit)

    if not predictions:
        return []

    df = pd.DataFrame(predictions)
    alerts = generate_alerts(df)

    if not alerts:
        return []

    await db["alerts"].insert_many(alerts)
    return alerts
