from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from datetime import datetime
from typing import Dict

app = FastAPI(title="Name Classifier API")

# ====================== CORS (Required for grading script) ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Access-Control-Allow-Origin: *
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== PROCESSING LOGIC ======================
def process_genderize_response(raw_data: Dict, original_name: str) -> Dict:
    """Process raw Genderize response according to exact spec"""
    if not raw_data or raw_data.get("gender") is None or raw_data.get("count", 0) == 0:
        return {
            "status": "error",
            "message": "No prediction available for the provided name"
        }

    probability = raw_data.get("probability", 0.0)
    sample_size = raw_data.get("count", 0)

    is_confident = (probability >= 0.7) and (sample_size >= 100)

    return {
        "status": "success",
        "data": {
            "name": original_name.strip().capitalize(),
            "gender": raw_data["gender"],
            "probability": round(probability, 4),
            "sample_size": sample_size,
            "is_confident": is_confident,
            "processed_at": datetime.utcnow().isoformat() + "Z"   # UTC ISO 8601
        }
    }


# ====================== ENDPOINT ======================
@app.get("/api/classify")
async def classify_name(name: str = Query(..., min_length=2, description="Name to classify")):
    """
    GET /api/classify/?name=Blessed
    """
    name = name.strip()

    # Input validation (FastAPI + manual check)
    if not name:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Missing or empty name"}
        )

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(
                "https://api.genderize.io/",
                params={"name": name}
            )
            
            response.raise_for_status()
            raw_data = response.json()

        # Process the raw response
        result = process_genderize_response(raw_data, name)
        return result

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail={"status": "error", "message": "Request to Genderize API timed out"}
        )
    except httpx.RequestError:
        raise HTTPException(
            status_code=502,
            detail={"status": "error", "message": "Failed to connect to Genderize API"}
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": "Internal server error"}
        )
