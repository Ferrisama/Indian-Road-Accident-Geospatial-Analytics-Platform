#!/usr/bin/env python3
"""
AccidentIQ REST API
FastAPI endpoint for real-time accident risk prediction and analytics
"""
from src.models.ml_models import AccidentMLPredictor


from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, time
import h3
from sqlalchemy import create_engine
import uvicorn
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append('src')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AccidentIQ API",
    description="AI-Powered Road Safety Analytics and Prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ML predictor instance
predictor = None

# Pydantic models for API requests/responses


class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    hour: int
    vehicle_type: str
    road_type: str
    weather: str = "Clear"
    day_of_week: Optional[int] = None

    @validator('latitude')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v

    @validator('longitude')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v

    @validator('hour')
    def validate_hour(cls, v):
        if not 0 <= v <= 23:
            raise ValueError('Hour must be between 0 and 23')
        return v


class PredictionResponse(BaseModel):
    predicted_severity: int
    severity_probabilities: Dict[str, float]
    risk_score: float
    risk_level: str
    recommendations: List[str]
    h3_index: str


class HotspotRequest(BaseModel):
    latitude: float
    longitude: float
    radius_km: float = 5.0


class AccidentStatsResponse(BaseModel):
    total_accidents: int
    fatal_accidents: int
    avg_severity: float
    cities_covered: int
    date_range: Dict[str, str]


class CityRiskResponse(BaseModel):
    city: str
    total_accidents: int
    fatality_rate: float
    avg_severity: float
    peak_accident_hour: int
    riskiest_road_type: str
    top_hotspots: List[Dict[str, Any]]
    recommendations: List[str]


@app.on_event("startup")
async def startup_event():
    """Initialize ML models on startup"""
    global predictor
    try:
        predictor = AccidentMLPredictor()

        # Try to load existing models
        if (Path("models") / "severity_prediction.pkl").exists():
            predictor.load_models()
            logger.info("✅ Loaded pre-trained models")
        else:
            logger.warning(
                "⚠️ No pre-trained models found. Train models first!")
            # For demo purposes, we'll continue without models

    except Exception as e:
        logger.error(f"Error initializing predictor: {e}")


@app.get("/", summary="API Health Check")
async def root():
    """API health check endpoint"""
    return {
        "message": "AccidentIQ API is running!",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", summary="Detailed Health Check")
async def health_check():
    """Detailed health check with system status"""
    global predictor

    status = {
        "api_status": "healthy",
        "database_connected": False,
        "models_loaded": False,
        "timestamp": datetime.now().isoformat()
    }

    # Check database connection
    try:
        engine = create_engine(
            "postgresql://postgres:password123@localhost:5432/accidentiq")
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            status["database_connected"] = True
    except:
        status["database_connected"] = False

    # Check models
    if predictor and predictor.models:
        status["models_loaded"] = True
        status["available_models"] = list(predictor.models.keys())

    return status


@app.post("/predict/accident-risk", response_model=PredictionResponse, summary="Predict Accident Risk")
async def predict_accident_risk(request: PredictionRequest):
    """
    Predict accident risk for given location and conditions

    Returns severity prediction, risk probabilities, and safety recommendations
    """
    global predictor

    if not predictor or 'severity_prediction' not in predictor.models:
        raise HTTPException(
            status_code=503, detail="ML models not available. Please train models first.")

    try:
        # Get prediction from ML model
        prediction = predictor.predict_accident_risk(
            lat=request.latitude,
            lon=request.longitude,
            hour=request.hour,
            vehicle_type=request.vehicle_type,
            road_type=request.road_type,
            weather=request.weather
        )

        # Calculate H3 index for location
        h3_index = h3.geo_to_h3(request.latitude, request.longitude, 8)

        # Determine risk level
        risk_score = prediction['risk_score']
        if risk_score < 1.5:
            risk_level = "LOW"
        elif risk_score < 2.2:
            risk_level = "MEDIUM"
        elif risk_score < 2.7:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        # Generate recommendations
        recommendations = []

        if risk_level in ["HIGH", "CRITICAL"]:
            recommendations.append(
                "⚠️ High risk area - exercise extreme caution")

        if request.hour in [7, 8, 9, 17, 18, 19, 20]:
            recommendations.append(
                "🚦 Rush hour - expect heavy traffic and delays")

        if request.weather != "Clear":
            recommendations.append(
                "🌧️ Poor weather conditions - reduce speed and increase following distance")

        if request.vehicle_type in ["Motorcycle", "Bicycle"]:
            recommendations.append(
                "🏍️ Vulnerable vehicle type - wear protective gear and stay visible")

        if request.road_type == "Highway":
            recommendations.append(
                "🛣️ Highway driving - maintain safe speeds and distances")

        if not recommendations:
            recommendations.append(
                "✅ Relatively safe conditions - maintain standard safety practices")

        return PredictionResponse(
            predicted_severity=prediction['predicted_severity'],
            severity_probabilities=prediction['severity_probabilities'],
            risk_score=risk_score,
            risk_level=risk_level,
            recommendations=recommendations,
            h3_index=h3_index
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/analytics/stats", response_model=AccidentStatsResponse, summary="Get Overall Statistics")
async def get_accident_stats():
    """Get overall accident statistics from the database"""
    try:
        engine = create_engine(
            "postgresql://postgres:password123@localhost:5432/accidentiq")

        # Get basic stats
        query = """
        SELECT 
            COUNT(*) as total_accidents,
            SUM(CASE WHEN severity = 3 THEN 1 ELSE 0 END) as fatal_accidents,
            AVG(severity) as avg_severity,
            COUNT(DISTINCT city) as cities_covered,
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM accidents
        """

        result = pd.read_sql(query, engine)
        row = result.iloc[0]

        return AccidentStatsResponse(
            total_accidents=int(row['total_accidents']),
            fatal_accidents=int(row['fatal_accidents']),
            avg_severity=float(row['avg_severity']),
            cities_covered=int(row['cities_covered']),
            date_range={
                "start": str(row['min_date']),
                "end": str(row['max_date'])
            }
        )

    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get statistics: {str(e)}")


@app.get("/analytics/city/{city_name}", response_model=CityRiskResponse, summary="Get City Risk Analysis")
async def get_city_analysis(city_name: str):
    """Get detailed risk analysis for a specific city"""
    global predictor

    if not predictor:
        raise HTTPException(
            status_code=503, detail="Predictor not initialized")

    try:
        report = predictor.generate_city_risk_report(city_name)

        if "error" in report:
            raise HTTPException(status_code=404, detail=report["error"])

        return CityRiskResponse(
            city=report['city'],
            total_accidents=report['total_accidents'],
            fatality_rate=report['fatality_rate'],
            avg_severity=report['avg_severity'],
            peak_accident_hour=report['peak_accident_hour'],
            riskiest_road_type=report['riskiest_road_type'],
            top_hotspots=report.get('top_hotspots', []),
            recommendations=report['recommendations']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"City analysis error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/analytics/hotspots", summary="Get Accident Hotspots")
async def get_hotspots(
    city: Optional[str] = Query(None, description="Filter by city"),
    limit: int = Query(10, description="Number of hotspots to return")
):
    """Get accident hotspots, optionally filtered by city"""
    try:
        engine = create_engine(
            "postgresql://postgres:password123@localhost:5432/accidentiq")

        query = """
        SELECT 
            h3_index,
            city,
            state,
            accident_count,
            avg_severity,
            total_casualties,
            total_fatalities
        FROM accident_hotspots
        """

        if city:
            query += f" WHERE city = '{city}'"

        query += f" ORDER BY accident_count DESC LIMIT {limit}"

        hotspots = pd.read_sql(query, engine)

        # Convert to list of dictionaries
        hotspots_list = []
        for _, row in hotspots.iterrows():
            # Get center coordinates from H3 index
            try:
                lat, lon = h3.h3_to_geo(row['h3_index'])
                hotspots_list.append({
                    "h3_index": row['h3_index'],
                    "city": row['city'],
                    "state": row['state'],
                    "latitude": lat,
                    "longitude": lon,
                    "accident_count": int(row['accident_count']),
                    "avg_severity": float(row['avg_severity']),
                    "total_casualties": int(row['total_casualties']),
                    "total_fatalities": int(row['total_fatalities'])
                })
            except:
                continue

        return {
            "hotspots": hotspots_list,
            "total_found": len(hotspots_list),
            "filter_applied": {"city": city} if city else None
        }

    except Exception as e:
        logger.error(f"Hotspots error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get hotspots: {str(e)}")


@app.get("/analytics/temporal", summary="Get Temporal Analysis")
async def get_temporal_analysis(
    city: Optional[str] = Query(None, description="Filter by city"),
    analysis_type: str = Query(
        "hourly", description="Type: hourly, daily, monthly")
):
    """Get temporal accident patterns"""
    try:
        engine = create_engine(
            "postgresql://postgres:password123@localhost:5432/accidentiq")

        if analysis_type == "hourly":
            group_by = "hour"
        elif analysis_type == "daily":
            group_by = "day_of_week"
        elif analysis_type == "monthly":
            group_by = "month"
        else:
            raise HTTPException(
                status_code=400, detail="Invalid analysis_type. Use: hourly, daily, monthly")

        query = f"""
        SELECT 
            {group_by},
            COUNT(*) as accident_count,
            AVG(severity) as avg_severity,
            SUM(casualties) as total_casualties
        FROM accidents
        """

        if city:
            query += f" WHERE city = '{city}'"

        query += f" GROUP BY {group_by} ORDER BY {group_by}"

        temporal_data = pd.read_sql(query, engine)

        return {
            "analysis_type": analysis_type,
            "city_filter": city,
            "data": temporal_data.to_dict('records')
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Temporal analysis error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/analytics/risk-factors", summary="Get Risk Factor Analysis")
async def get_risk_factors():
    """Analyze risk factors across different dimensions"""
    try:
        engine = create_engine(
            "postgresql://postgres:password123@localhost:5432/accidentiq")

        # Risk by vehicle type
        vehicle_risk = pd.read_sql("""
        SELECT 
            vehicle_type,
            COUNT(*) as accident_count,
            AVG(severity) as avg_severity,
            SUM(fatalities) as total_fatalities
        FROM accidents
        GROUP BY vehicle_type
        ORDER BY avg_severity DESC
        """, engine)

        # Risk by road type
        road_risk = pd.read_sql("""
        SELECT 
            road_type,
            COUNT(*) as accident_count,
            AVG(severity) as avg_severity,
            SUM(fatalities) as total_fatalities
        FROM accidents
        GROUP BY road_type
        ORDER BY avg_severity DESC
        """, engine)

        # Risk by weather
        weather_risk = pd.read_sql("""
        SELECT 
            weather,
            COUNT(*) as accident_count,
            AVG(severity) as avg_severity,
            SUM(fatalities) as total_fatalities
        FROM accidents
        GROUP BY weather
        ORDER BY avg_severity DESC
        """, engine)

        return {
            "vehicle_risk": vehicle_risk.to_dict('records'),
            "road_risk": road_risk.to_dict('records'),
            "weather_risk": weather_risk.to_dict('records')
        }

    except Exception as e:
        logger.error(f"Risk factors error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/cities", summary="Get Available Cities")
async def get_cities():
    """Get list of cities with accident data"""
    try:
        engine = create_engine(
            "postgresql://postgres:password123@localhost:5432/accidentiq")

        cities = pd.read_sql("""
        SELECT 
            city,
            state,
            COUNT(*) as accident_count,
            AVG(severity) as avg_severity
        FROM accidents
        GROUP BY city, state
        ORDER BY accident_count DESC
        """, engine)

        return {
            "cities": cities.to_dict('records'),
            "total_cities": len(cities)
        }

    except Exception as e:
        logger.error(f"Cities error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get cities: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "api_endpoint:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
