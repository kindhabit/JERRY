from fastapi import Request, HTTPException, APIRouter
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.logger import logger
from typing import Any

from ..models.health_data import HealthDataInput
from ..services.health_service import health_service

router = APIRouter()

@router.post("/analyze")
async def analyze_supplements(request: Request, health_data: HealthDataInput):
    try:
        client_ip = request.client.host
        logger.info(f"Client IP: {client_ip}, Request: {request.method} {request.url}")
        
        result = await health_service.analyze_supplements(health_data)
        return result
    except Exception as e:
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error(f"Error Message: {str(e)}")
        logger.error("Stack Trace:", exc_info=True)
        logger.error(f"Request Data: {health_data.dict()}")
        logger.error(f"Response status: 500 for {request.client.host}")
        raise HTTPException(status_code=500, detail=str(e)) 