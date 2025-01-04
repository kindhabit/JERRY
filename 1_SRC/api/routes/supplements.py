from fastapi import APIRouter
from typing import Dict
from models.health_data import HealthData
from core.services.health_service import HealthService
from core.vector_db.vector_store_manager import VectorStoreManager

router = APIRouter()
health_service = HealthService(VectorStoreManager())

@router.post("/analyze")
async def analyze_supplements(health_data: HealthData):
    """영양제 분석 API"""
    # 1. 1차 분석 (건강지표 기반)
    initial_analysis = await health_service.analyze_health_metrics(health_data)
    
    # 2. 간섭 가능성 확인
    if initial_analysis["has_interactions"]:
        interaction_notice = await health_service.generate_interaction_notice(
            initial_analysis
        )
        initial_analysis["interaction_notice"] = interaction_notice
    
    return initial_analysis

@router.post("/detailed-analysis")
async def get_detailed_analysis(
    health_data: HealthData,
    initial_recommendations: Dict
):
    """상세 분석 API"""
    return await health_service.detailed_interaction_analysis(
        health_data,
        initial_recommendations
    ) 