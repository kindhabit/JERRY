from fastapi import APIRouter
from typing import Dict
from models.health_data import HealthData
from core.services.health_service import HealthService
from core.vector_db.vector_store_manager import ChromaManager
from fastapi.responses import JSONResponse
import logging

router = APIRouter()
health_service = HealthService(ChromaManager())
logger = logging.getLogger(__name__)

@router.post("/analyze")
async def analyze_supplements(health_data: HealthData):
    """영양제 분석 API"""
    try:
        logger.info("영양제 분석 API 호출 시작")
        
        # 1. 1차 분석 (건강지표 기반)
        initial_analysis = await health_service.analyze_health_metrics(health_data)
        logger.info(f"1차 분석 결과: {initial_analysis}")
        
        # 2. 간섭 가능성 확인
        try:
            has_interactions = initial_analysis.get("has_interactions", False)
            logger.info(f"간섭 가능성 확인: {has_interactions}")
            
            if has_interactions:
                interaction_notice = await health_service.generate_interaction_notice(
                    initial_analysis
                )
                initial_analysis["interaction_notice"] = interaction_notice
                logger.info("간섭 알림 생성 완료")
        except Exception as e:
            logger.error(f"간섭 확인 중 에러 발생 - 타입: {type(e).__name__}")
            logger.error(f"에러 메시지: {str(e)}")
            logger.error("스택 트레이스:", exc_info=True)
            logger.error(f"initial_analysis 데이터: {initial_analysis}")
            raise
        
        return JSONResponse(
            status_code=200,
            content=initial_analysis
        )
    except Exception as e:
        logger.error(f"API 처리 중 에러 발생 - 타입: {type(e).__name__}")
        logger.error(f"에러 메시지: {str(e)}")
        logger.error("스택 트레이스:", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )

@router.post("/detailed-analysis")
async def get_detailed_analysis(
    health_data: HealthData,
    initial_recommendations: Dict
):
    """상세 분석 API"""
    try:
        result = await health_service.detailed_interaction_analysis(
            health_data,
            initial_recommendations
        )
        return JSONResponse(
            status_code=200,
            content=result
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        ) 