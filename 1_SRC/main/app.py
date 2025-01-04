import logging
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from core.vector_db.vector_store_manager import ChromaManager
from config.config_loader import CONFIG
import os
import resource
import gc
from typing import Dict, List
import uuid
from core.services.health_service import HealthService
import json
import psutil
from models.health_data import HealthData
from core.analysis.client_health_analyzer import HealthDataAnalyzer
from fastapi.responses import JSONResponse
import uvicorn
import yaml

# 메모리 제한 설정
def limit_memory(max_mem_mb=1024):  # 1GB
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (max_mem_mb * 1024 * 1024, hard))
    except Exception as e:
        logger.error(f"메모리 제한 설정 실패: {e}")

# ChromaDB 클라이언트 초기화
chroma_client = ChromaManager()

# FastAPI 서버 설정
def load_config():
    with open('config/config.yaml', 'r') as file:
        return yaml.safe_load(file)

config = load_config()
fastapi_config = config["fastapi"]

# FastAPI 애플리케이션 초기화
kindhabit_app = FastAPI()

# 로그 설정
log_dir = os.path.abspath("./logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "server.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@kindhabit_app.middleware("http")
async def gc_middleware(request: Request, call_next):
    gc.collect()  # 요청 처리 전 GC 실행
    response = await call_next(request)
    return response

@kindhabit_app.middleware("http")
async def log_requests(request: Request, call_next):
    """모든 HTTP 요청을 로깅하는 미들웨어."""
    client_ip = request.client.host
    logger.info(f"Client IP: {client_ip}, Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code} for {client_ip}")
    return response

@kindhabit_app.on_event("shutdown")
async def shutdown_event():
    """앱 종료 시 정리 작업"""
    logger.info("Shutting down application...")
    if 'chroma_client' in globals():
        try:
            await chroma_client.close()
            logger.info("ChromaDB connection closed")
        except Exception as e:
            logger.error(f"Error closing ChromaDB connection: {e}")

@kindhabit_app.get("/")
def read_root():
    """루트 엔드포인트."""
    logger.info("Root endpoint accessed.")
    return {"status": "kindhabit-RAG-FASTAPI is running"}

@kindhabit_app.post("/api/analyze-request/")
async def analyze_request(request: Request):
    config_manager = ConfigManager()
    # 설정 기반 처리
    pass

@kindhabit_app.get("/api/health-categories/")
async def get_health_categories():
    config_manager = ConfigManager()
    return config_manager.get_all_health_categories()

@kindhabit_app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """404 에러 핸들러"""
    return JSONResponse(
        status_code=404,
        content={
            "status": "error",
            "message": "요청하신 엔드포인트를 찾을 수 없습니다.",
            "detail": "올바른 API 엔드포인트인지 확인해주세요.",
            "available_endpoints": [
                "/api/analyze-request/",
                "/api/analyze-health-data/interactions",
                "/api/analyze-health-data/final",
                "/api/analyze-health-data/process-answers"
            ]
        }
    )

@kindhabit_app.post("/api/analyze-health-data/interactions")
async def analyze_interactions(
    analysis_id: str,
    recommendations: Dict[str, List[str]]
):
    """추천 보충제들의 상호작용 분석"""
    try:
        recommender = HealthService()
        interaction_analysis = await recommender.analyze_interactions(recommendations)
        
        if interaction_analysis["has_interactions"]:
            return {
                "status": "success",
                "analysis_id": analysis_id,
                "has_interactions": True,
                "interaction_details": interaction_analysis["interactions"],
                "questions": interaction_analysis["questions"],
                "evidence": interaction_analysis["evidence"]
            }
        return {
            "status": "success",
            "has_interactions": False
        }
    except Exception as e:
        logger.error(f"상호작용 분석 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@kindhabit_app.post("/api/analyze-health-data/final")
async def get_final_recommendations(
    analysis_id: str,
    answers: List[str],
    initial_recommendations: Dict
):
    """사용자 답변 기반 최종 추천"""
    try:
        recommender = HealthService()
        final_analysis = await recommender.process_user_answers(
            answers, 
            initial_recommendations
        )
        
        return {
            "status": "success",
            "analysis_id": analysis_id,
            "final_recommendations": final_analysis["recommendations"],
            "evidence": final_analysis["evidence"],
            "safety_notes": final_analysis["safety_notes"],
            "dosage_advice": final_analysis["dosage_advice"]
        }
    except Exception as e:
        logger.error(f"최종 추천 생성 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@kindhabit_app.put("/papers/{pmid}/update")
async def update_paper(pmid: str, new_data: Dict):
    """논문 데이터 업데이트"""
    try:
        await chroma_client.update_paper(pmid, new_data)
        return {"status": "success", "message": f"Paper {pmid} updated successfully"}
    except Exception as e:
        logger.error(f"Paper update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@kindhabit_app.get("/papers/{pmid}/history")
async def get_paper_history(pmid: str):
    """논문 버전 히스토리 조회"""
    try:
        history = await chroma_client.get_paper_history(pmid)
        return {
            "pmid": pmid,
            "history": history
        }
    except Exception as e:
        logger.error(f"Failed to get paper history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@kindhabit_app.get("/supplements")
async def get_supplements():
    """설정된 영양 목록 조회"""
    return {
        "supplements": CONFIG["pubmed"]["supplements"],
        "health_keywords": CONFIG["pubmed"]["health_keywords"]
    }

def remove_duplicates(results: List[Dict]) -> List[Dict]:
    """중복 결과 제거"""
    seen = set()
    unique = []
    for result in results:
        key = f"{result['pmid']}_{result.get('title', '')}"
        if key not in seen:
            seen.add(key)
            unique.append(result)
    return unique

def sort_by_relevance(results: List[Dict]) -> List[Dict]:
    """관련성 점수로 결과 정렬"""
    def get_score(result):
        score = 0
        # evidence level 점수
        evidence_scores = {"A": 3, "B": 2, "C": 1}
        score += evidence_scores.get(result.get("evidence_level", "C"), 0)
        # 긍정적 효과 수
        score += len(result.get("positive_effects", []))
        return score
        
    return sorted(results, key=get_score, reverse=True)

def extract_conditions(data: Dict) -> List[str]:
    """건강 데이터에서 조건 추출"""
    conditions = []
    health_keywords = get_health_keywords()
    
    # 키워드 매핑 생성
    keyword_mapping = {
        keyword.replace(" ", "_").lower(): [keyword]
        for keyword in health_keywords
    }
    
    # 데이터의 각 필드를 검사하여 매칭되는 키워드 찾기
    for field, values in data.items():
        for key, keywords in keyword_mapping.items():
            if any(keyword.lower() in str(values).lower() for keyword in keywords):
                conditions.append(key)
    
    return list(set(conditions))  # 중복 제거

@kindhabit_app.post("/admin/chroma/reinit")
async def reinit_chroma(background_tasks: BackgroundTasks):
    """ChromaDB 초기화 엔드포인트"""
    try:
        # 백그라운드 작업으로 실행
        background_tasks.add_task(
            manage_chroma_database,
            action="reinit",
            force=True,  # API 호출은 자동으로 force 적용
            debug=True   # 디버그 모드 성화
        )
        return {"message": "ChromaDB 초기화 작업이 시작되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@kindhabit_app.post("/admin/chroma/update")
async def update_chroma(background_tasks: BackgroundTasks):
    """ChromaDB 업데이트 엔드포인트"""
    try:
        # 백그라운드 작업으로 실행
        background_tasks.add_task(
            manage_chroma_database,
            action="update",
            debug=True
        )
        return {"message": "ChromaDB 업데이트 작업이 시작되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 테스트용 엔드포인트
@kindhabit_app.post("/admin/chroma/test")
async def test_chroma(background_tasks: BackgroundTasks):
    """ChromaDB 테스트 엔드포인트"""
    try:
        background_tasks.add_task(
            manage_chroma_database,
            action="update",
            debug=True,
            test_mode=True
        )
        return {"message": "ChromaDB 테스트 작업이 시작되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 추가 엔드포인트 - 사용자 답변 처리용
@kindhabit_app.post("/api/analyze-health-data/process-answers")
async def process_user_answers(
    answers: List[str],
    initial_recommendations: Dict
):
    """사용자 답변을 처리하여 최종 추천 생성"""
    try:
        recommender = HealthService()
        final_recommendations = await recommender.process_user_answers(
            answers, 
            initial_recommendations
        )
        return {
            "status": "success",
            "final_recommendations": final_recommendations["final_recommendations"],
            "evidence": final_recommendations["evidence"],
            "safety_notes": final_recommendations["safety_notes"]
        }
    except Exception as e:
        logger.error(f"답변 처리 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class MemoryMonitor:
    def __init__(self, threshold_mb=1024):  # 1GB
        self.threshold = threshold_mb * 1024 * 1024  # MB to bytes

    def check_memory_usage(self):
        """메모리 사용량 체크"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # 메모리 사용량 로깅
        current_usage = memory_info.rss
        logger.debug(f"Current memory usage: {current_usage / 1024 / 1024:.2f} MB")
        
        # 임계값 초과 시 경고
        if current_usage > self.threshold:
            logger.warning(f"Memory usage ({current_usage / 1024 / 1024:.2f} MB) exceeds threshold ({self.threshold / 1024 / 1024:.2f} MB)")
            gc.collect()  # 가비지 컬렉션 실행

if __name__ == "__main__":
    config = load_config()
    uvicorn.run(
        "main:kindhabit_app",
        host=config['fastapi']['server_host'],
        port=config['fastapi']['server_port'],
        log_level=config['fastapi']['log_level']
    )