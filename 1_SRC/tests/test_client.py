import asyncio
import aiohttp
import json
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 테스트 설정
BASE_URL = "http://localhost:8000"

# 테스트용 샘플 데이터
sample_health_data = {
    "user_id": "test_user_001",
    "timestamp": datetime.now().isoformat(),
    "health_metrics": {
        "blood_pressure": {
            "systolic": 135,
            "diastolic": 85
        },
        "heart_rate": 75,
        "blood_sugar": {
            "fasting": 95,
            "post_meal": 145
        },
        "cholesterol": {
            "total": 210,
            "hdl": 45,
            "ldl": 140,
            "triglycerides": 180
        },
        "vitamin_d": 18,
        "omega_3_index": 4.2
    },
    "symptoms": [
        "fatigue",
        "joint_pain",
        "muscle_weakness",
        "dry_skin"
    ],
    "current_medications": [
        "vitamin_d3_1000iu",
        "omega_3_1000mg",
        "calcium_500mg"
    ],
    "lifestyle_factors": {
        "exercise_frequency": "2_times_week",
        "sleep_hours": 6,
        "stress_level": "high",
        "diet_type": "irregular",
        "smoking": "non_smoker",
        "alcohol": "moderate",
        "sun_exposure": "low"
    },
    "medical_history": {
        "conditions": [
            "hypertension_stage1",
            "vitamin_d_deficiency",
            "dyslipidemia"
        ],
        "family_history": [
            "cardiovascular_disease",
            "type2_diabetes"
        ],
        "allergies": []
    }
}

async def test_root():
    """루트 엔드포인트 테스트"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/") as response:
            logger.info(f"Status: {response.status}")
            data = await response.json()
            logger.info(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")

async def test_analyze_request():
    """건강 데이터 분석 요청 테스트"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/api/analyze-request/",
            json=sample_health_data
        ) as response:
            logger.info(f"Status: {response.status}")
            data = await response.json()
            logger.info(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")

async def test_health_categories():
    """건강 카테고리 조회 테스트"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/api/health-categories/") as response:
            logger.info(f"Status: {response.status}")
            try:
                data = await response.json()
            except aiohttp.ContentTypeError:
                text = await response.text()
                logger.error(f"Error response: {text}")
                return
            logger.info(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")

async def test_analyze_interactions():
    """영양제 상호작용 분석 테스트"""
    # 건강 데이터 분석 요청
    async with aiohttp.ClientSession() as session:
        # 1. 먼저 건강 데이터 분석 요청
        async with session.post(
            f"{BASE_URL}/api/analyze-request/",
            json=sample_health_data
        ) as response:
            analysis_request = await response.json()
            analysis_id = analysis_request["analysis_id"]
            
            # 2. 상호작용 분석 요청
            test_recommendations = {
                "vitamin_d3": ["calcium", "magnesium"],
                "omega_3": ["vitamin_e", "aspirin"],
                "calcium": ["vitamin_d3", "iron"]
            }
            
            async with session.post(
                f"{BASE_URL}/api/analyze-health-data/interactions",
                params={"analysis_id": analysis_id},
                json=test_recommendations
            ) as response:
                logger.info(f"Status: {response.status}")
                data = await response.json()
                logger.info(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
                
                # 3. 결과 검증
                if data["status"] == "success":
                    logger.info("\n=== 영양제 상호작용 분석 결과 ===")
                    if data.get("has_interactions"):
                        logger.info("상호작용이 발견되었습니다:")
                        for interaction in data.get("interaction_details", []):
                            logger.info(f"- {interaction.get('message', '')}")
                    else:
                        logger.info("유의미한 상호작용이 발견되지 않았습니다.")

async def main():
    """모든 테스트 실행"""
    logger.info("=== API 테스트 시작 ===")
    
    # 기본 연결 테스트
    logger.info("\n>>> 루트 엔드포인트 테스트")
    await test_root()
    
    # 건강 카테고리 조회 테스트
    logger.info("\n>>> 건강 카테고리 조회 테스트")
    await test_health_categories()
    
    # 건강 데이터 분석 요청 테스트
    logger.info("\n>>> 건강 데이터 분석 요청 테스트")
    await test_analyze_request()
    
    # 상호작용 분석 테스트
    logger.info("\n>>> 영양제 상호작용 분석 테스트")
    await test_analyze_interactions()
    
    logger.info("\n=== API 테스트 완료 ===")

if __name__ == "__main__":
    asyncio.run(main()) 