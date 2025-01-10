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
    "basic_info": {
        "age": 45,
        "gender": "male",
        "height": 170.2,
        "weight": 66.8,
        "blood_type": "A"
    },
    "vital_signs": {
        "blood_pressure_systolic": 135,
        "blood_pressure_diastolic": 79,
        "heart_rate": 75
    },
    "blood_test": {
        "glucose_fasting": 103,
        "total_cholesterol": 328,
        "hdl_cholesterol": 65,
        "ldl_cholesterol": 219,
        "triglycerides": 218,
        "hemoglobin": 14.0,
        "hematocrit": 42.0,
        "alt": 20,
        "ast": 18,
        "creatinine": 1.1,
        "gfr": 76
    },
    "lifestyle": {
        "smoking": False,
        "alcohol_consumption": True,
        "exercise_frequency": 2,
        "sleep_hours": 7.0,
        "stress_level": 3
    },
    "medical_history": {
        "chronic_conditions": [
            "고콜레스테롤",
            "경계성 고혈압"
        ],
        "medications": [
            "없음"
        ],
        "allergies": [],
        "family_history": [],
        "additional_info": {
            "waist_circumference": 86,
            "bmi": 23.1,
            "cancer_screening": {
                "type": "위암",
                "result": "기타(식도염)",
                "recommendation": "위내시경검사에서 역류성 식도염이 있습니다. 역류성 식도염은 위산과 같은 위속 물질이 식도로 역류하여 식도 점막에 만성 염증을 일으킨 것으로 흡연, 음주, 커피, 야식과 복부 비만 등이 주된 원인입니다."
            }
        }
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
    logger.info("\n>>> 건강 데이터 분석 요청 테스트")
    
    async with aiohttp.ClientSession() as session:
        data = {
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
            "endoscopy_results": {
                "colonoscopy": {
                    "date": "2023-12-15",
                    "findings": ["용종 2개 (5mm, 3mm) - 완전절제"],
                    "recommendations": "5년 후 재검사 권고"
                },
                "gastroscopy": {
                    "date": "2023-12-15",
                    "findings": [
                        "만성 위염",
                        "헬리코박터 파일로리 양성"
                    ],
                    "recommendations": "헬리코박터 제균 치료 필요"
                }
            },
            # "symptoms": [
            #     "fatigue",
            #     "joint_pain",
            #     "muscle_weakness",
            #     "dry_skin"
            # ],
            # "current_medications": [
            #     "vitamin_d3_1000iu",
            #     "omega_3_1000mg",
            #     "calcium_500mg"
            # ],
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
        
        async with session.post(f"{BASE_URL}/api/analyze-request/", json=data) as response:
            logger.info(f"Status: {response.status}")
            result = await response.json()
            logger.info(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return result

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

async def test_supplement_interaction():
    """영양제 상호작용 분석 테스트"""
    logger.info("\n>>> 영양제 상호작용 분석 테스트")
    timeout = aiohttp.ClientTimeout(total=300)  # 5분 타임아웃
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            # 1. 건강 상태 데이터 전송
            logger.info("\n1. 건강 상태 데이터 전송")
            request_data = sample_health_data
            
            async with session.post(f"{BASE_URL}/api/supplements/analyze", json=request_data) as response:
                result = await response.json()
                logger.info(f"Status: {response.status}")
                logger.info(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
                
                # 2. 1차 추천 결과 확인
                logger.info("\n2. 1차 추천 결과")
                recommendations = result.get("recommendations", [])
                logger.info(f"추천된 영양제: {json.dumps(recommendations, indent=2, ensure_ascii=False)}")
                
                # 3. 상호작용 분석 요청
                logger.info("\n3. 상호작용 분석 요청")
                interaction_data = {
                    "recommendations": {
                        "vitamin_d3_1000iu": [],
                        "omega_3_1000mg": [],
                        "calcium_500mg": [],
                        "magnesium_400mg": [],
                        "vitamin_e_400iu": [],
                        "iron_65mg": [],
                        "zinc_50mg": []
                    }
                }
                
                async with session.post(
                    f"{BASE_URL}/api/supplements/detailed-analysis",
                    json={
                        "health_data": request_data,
                        "initial_recommendations": interaction_data
                    }
                ) as response:
                    result = await response.json()
                    logger.info(f"Status: {response.status}")
                    logger.info(f"전체 응답 데이터:")
                    logger.info(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
                    
                    # 4. 상호작용 분석 결과 확인
                    logger.info("\n4. 상호작용 분석 결과")
                    if result.get("has_interactions"):
                        logger.info("상호작용이 발견되었습니다:")
                        logger.info(f"상호작용 상세: {json.dumps(result.get('interactions', []), indent=2, ensure_ascii=False)}")
                        logger.info(f"추가 질문: {json.dumps(result.get('questions', []), indent=2, ensure_ascii=False)}")
                    else:
                        logger.info("상호작용이 발견되지 않았습니다.")
                    
                    return result
                    
        except Exception as e:
            logger.error(f"Error during supplement interaction test: {str(e)}")
            return None

async def main():
    """메인 함수"""
    logger.info("=== API 테스트 시작 ===")
    
    # 각 테스트 실행
    await test_root()
    await test_health_categories()
    await test_analyze_request()
    interaction_result = await test_supplement_interaction()
    
    # 최종 결과 출력
    if interaction_result:
        logger.info("\n=== 영양제 상호작용 분석 최종 결과 ===")
        logger.info(json.dumps(interaction_result, indent=2, ensure_ascii=False))
    
    logger.info("\n=== API 테스트 완료 ===")

if __name__ == "__main__":
    asyncio.run(main()) 