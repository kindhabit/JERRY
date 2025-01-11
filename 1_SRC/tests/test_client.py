import asyncio
import aiohttp
import json
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.propagate = False  # 중복 로깅 방지

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

async def test_supplement_interaction():
    """영양제 상호작용 분석 테스트"""
    logger.info("\n>>> 영양제 상호작용 분석 테스트")
    timeout = aiohttp.ClientTimeout(total=600)  # 10분 타임아웃
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # 1. 건강 상태 데이터 전송
            logger.info("\n1. 건강 상태 데이터 전송")
            request_data = sample_health_data
            
            # 첫 번째 요청 및 응답 대기
            first_response = await session.post(f"{BASE_URL}/api/supplements/analyze", json=request_data)
            first_response_text = await first_response.text()  # 먼저 텍스트로 받음
            
            logger.info(f"Status: {first_response.status}")
            logger.info(f"Raw Response: {first_response_text}")  # 원본 응답 로깅
            
            try:
                analyze_result = json.loads(first_response_text)
                logger.info("분석 응답 데이터:")
                logger.info(f"{json.dumps(analyze_result, indent=2, ensure_ascii=False)}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패: {str(e)}")
                return None
            
            # 응답 확인
            if first_response.status != 200:
                logger.error(f"첫 번째 요청 실패: {first_response_text}")
                return None
            
            # 2. 1차 추천 결과 확인
            logger.info("\n2. 1차 추천 결과")
            recommendations = analyze_result.get("recommendations", [])
            if recommendations:
                logger.info(f"추천된 영양제: {json.dumps(recommendations, indent=2, ensure_ascii=False)}")
            else:
                logger.warning("추천 결과가 없습니다")
            
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
            
            # 두 번째 요청 및 응답 대기
            second_response = await session.post(
                f"{BASE_URL}/api/supplements/detailed-analysis",
                json={
                    "health_data": request_data,
                    "initial_recommendations": interaction_data
                }
            )
            second_response_text = await second_response.text()
            
            logger.info(f"Status: {second_response.status}")
            logger.info(f"Raw Response: {second_response_text}")
            
            try:
                interaction_result = json.loads(second_response_text)
                logger.info("상호작용 분석 응답 데이터:")
                logger.info(f"{json.dumps(interaction_result, indent=2, ensure_ascii=False)}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패: {str(e)}")
                return None
            
            # 응답 확인
            if second_response.status != 200:
                logger.error(f"두 번째 요청 실패: {second_response_text}")
                return None
            
            # 4. 상호작용 분석 결과 확인
            logger.info("\n4. 상호작용 분석 결과")
            if interaction_result.get("has_interactions"):
                logger.info("상호작용이 발견되었습니다:")
                logger.info(f"상호작용 상세: {json.dumps(interaction_result.get('interactions', []), indent=2, ensure_ascii=False)}")
                logger.info(f"추가 질문: {json.dumps(interaction_result.get('questions', []), indent=2, ensure_ascii=False)}")
            else:
                logger.info("상호작용이 발견되지 않았습니다.")
            
            await first_response.release()
            await second_response.release()
            return interaction_result
                
    except asyncio.TimeoutError as e:
        logger.error(f"Timeout error during supplement interaction test: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error during supplement interaction test: {str(e)}")
        return None

if __name__ == "__main__":
    asyncio.run(test_supplement_interaction()) 