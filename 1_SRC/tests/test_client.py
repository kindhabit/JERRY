import asyncio
import aiohttp
import json
from datetime import datetime
from utils.logger_config import PrettyLogger

# 로깅 설정
logger = PrettyLogger('test_client')

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
    logger.info("영양제 상호작용 분석 테스트 시작")
    timeout = aiohttp.ClientTimeout(total=600)  # 10분 타임아웃
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # 1. 건강 상태 데이터 전송
            logger.info("건강 상태 데이터 전송", data=sample_health_data, step="health_data_submit")
            
            # 첫 번째 요청 및 응답 대기
            first_response = await session.post(f"{BASE_URL}/api/supplements/analyze", json=sample_health_data)
            first_response_text = await first_response.text()
            
            logger.info("응답 상태", data={'status': first_response.status}, step="health_data_response")
            
            try:
                analyze_result = json.loads(first_response_text)
                logger.info("분석 응답", data=analyze_result, step="health_data_analysis")
            except json.JSONDecodeError as e:
                logger.error("JSON 파싱 실패", error=e, data=first_response_text)
                return None
            
            # 2. 1차 추천 결과 확인
            recommendations = analyze_result.get("recommendations", [])
            if recommendations:
                logger.info("1차 추천 결과", data=recommendations, step="initial_recommendations")
            else:
                logger.warning("추천 결과 없음", step="initial_recommendations")
            
            # 3. 상호작용 분석 요청
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
            
            logger.info("상호작용 분석 요청", data=interaction_data, step="interaction_request")
            
            # 두 번째 요청 및 응답 대기
            request_data = {
                "health_data": sample_health_data,
                "initial_recommendations": interaction_data
            }
            
            second_response = await session.post(
                f"{BASE_URL}/api/supplements/detailed-analysis",
                json=request_data
            )
            second_response_text = await second_response.text()
            
            logger.info("상호작용 분석 응답 상태", data={'status': second_response.status}, step="interaction_response")
            
            try:
                interaction_result = json.loads(second_response_text)
                logger.info("상호작용 분석 결과", data=interaction_result, step="interaction_analysis")
            except json.JSONDecodeError as e:
                logger.error("JSON 파싱 실패", error=e, data=second_response_text)
                return None
            
            # 4. 상호작용 분석 결과 확인
            if interaction_result.get("has_interactions"):
                logger.info("상호작용 발견", data={
                    'interactions': interaction_result.get('interactions', []),
                    'questions': interaction_result.get('questions', [])
                }, step="interaction_found")
            else:
                logger.info("상호작용 없음", step="interaction_not_found")
            
            await first_response.release()
            await second_response.release()
            return interaction_result
                
    except asyncio.TimeoutError as e:
        logger.error("타임아웃 발생", error=e)
        return None
    except Exception as e:
        logger.error("테스트 중 오류 발생", error=e)
        return None

if __name__ == "__main__":
    asyncio.run(test_supplement_interaction()) 