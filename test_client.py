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

async def test_supplement_interaction():
    """영양제 상호작용 분석 테스트"""
    logger.info("\n>>> 영양제 상호작용 분석 테스트")
    timeout = aiohttp.ClientTimeout(total=600)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # 1. 건강 상태 데이터 전송
            logger.info("\n1. 건강 상태 데이터 전송")
            request_data = sample_health_data
            
            # 첫 번째 요청 및 응답 대기
            first_response = await session.post(f"{BASE_URL}/api/supplements/analyze", json=request_data)
            analyze_result = await first_response.json()
            
            logger.info(f"Status: {first_response.status}")
            logger.info(f"분석 응답 데이터:")
            logger.info(f"{json.dumps(analyze_result, indent=2, ensure_ascii=False)}")
            
            # 응답 확인
            if first_response.status != 200:
                logger.error("첫 번째 요청 실패")
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
            interaction_result = await second_response.json()
            
            logger.info(f"Status: {second_response.status}")
            logger.info(f"상호작용 분석 응답 데이터:")
            logger.info(f"{json.dumps(interaction_result, indent=2, ensure_ascii=False)}")
            
            # 응답 확인
            if second_response.status != 200:
                logger.error("두 번째 요청 실패")
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