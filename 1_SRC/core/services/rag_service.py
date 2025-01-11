from typing import Dict, List, Optional
import logging
from datetime import datetime
from utils.openai_client import OpenAIClient
from core.services.pattern_service import PatternService
from core.vector_db.vector_store_manager import ChromaManager
import itertools
from models.health_data import HealthData
from config.config_loader import CONFIG
from utils.logger_config import setup_logger
import json
import uuid
from json import JSONEncoder

logger = setup_logger('rag_service')

class DateTimeEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class RAGService:
    def __init__(self, openai_client, chroma_manager):
        self.openai_client = openai_client
        self.chroma_manager = chroma_manager
        self.MIN_CONFIDENCE_THRESHOLD = 0.7

    async def analyze_health_data(self, health_data):
        try:
            # 건강 데이터 유효성 검증
            if not self._validate_health_data(health_data):
                return {
                    "status": "error",
                    "message": "올바르지 않은 건강 데이터 형식입니다."
                }

            # 현재 복용 중인 영양제 추출
            current_supplements = health_data.get('current_medications', [])
            
            # 벡터 스토어를 통한 분석 수행
            analysis_result = await self._perform_analysis(health_data, current_supplements)
            
            if analysis_result['status'] != 'success':
                return analysis_result

            # 결과의 신뢰도 검증
            if not self._validate_confidence(analysis_result):
                return {
                    "status": "low_confidence",
                    "message": "분석 결과의 신뢰도가 낮습니다. 더 많은 데이터가 필요합니다.",
                    "partial_results": analysis_result
                }

            return analysis_result

        except Exception as e:
            return {
                "status": "error",
                "message": "분석 중 오류가 발생했습니다.",
                "error_details": str(e)
            }

    def _validate_health_data(self, health_data):
        """건강 데이터 유효성 검증"""
        required_fields = {
            'basic_info',
            'vital_signs',
            'blood_test',
            'lifestyle',
            'medical_history'
        }
        return all(field in health_data for field in required_fields)

    def _get_current_medications(self, health_data):
        """현재 복용 중인 약물 정보 추출"""
        try:
            return health_data.get('medical_history', {}).get('medications', [])
        except:
            return []

    def _validate_confidence(self, analysis_result):
        if 'data_quality' not in analysis_result:
            return False
            
        confidence = analysis_result['data_quality'].get('confidence_level', 'low')
        return {
            'high': 1.0,
            'medium': 0.8,
            'low': 0.5
        }.get(confidence, 0) >= self.MIN_CONFIDENCE_THRESHOLD

    async def _perform_analysis(self, health_data, current_supplements):
        # 벡터 스토어에서 관련 정보 검색
        search_result = await self.chroma_manager.get_supplement_interaction(
            health_data, 
            current_supplements
        )

        if search_result['status'] != 'success':
            return search_result

        # 검색 결과를 바탕으로 상세 분석 수행
        try:
            detailed_analysis = await self._generate_detailed_analysis(
                health_data,
                search_result
            )
            
            return {
                "status": "success",
                "analysis_id": str(uuid.uuid4()),
                "recommendations": detailed_analysis.get('recommendations', {}),
                "data_quality": detailed_analysis.get('data_quality', {}),
                "context": {
                    "analyzed_symptoms": health_data.get('symptoms', []),
                    "considered_factors": list(health_data.get('lifestyle_factors', {}).keys()),
                    "current_supplements": current_supplements
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "message": "상세 분석 중 오류가 발생했습니다.",
                "error_details": str(e)
            }

    async def _generate_detailed_analysis(self, health_data, search_result):
        prompt = f"""
        다음 건강 데이터와 검색 결과를 바탕으로 상세한 영양제 분석을 수행해주세요:

        건강 데이터:
        {json.dumps(health_data, indent=2, ensure_ascii=False)}

        검색 결과:
        {json.dumps(search_result, indent=2, ensure_ascii=False)}

        다음 기준으로 분석해주세요:
        1. 각 추천의 신뢰도와 근거 명시
        2. 현재 복용 중인 영양제와의 상호작용
        3. 사용자의 건강 상태를 고려한 주의사항
        4. 데이터 부족 시 명확히 표시
        """

        try:
            response = await self.openai_client.chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            
            return json.loads(response['content'])
            
        except Exception as e:
            raise Exception(f"상세 분석 생성 중 오류: {str(e)}")

    async def analyze_with_patterns(self, query: str, context: dict) -> dict:
        """
        패턴 기반 상세 분석을 수행합니다.
        
        Args:
            query: 분석할 쿼리
            context: 분석 컨텍스트
            
        Returns:
            분석 결과를 담은 딕셔너리
        """
        logger.info(f"[분석 시작] 패턴 기반 상세 분석 - 쿼리: {query[:50]}...")
        try:
            # 프롬프트 템플릿 구성
            prompt = f"""
            다음 건강 데이터와 컨텍스트를 바탕으로 상세 분석을 수행하여 정확한 JSON 형식으로 응답해주세요.
            코드 블록이나 마커(```json 등)를 사용하지 말고 순수한 JSON 형식으로만 응답해주세요.

            분석 쿼리: {query}

            컨텍스트 정보:
            {json.dumps(context, indent=2, ensure_ascii=False, cls=DateTimeEncoder)}

            다음 형식의 JSON으로 정확히 응답해주세요:
            {{
                "status": "success",
                "description": "상세한 상호작용 설명",
                "evidence": ["근거1", "근거2"],
                "severity": "high/medium/low",
                "confidence_score": 0.95
            }}

            응답은 반드시 위의 JSON 형식을 따라야 하며, 모든 필드가 포함되어야 합니다.
            """

            logger.info("[API 요청] OpenAI API 호출 시작")
            # OpenAI API 호출
            response = await self.openai_client.chat_completion(
                messages=[
                    {
                        "role": "system", 
                        "content": "당신은 건강 데이터를 분석하고 영양제 상호작용을 평가하는 전문가입니다. 반드시 순수한 JSON 형식으로만 응답해야 합니다. 마크다운 코드 블록이나 다른 포맷팅을 사용하지 마세요."
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            logger.info("[API 응답] OpenAI API 응답 수신 완료")

            # 응답 검증
            if not response or 'content' not in response:
                logger.error("[검증 실패] API 응답이 비어있거나 유효하지 않음")
                return self._create_error_response("유효하지 않은 API 응답")

            try:
                logger.info("[응답 처리] JSON 파싱 및 검증 시작")
                # 응답 정제
                content = response['content']
                # 코드 블록 마커 제거
                content = content.replace('```json', '').replace('```', '').strip()
                logger.debug(f"[응답 정제] 마커 제거 후 길이: {len(content)} 문자")
                
                # JSON 파싱 시도
                result = json.loads(content)
                logger.info("[JSON 파싱] 성공")
                
                # 필수 필드 검증
                required_fields = ['status', 'description', 'evidence', 'severity', 'confidence_score']
                if not all(field in result for field in required_fields):
                    missing_fields = [field for field in required_fields if field not in result]
                    logger.error(f"[필드 검증 실패] 누락된 필드: {missing_fields}")
                    return self._create_error_response(f"필수 필드 누락: {missing_fields}")

                logger.info(f"[분석 완료] 상태: {result['status']}, 심각도: {result['severity']}")
                return result

            except json.JSONDecodeError as e:
                logger.error(f"[JSON 파싱 오류] 원인: {str(e)}")
                logger.debug(f"[JSON 파싱 오류] 정제된 응답: {content[:200]}...")
                return self._create_error_response("JSON 파싱 오류")

        except Exception as e:
            logger.error(f"[처리 실패] 패턴 기반 분석 중 오류 발생: {str(e)}")
            return self._create_error_response(str(e))

    def _create_error_response(self, error_message: str) -> dict:
        """
        표준화된 에러 응답을 생성합니다.
        """
        return {
            "status": "error",
            "description": f"분석 중 오류 발생: {error_message}",
            "evidence": [],
            "severity": "unknown",
            "confidence_score": 0.0
        } 