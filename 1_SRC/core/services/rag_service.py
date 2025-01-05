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

logger = setup_logger('rag_service')

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
        required_fields = [
            'health_metrics',
            'symptoms',
            'current_medications',
            'lifestyle_factors'
        ]
        return all(field in health_data for field in required_fields)

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
        search_result = self.chroma_manager.get_supplement_interaction(
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
            response = self.openai_client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            raise Exception(f"상세 분석 생성 중 오류: {str(e)}") 