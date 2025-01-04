from typing import List, Dict, Set, Optional
from models.supplement import Supplement, HealthEffect, Interaction
from models.health_data import HealthData
from config.config_loader import CONFIG
from core.vector_db.vector_store_manager import ChromaManager
from core.services.rag_service import RAGService
from core.analysis.client_health_analyzer import HealthDataAnalyzer
from utils.logger_config import setup_logger
import logging
import json

logger = setup_logger('health_service')

class HealthService:
    def __init__(self, chroma_manager: ChromaManager):
        self.chroma_manager = chroma_manager
        self.rag_service = RAGService(chroma_manager)
        self.health_analyzer = HealthDataAnalyzer()
        
    async def analyze_health_metrics(self, health_data: HealthData) -> Dict:
        """건강 지표 분석"""
        try:
            # 1. 건강 데이터 상세 분석
            analysis_result = await self.health_analyzer.analyze_health_data(
                health_data.model_dump()
            )
            
            # 2. 1차 추천 (건강 지표별)
            primary_recs = await self._get_primary_recommendations(
                analysis_result["context"]
            )
            
            # 3. 간섭도 분석
            interactions = await self._analyze_interactions(
                recommendations=primary_recs,
                health_data=health_data,
                risk_factors=analysis_result["risk_factors"]
            )
            
            return {
                "recommendations": primary_recs,
                "has_interactions": bool(interactions),
                "interactions": interactions,
                "risk_factors": analysis_result["risk_factors"],
                "context": analysis_result["context"]
            }
            
        except Exception as e:
            logger.error(f"건강 지표 분석 중 오류: {str(e)}")
            raise

    async def generate_interaction_notice(self, analysis: Dict) -> Dict:
        """상호작용 알림 생성"""
        try:
            if not analysis.get("interactions"):
                return {"notice": None}
                
            return {
                "notice": self._create_interaction_notice(
                    analysis["recommendations"],
                    analysis["interactions"]
                )
            }
        except Exception as e:
            logger.error(f"상호작용 알림 생성 중 오류: {str(e)}")
            raise

    async def detailed_interaction_analysis(
        self,
        health_data: HealthData,
        initial_recommendations: Dict
    ) -> Dict:
        """상세 상호작용 분석"""
        try:
            # 1. 상세 컨텍스트 검색
            context = await self._get_detailed_context(
                health_data,
                initial_recommendations
            )
            
            # 2. RAG 기반 상세 분석
            analysis = await self.rag_service.analyze_with_patterns(
                query=self._create_detailed_query(health_data),
                context=context
            )
            
            return {
                "analysis": analysis,
                "evidence": await self._gather_evidence(analysis)
            }
        except Exception as e:
            logger.error(f"상세 분석 중 오류: {str(e)}")
            raise

    def _create_interaction_notice(
        self,
        recommendations: List[str],
        interactions: List[Dict]
    ) -> str:
        return f"""
            건강 상태 개선을 위해 {', '.join(recommendations)}을(를) 추천드립니다.
            
            다만, {interactions[0]['supplements'][0]}와(과) {interactions[0]['supplements'][1]}의 경우
            {interactions[0]['mechanism']} 때문에 주의가 필요합니다.
            
            이러한 영양제들의 간섭도에 대해 몇 가지 여쭤보고 싶은 점이 있습니다.
            답변해 주시면 최적의 복용 조합과 시간을 찾아드리겠습니다.
        """
    
    async def _get_primary_recommendations(self, health_data: Dict) -> List[Dict]:
        """건강 지표별 1차 추천"""
        recommendations = []
        
        for field, values in health_data.items():
            condition = self._analyze_health_condition(field, values)
            if condition:
                supplements = await self.chroma_manager.search_supplements_for_condition(
                    condition=condition,
                    n_results=3
                )
                recommendations.extend(supplements)
        
        return recommendations
    
    async def _analyze_interactions(
        self,
        recommendations: List[Dict],
        health_data: Dict,
        user_profile: Optional[Dict]
    ) -> Dict:
        """간섭도 분석"""
        interactions = {
            "supplement_interactions": [],
            "health_condition_impacts": [],
            "medication_interactions": []
        }
        
        # 1. 영양제 간 상호작용
        for i, supp1 in enumerate(recommendations):
            for supp2 in recommendations[i+1:]:
                interaction = await self.chroma_manager.get_supplement_interaction(
                    supp1['name'],
                    supp2['name']
                )
                if interaction:
                    interactions["supplement_interactions"].append(interaction)
        
        # 2. 건강 상태에 미치는 영향
        for supp in recommendations:
            impacts = await self.chroma_manager.get_health_impacts(
                supplement=supp['name'],
                health_data=health_data
            )
            if impacts:
                interactions["health_condition_impacts"].extend(impacts)
        
        # 3. 약물 상호작용 (사용자 프로필이 있는 경우)
        if user_profile and 'medications' in user_profile:
            for supp in recommendations:
                for med in user_profile['medications']:
                    interaction = await self.chroma_manager.get_medication_interaction(
                        supplement=supp['name'],
                        medication=med
                    )
                    if interaction:
                        interactions["medication_interactions"].append(interaction)
        
        return interactions 