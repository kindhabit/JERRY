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

logger = setup_logger('rag_service')

class RAGService:
    def __init__(self, chroma_manager: ChromaManager, openai_client: OpenAIClient):
        self.chroma = chroma_manager
        self.llm = openai_client
        self.pattern_learner = PatternService()
        
    async def analyze_with_patterns(self, query: str, context: Dict) -> Dict:
        try:
            # 1. 관련 패턴 검색
            relevant_patterns = await self._find_relevant_patterns(query)
            
            # 2. 컨텍스트 강화
            enhanced_context = self._enhance_context_with_patterns(
                context, 
                relevant_patterns
            )
            
            # 3. LLM 분석
            analysis = await self.llm.analyze_with_context(
                query=query,
                context=enhanced_context
            )
            
            # 4. 패턴 학습 및 결과 강화
            await self.pattern_learner.learn_from_interaction(analysis)
            enhanced_results = await self._enhance_results(
                analysis=analysis,
                patterns=relevant_patterns,
                context=context
            )
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"패턴 기반 분석 실패: {str(e)}")
            raise

    async def get_evidence_based_recommendation(
        self,
        supplement: str,
        health_metric: str
    ) -> Dict:
        """근거 기반 추천 생성"""
        evidence = await self._gather_evidence(supplement, health_metric)
        strength = self._evaluate_evidence_strength(evidence)
        
        return {
            "supplement": supplement,
            "health_metric": health_metric,
            "evidence_summary": self._generate_evidence_summary(evidence),
            "strength": strength,
            "recommendation": self._generate_evidence_based_recommendation(evidence, strength),
            "raw_evidence": evidence
        } 

    async def analyze_complex_interactions(
        self,
        supplements: List[str],
        medications: List[str],
        conditions: List[str]
    ) -> Dict:
        """복합 상호작용 분석"""
        # 1. 직접 상호작용
        direct = await self._analyze_direct_interactions(supplements)
        
        # 2. 약물 상호작용
        medication = await self._analyze_medication_interactions(
            supplements,
            medications
        )
        
        # 3. 건강상태 영향
        condition = await self._analyze_condition_impacts(
            supplements,
            conditions
        )
        
        # 4. 시너지 효과
        synergy = await self._analyze_synergistic_effects(supplements)
        
        return {
            "direct_interactions": direct,
            "medication_interactions": medication,
            "condition_impacts": condition,
            "synergistic_effects": synergy,
            "safety_score": self._calculate_safety_score(
                direct, medication, condition
            )
        } 

    async def _analyze_direct_interactions(self, supplements: List[str]) -> Dict:
        """영양제 간 직접 상호작용 분석"""
        interactions = []
        for pair in itertools.combinations(supplements, 2):
            result = await self.chroma.similarity_search(
                query=f"{pair[0]} {pair[1]} interaction",
                collection_name="interactions",
                n_results=3
            )
            if result["documents"]:
                interactions.append({
                    "pair": pair,
                    "effect": self._analyze_interaction_effect(result),
                    "evidence": result["metadatas"]
                })
        return {"interactions": interactions}

    async def _analyze_medication_interactions(
        self,
        supplements: List[str],
        medications: List[str]
    ) -> Dict:
        """약물-영양제 상호작용 분석"""
        med_interactions = []
        for supp in supplements:
            for med in medications:
                result = await self.chroma.similarity_search(
                    query=f"{supp} {med} drug interaction",
                    collection_name="interactions",
                    n_results=3
                )
                if result["documents"]:
                    med_interactions.append({
                        "supplement": supp,
                        "medication": med,
                        "effect": self._analyze_interaction_effect(result),
                        "severity": self._assess_interaction_severity(result)
                    })
        return {"medication_interactions": med_interactions} 

    async def create_enhanced_embeddings(self, texts: List[str]) -> List[Dict]:
        """향상된 임베딩 생성"""
        try:
            # 1. 기본 임베딩 생성
            embeddings = await self.chroma.embeddings.embed_documents(texts)
            
            # 2. 메타데이터 추가
            enhanced_embeddings = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                enhanced_embeddings.append({
                    "text": text,
                    "embedding": embedding,
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "source": "enhanced_rag",
                        "version": "1.0"
                    }
                })
            
            return enhanced_embeddings
            
        except Exception as e:
            logger.error(f"임베딩 생성 중 오류: {str(e)}")
            raise 