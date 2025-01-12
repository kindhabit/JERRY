from typing import Dict, List, Optional
from datetime import datetime
from utils.logger_config import get_logger
from core.vector_db.vector_store_manager import ChromaManager
from models.health_data import HealthData
from config.config_loader import CONFIG
import json

logger = get_logger('pattern_service')

class PatternService:
    def __init__(self):
        self.patterns = {
            "supplement_interactions": {},
            "health_conditions": {},
            "medication_interactions": {},
            "temporal_patterns": {}
        }
        self.feedback_history = []
        self.confidence_threshold = 0.7

    async def initialize(self):
        """패턴 학습 시스템 초기화"""
        self.patterns = {
            "supplement_interactions": {},
            "health_conditions": {},
            "medication_interactions": {},
            "temporal_patterns": {}
        }
        
    async def learn_from_interaction(self, data: Dict):
        """새로운 상호작용에서 패턴 학습"""
        try:
            # 1. 패턴 추출
            pattern = await self._extract_pattern(data)
            
            # 2. 유사 패턴 검색
            similar_patterns = self._find_similar_patterns(pattern)
            
            # 3. 패턴 강화 또는 새로운 패턴 추가
            if similar_patterns:
                await self._strengthen_pattern(similar_patterns[0], pattern)
            else:
                await self._add_new_pattern(pattern)
                
            # 4. 피드백 기록
            self._record_feedback(pattern)
            
        except Exception as e:
            logger.error(f"패턴 학습 실패: {str(e)}")
            raise

    async def _extract_pattern(self, data: Dict) -> Dict:
        """데이터에서 패턴 추출"""
        return {
            "type": data.get("interaction_type"),
            "entities": data.get("entities", []),
            "effect": data.get("effect"),
            "confidence": data.get("confidence", 0.0),
            "context": data.get("context", {}),
            "timestamp": datetime.now().isoformat()
        }

    async def _strengthen_pattern(self, existing: Dict, new: Dict):
        """기존 패턴 강화"""
        # 1. 신뢰도 업데이트
        existing["confidence"] = (
            existing["confidence"] * existing["frequency"] + new["confidence"]
        ) / (existing["frequency"] + 1)
        
        # 2. 빈도 증가
        existing["frequency"] += 1
        
        # 3. 컨텍스트 병합
        existing["context"] = self._merge_contexts(
            existing["context"], 
            new["context"]
        )

    async def _add_new_pattern(self, pattern: Dict):
        """새로운 패턴 추가"""
        pattern_type = pattern["type"]
        pattern_id = f"{pattern_type}_{len(self.patterns[pattern_type])}"
        
        self.patterns[pattern_type][pattern_id] = {
            **pattern,
            "frequency": 1,
            "last_updated": datetime.now().isoformat(),
            "related_patterns": []
        }

    def _find_similar_patterns(self, pattern: Dict) -> List[Dict]:
        """유사한 패턴 검색"""
        pattern_type = pattern["type"]
        similar_patterns = []
        
        for existing_pattern in self.patterns[pattern_type].values():
            similarity_score = self._calculate_similarity(
                pattern,
                existing_pattern
            )
            if similarity_score >= self.confidence_threshold:
                similar_patterns.append(existing_pattern)
        
        return sorted(
            similar_patterns,
            key=lambda x: x["confidence"],
            reverse=True
        )

    def _calculate_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """패턴 간 유사도 계산"""
        # 1. 엔티티 유사도
        entity_similarity = len(
            set(pattern1["entities"]) & set(pattern2["entities"])
        ) / len(set(pattern1["entities"]) | set(pattern2["entities"]))
        
        # 2. 효과 유사도
        effect_similarity = 1.0 if pattern1["effect"] == pattern2["effect"] else 0.0
        
        # 3. 컨텍스트 유사도
        context_similarity = self._calculate_context_similarity(
            pattern1.get("context", {}),
            pattern2.get("context", {})
        )
        
        # 가중치 적용
        weights = {"entity": 0.4, "effect": 0.4, "context": 0.2}
        final_similarity = (
            weights["entity"] * entity_similarity +
            weights["effect"] * effect_similarity +
            weights["context"] * context_similarity
        )
        
        return final_similarity

    def _calculate_context_similarity(self, context1: Dict, context2: Dict) -> float:
        """컨텍스트 유사도 계산"""
        all_keys = set(context1.keys()) | set(context2.keys())
        if not all_keys:
            return 0.0
            
        matching_keys = set(context1.keys()) & set(context2.keys())
        matching_values = sum(
            1 for k in matching_keys if context1[k] == context2[k]
        )
        
        return matching_values / len(all_keys)

    def _merge_contexts(self, context1: Dict, context2: Dict) -> Dict:
        """컨텍스트 병합"""
        merged = context1.copy()
        
        for key, value in context2.items():
            if key in merged:
                if isinstance(merged[key], list):
                    merged[key] = list(set(merged[key] + [value]))
                else:
                    merged[key] = [merged[key], value]
            else:
                merged[key] = value
                
        return merged

    def _record_feedback(self, pattern: Dict):
        """피드백 기록"""
        self.feedback_history.append({
            "pattern": pattern,
            "timestamp": datetime.now().isoformat(),
            "action": "new" if pattern.get("is_new") else "strengthen"
        })

    async def analyze_patterns(self, query_data: Dict) -> Dict:
        """패턴 기반 분석 수행"""
        try:
            # 1. 관련 패턴 검색
            relevant_patterns = self._find_relevant_patterns(query_data)
            
            # 2. 패턴 적용
            analysis_results = self._apply_patterns(
                query_data,
                relevant_patterns
            )
            
            # 3. 신뢰도 평가
            confidence_scores = self._evaluate_confidence(analysis_results)
            
            return {
                "results": analysis_results,
                "confidence": confidence_scores,
                "patterns_used": len(relevant_patterns)
            }
            
        except Exception as e:
            logger.error(f"패턴 분석 실패: {str(e)}")
            raise