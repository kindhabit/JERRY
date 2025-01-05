from typing import Dict, List, Optional
from datetime import datetime
from models.session import AnalysisResult, Recommendation, InteractionWarning, Question
from core.analysis.client_health_analyzer import HealthDataAnalyzer
from core.vector_db.vector_store_manager import ChromaManager
from utils.logger_config import setup_logger

logger = setup_logger('health_analyzer')

class EnhancedHealthAnalyzer:
    def __init__(self, chroma_manager: ChromaManager):
        self.base_analyzer = HealthDataAnalyzer()
        self.chroma = chroma_manager
        self.logger = logger

    async def analyze(self, health_data: Dict) -> AnalysisResult:
        """건강 데이터 종합 분석을 수행합니다."""
        try:
            # 1. 기본 건강 데이터 분석
            base_analysis = await self.base_analyzer.analyze_health_data(health_data)
            
            # 2. 위험 요인 기반 검색
            risk_evidence = await self._search_risk_evidence(
                base_analysis["risk_factors"]
            )
            
            # 3. 건강 상태 기반 추천 검색
            recommendations = await self._search_recommendations(
                base_analysis["context"]
            )
            
            # 4. 상호작용 분석
            interaction_warnings = await self._analyze_interactions(
                recommendations,
                health_data
            )
            
            # 5. 생활습관 제안 생성
            lifestyle_suggestions = await self._generate_lifestyle_suggestions(
                health_data,
                base_analysis
            )
            
            # 6. 신뢰도 점수 계산
            confidence_levels = self._calculate_confidence_levels(
                recommendations,
                risk_evidence
            )
            
            # 7. 필요한 추가 확인사항 도출
            required_checks = self._determine_required_checks(
                health_data,
                interaction_warnings
            )
            
            # 8. 분석 결과 통합
            return AnalysisResult(
                primary_concerns=base_analysis["risk_factors"],
                recommendations=[
                    Recommendation(
                        type=rec["type"],
                        name=rec["name"],
                        confidence=rec["confidence"],
                        reason=rec["reason"],
                        evidence=rec.get("evidence")
                    ) for rec in recommendations
                ],
                lifestyle_suggestions=lifestyle_suggestions,
                interaction_warnings=[
                    InteractionWarning(
                        source=warning["source"],
                        target=warning["target"],
                        severity=warning["severity"],
                        description=warning["description"],
                        evidence=warning.get("evidence")
                    ) for warning in interaction_warnings
                ],
                required_checks=required_checks,
                evidence=risk_evidence,
                confidence_levels=confidence_levels
            )
            
        except Exception as e:
            self.logger.error(f"건강 데이터 분석 중 오류: {str(e)}")
            raise

    async def _search_risk_evidence(self, risk_factors: List[Dict]) -> List[Dict]:
        """위험 요인 관련 근거 데이터를 검색합니다."""
        evidence = []
        for risk in risk_factors:
            # 위험 요인별 관련 논문 검색
            results = await self.chroma.similarity_search(
                query=f"{risk['type']} health risk",
                collection_name="health_data",
                n_results=3
            )
            
            if results["documents"]:
                evidence.append({
                    "risk_type": risk["type"],
                    "papers": results["metadatas"],
                    "relevance_score": results["distances"][0]
                })
        
        return evidence

    async def _search_recommendations(self, health_context: Dict) -> List[Dict]:
        """건강 상태 기반 추천 사항을 검색합니다."""
        recommendations = []
        
        # 1. 기본 건강 상태 기반 검색
        basic_results = await self.chroma.similarity_search(
            query=self._build_health_query(health_context),
            collection_name="supplements",
            n_results=5
        )
        
        if basic_results["documents"]:
            for doc, meta, score in zip(
                basic_results["documents"],
                basic_results["metadatas"],
                basic_results["distances"]
            ):
                recommendations.append({
                    "type": "supplement",
                    "name": meta.get("name"),
                    "evidence": doc,
                    "confidence": score,
                    "metadata": meta
                })
        
        # 2. 생활습관 기반 추천
        if health_context.get("lifestyle"):
            lifestyle_results = await self.chroma.similarity_search(
                query=self._build_lifestyle_query(health_context["lifestyle"]),
                collection_name="health_data",
                n_results=3
            )
            
            if lifestyle_results["documents"]:
                recommendations.extend([
                    {
                        "type": "lifestyle",
                        "recommendation": doc,
                        "evidence": meta,
                        "confidence": score
                    }
                    for doc, meta, score in zip(
                        lifestyle_results["documents"],
                        lifestyle_results["metadatas"],
                        lifestyle_results["distances"]
                    )
                ])
        
        return recommendations

    async def _analyze_interactions(
        self,
        recommendations: List[Dict],
        health_data: Dict
    ) -> List[Dict]:
        """추천사항과 현재 건강상태/약물 간의 상호작용을 분석합니다."""
        warnings = []
        
        # 1. 현재 복용 중인 약물 확인
        current_medications = health_data.get("medical_history", {}).get("medications", [])
        
        # 2. 만성질환 확인
        chronic_conditions = health_data.get("medical_history", {}).get("chronic_conditions", [])
        
        for rec in recommendations:
            # 약물 상호작용 검색
            for med in current_medications:
                results = await self.chroma.similarity_search(
                    query=f"{rec['name']} interaction with {med}",
                    collection_name="interactions",
                    n_results=2
                )
                
                if results["documents"]:
                    warnings.append({
                        "source": f"medication_{med}",
                        "target": rec["name"],
                        "severity": "high" if results["distances"][0] > 0.8 else "medium",
                        "description": f"{rec['name']}과(와) {med} 간의 상호작용 가능성이 있습니다.",
                        "evidence": results["metadatas"]
                    })
            
            # 건강상태 관련 주의사항 검색
            for condition in chronic_conditions:
                results = await self.chroma.similarity_search(
                    query=f"{rec['name']} risks with {condition}",
                    collection_name="health_data",
                    n_results=2
                )
                
                if results["documents"]:
                    warnings.append({
                        "source": f"condition_{condition}",
                        "target": rec["name"],
                        "severity": "high" if results["distances"][0] > 0.8 else "medium",
                        "description": f"{condition} 환자의 경우 {rec['name']} 복용 시 주의가 필요합니다.",
                        "evidence": results["metadatas"]
                    })
        
        return warnings

    async def _generate_lifestyle_suggestions(
        self,
        health_data: Dict,
        base_analysis: Dict
    ) -> List[Dict]:
        """생활습관 개선 제안을 생성합니다."""
        suggestions = []
        lifestyle = health_data.get("lifestyle", {})
        
        # 운동 관련 제안
        if lifestyle.get("exercise_frequency", 0) < 3:
            suggestions.append({
                "type": "exercise",
                "suggestion": "일주일에 3회 이상의 중간 강도 운동을 권장합니다.",
                "priority": "high",
                "reason": "운동부족은 전반적인 건강 위험을 증가시킬 수 있습니다."
            })
        
        # 수면 관련 제안
        if lifestyle.get("sleep_hours", 0) < 7:
            suggestions.append({
                "type": "sleep",
                "suggestion": "하루 7-8시간의 수면을 취하도록 노력해주세요.",
                "priority": "medium",
                "reason": "충분한 수면은 건강 유지에 필수적입니다."
            })
        
        # 스트레스 관련 제안
        if lifestyle.get("stress_level", 0) > 3:
            suggestions.append({
                "type": "stress",
                "suggestion": "스트레스 관리를 위한 명상이나 가벼운 운동을 추천드립니다.",
                "priority": "medium",
                "reason": "지속적인 스트레스는 건강에 부정적인 영향을 미칠 수 있습니다."
            })
        
        return suggestions

    def _calculate_confidence_levels(
        self,
        recommendations: List[Dict],
        evidence: List[Dict]
    ) -> Dict[str, float]:
        """추천사항별 신뢰도 점수를 계산합니다."""
        confidence_levels = {}
        
        for rec in recommendations:
            # 기본 신뢰도
            base_confidence = rec.get("confidence", 0.5)
            
            # 증거 기반 신뢰도 조정
            evidence_boost = 0.0
            relevant_evidence = [e for e in evidence if e.get("type") == rec.get("type")]
            
            if relevant_evidence:
                evidence_boost = sum(
                    e.get("relevance_score", 0.0) for e in relevant_evidence
                ) / len(relevant_evidence)
            
            # 최종 신뢰도 계산
            final_confidence = min(0.95, base_confidence + (evidence_boost * 0.2))
            confidence_levels[rec["name"]] = round(final_confidence, 2)
        
        return confidence_levels

    def _determine_required_checks(
        self,
        health_data: Dict,
        interaction_warnings: List[Dict]
    ) -> List[str]:
        """추가 확인이 필요한 사항들을 결정합니다."""
        required_checks = []
        
        # 상호작용 경고 기반 체크항목
        for warning in interaction_warnings:
            if warning["severity"] == "high":
                required_checks.append(
                    f"{warning['target']} 복용 전 {warning['source']} 관련 전문의 상담 필요"
                )
        
        # 건강 상태 기반 체크항목
        medical_history = health_data.get("medical_history", {})
        if medical_history.get("chronic_conditions"):
            required_checks.append(
                "만성질환 관리 상태 확인 필요"
            )
        
        # 검사 결과 기반 체크항목
        blood_test = health_data.get("blood_test", {})
        if blood_test.get("glucose_fasting", 0) > 100:
            required_checks.append(
                "공복혈당 관련 추가 검사 권장"
            )
        
        return required_checks

    def _build_health_query(self, context: Dict) -> str:
        """건강 상태 기반 검색 쿼리를 생성합니다."""
        query_parts = []
        
        if context.get("risk_factors"):
            risk_types = [r["type"] for r in context["risk_factors"]]
            query_parts.append(f"health conditions: {', '.join(risk_types)}")
        
        if context.get("basic_info"):
            basic = context["basic_info"]
            query_parts.append(
                f"age: {basic.get('age', 'unknown')}, "
                f"gender: {basic.get('gender', 'unknown')}"
            )
        
        return " AND ".join(query_parts)

    def _build_lifestyle_query(self, lifestyle: Dict) -> str:
        """생활습관 기반 검색 쿼리를 생성합니다."""
        query_parts = []
        
        if lifestyle.get("smoking"):
            query_parts.append("smoking")
        if lifestyle.get("alcohol"):
            query_parts.append("alcohol consumption")
        if lifestyle.get("exercise_frequency") is not None:
            query_parts.append(
                f"exercise frequency: {lifestyle['exercise_frequency']} times per week"
            )
        
        return " AND ".join(query_parts)

    def _generate_risk_question(self, risk: Dict) -> str:
        """위험 요인 관련 질문을 생성합니다."""
        templates = {
            "high_cholesterol": "콜레스테롤 수치가 높게 나왔습니다. 현재 복용 중인 약물이 있으신가요?",
            "hypertension": "혈압이 높게 측정되었습니다. 평소 혈압 관리를 어떻게 하고 계신가요?",
            "liver_function_abnormal": "간 수치가 정상 범위를 벗어났습니다. 최근 음주나 약물 복용이 있으셨나요?",
            "obesity": "체질량지수가 높게 나왔습니다. 평소 운동은 얼마나 하시나요?",
            "sedentary_lifestyle": "운동량이 부족한 것 같습니다. 운동하기 어려운 특별한 이유가 있으신가요?"
        }
        
        return templates.get(
            risk["type"],
            f"{risk['type']} 관련하여 특별한 증상이나 불편함을 느끼시나요?"
        )

    def _generate_supplement_question(self, recommendation: Dict) -> str:
        """영양제 관련 질문을 생성합니다."""
        return f"{recommendation['name']}을(를) 이전에 복용해보신 적이 있으신가요? 있다면 어떤 효과를 보셨나요?"

    def _generate_lifestyle_questions(self, lifestyle: Dict) -> List[Question]:
        """생활습관 관련 질문을 생성합니다."""
        questions = []
        
        if lifestyle.get("smoking"):
            questions.append(Question(
                id="lifestyle_smoking",
                text="금연을 시도해보신 적이 있으신가요?",
                context="흡연 여부: True",
                priority=2
            ))
        
        if lifestyle.get("alcohol"):
            questions.append(Question(
                id="lifestyle_alcohol",
                text="평소 음주량과 빈도는 어떻게 되시나요?",
                context="음주 여부: True",
                priority=1
            ))
        
        if lifestyle.get("exercise_frequency", 0) < 3:
            questions.append(Question(
                id="lifestyle_exercise",
                text="규칙적인 운동을 하기 어려운 특별한 이유가 있으신가요?",
                context=f"운동 빈도: {lifestyle.get('exercise_frequency')}회/주",
                priority=1
            ))
        
        return questions 