from typing import Dict, List, Optional
from models.session import Session, Answer, Question, AnalysisResult
from core.vector_db.vector_store_manager import ChromaManager
from utils.logger_config import setup_logger

logger = setup_logger('answer_processor')

class AnswerProcessor:
    def __init__(self, chroma_manager: ChromaManager):
        self.chroma = chroma_manager
        self.logger = logger

    async def process_answer(
        self,
        session: Session,
        answer: Answer
    ) -> Optional[AnalysisResult]:
        """사용자 답변을 처리하고 분석 결과를 업데이트합니다."""
        try:
            # 1. 답변 컨텍스트 수집
            context = self._build_answer_context(session, answer)
            
            # 2. 답변 분석
            analysis = await self._analyze_answer(answer, context)
            
            # 3. 기존 분석 결과 업데이트
            updated_result = await self._update_analysis_result(
                session.analysis_results,
                analysis,
                context
            )
            
            # 4. 추가 증거 검색
            evidence = await self._search_additional_evidence(
                analysis,
                context
            )
            
            # 5. 최종 결과 통합
            if updated_result:
                updated_result.evidence.extend(evidence)
            
            return updated_result
            
        except Exception as e:
            self.logger.error(f"답변 처리 중 오류: {str(e)}")
            raise

    def _build_answer_context(self, session: Session, answer: Answer) -> Dict:
        """답변 처리를 위한 컨텍스트를 구축합니다."""
        # 관련 질문 찾기
        question = next(
            (q for q in session.current_questions if q.id == answer.question_id),
            None
        )
        
        context = {
            "session_status": session.status,
            "current_step": session.current_step,
            "question_context": question.context if question else None,
            "previous_answers": [
                a.dict() for a in session.answers
                if a.question_id != answer.question_id
            ]
        }
        
        if session.analysis_results:
            context["current_analysis"] = session.analysis_results.dict()
        
        return context

    async def _analyze_answer(self, answer: Answer, context: Dict) -> Dict:
        """답변 내용을 분석합니다."""
        analysis = {
            "answer_type": self._determine_answer_type(answer, context),
            "relevant_factors": [],
            "confidence": 0.0
        }
        
        # 답변 유형별 분석
        if analysis["answer_type"] == "health_risk":
            analysis.update(
                await self._analyze_health_risk_answer(answer, context)
            )
        elif analysis["answer_type"] == "lifestyle":
            analysis.update(
                await self._analyze_lifestyle_answer(answer, context)
            )
        elif analysis["answer_type"] == "medication":
            analysis.update(
                await self._analyze_medication_answer(answer, context)
            )
        
        return analysis

    async def _update_analysis_result(
        self,
        current_result: Optional[AnalysisResult],
        analysis: Dict,
        context: Dict
    ) -> Optional[AnalysisResult]:
        """기존 분석 결과를 새로운 답변 분석 결과로 업데이트합니다."""
        if not current_result:
            return None
            
        updated_result = current_result.copy(deep=True)
        
        # 위험 요인 업데이트
        if analysis["answer_type"] == "health_risk":
            self._update_health_risks(
                updated_result.primary_concerns,
                analysis["relevant_factors"]
            )
        
        # 추천 사항 업데이트
        if analysis.get("recommendations"):
            self._update_recommendations(
                updated_result.recommendations,
                analysis["recommendations"]
            )
        
        return updated_result

    async def _search_additional_evidence(
        self,
        analysis: Dict,
        context: Dict
    ) -> List[Dict]:
        """답변 분석 결과에 기반하여 추가 증거를 검색합니다."""
        evidence = []
        
        # 관련 요인별 증거 검색
        for factor in analysis.get("relevant_factors", []):
            results = await self.chroma.similarity_search(
                query=self._build_evidence_query(factor),
                collection_name="health_data",
                n_results=2
            )
            
            if results["documents"]:
                evidence.append({
                    "factor": factor["type"],
                    "papers": results["metadatas"],
                    "relevance": results["distances"][0]
                })
        
        return evidence

    def _determine_answer_type(self, answer: Answer, context: Dict) -> str:
        """답변의 유형을 결정합니다."""
        question_context = context.get("question_context", "")
        
        if question_context.startswith("health_risk_"):
            return "health_risk"
        elif question_context.startswith("lifestyle_"):
            return "lifestyle"
        elif question_context.startswith("medication_"):
            return "medication"
        else:
            return "general"

    async def _analyze_health_risk_answer(
        self,
        answer: Answer,
        context: Dict
    ) -> Dict:
        """건강 위험 관련 답변을 분석합니다."""
        risk_type = context.get("question_context", "").replace("health_risk_", "")
        
        # 위험 요인별 키워드 검색
        results = await self.chroma.similarity_search(
            query=f"{risk_type} {answer.answer_text}",
            collection_name="health_data",
            n_results=2
        )
        
        analysis = {
            "risk_type": risk_type,
            "relevant_factors": [],
            "confidence": 0.0
        }
        
        if results["documents"]:
            analysis["relevant_factors"].append({
                "type": risk_type,
                "evidence": results["metadatas"],
                "confidence": results["distances"][0]
            })
            analysis["confidence"] = results["distances"][0]
        
        return analysis

    async def _analyze_lifestyle_answer(
        self,
        answer: Answer,
        context: Dict
    ) -> Dict:
        """생활습관 관련 답변을 분석합니다."""
        lifestyle_type = context.get("question_context", "").replace("lifestyle_", "")
        
        # 생활습관 관련 키워드 검색
        results = await self.chroma.similarity_search(
            query=f"{lifestyle_type} lifestyle {answer.answer_text}",
            collection_name="health_data",
            n_results=2
        )
        
        analysis = {
            "lifestyle_type": lifestyle_type,
            "relevant_factors": [],
            "recommendations": []
        }
        
        if results["documents"]:
            analysis["relevant_factors"].append({
                "type": f"{lifestyle_type}_lifestyle",
                "evidence": results["metadatas"],
                "confidence": results["distances"][0]
            })
            
            # 생활습관 개선 추천 사항 추가
            analysis["recommendations"].append({
                "type": "lifestyle_change",
                "target": lifestyle_type,
                "suggestion": self._generate_lifestyle_suggestion(
                    lifestyle_type,
                    answer.answer_text
                )
            })
        
        return analysis

    async def _analyze_medication_answer(
        self,
        answer: Answer,
        context: Dict
    ) -> Dict:
        """약물/보조제 관련 답변을 분석합니다."""
        med_type = context.get("question_context", "").split("_")[0]
        med_name = "_".join(context.get("question_context", "").split("_")[1:])
        
        # 약물/보조제 상호작용 검색
        results = await self.chroma.similarity_search(
            query=f"{med_name} interaction effects",
            collection_name="interactions",
            n_results=2
        )
        
        analysis = {
            "medication_type": med_type,
            "medication_name": med_name,
            "relevant_factors": [],
            "recommendations": []
        }
        
        if results["documents"]:
            analysis["relevant_factors"].append({
                "type": "medication_interaction",
                "evidence": results["metadatas"],
                "confidence": results["distances"][0]
            })
        
        return analysis

    def _update_health_risks(
        self,
        current_risks: List[Dict],
        new_factors: List[Dict]
    ) -> None:
        """건강 위험 요인을 업데이트합니다."""
        for factor in new_factors:
            # 기존 위험 요인 찾기
            existing = next(
                (r for r in current_risks if r["type"] == factor["type"]),
                None
            )
            
            if existing:
                # 기존 위험 요인 업데이트
                existing.update(factor)
            else:
                # 새로운 위험 요인 추가
                current_risks.append(factor)

    def _update_recommendations(
        self,
        current_recs: List[Dict],
        new_recs: List[Dict]
    ) -> None:
        """추천 사항을 업데이트합니다."""
        for rec in new_recs:
            # 기존 추천 사항 찾기
            existing = next(
                (
                    r for r in current_recs
                    if r["type"] == rec["type"] and r.get("target") == rec.get("target")
                ),
                None
            )
            
            if existing:
                # 기존 추천 사항 업데이트
                existing.update(rec)
            else:
                # 새로운 추천 사항 추가
                current_recs.append(rec)

    def _build_evidence_query(self, factor: Dict) -> str:
        """증거 검색을 위한 쿼리를 생성합니다."""
        return f"{factor['type']} evidence research papers"

    def _generate_lifestyle_suggestion(
        self,
        lifestyle_type: str,
        answer_text: str
    ) -> str:
        """생활습관 개선 제안을 생성합니다."""
        templates = {
            "exercise": {
                "low": "일상 생활에서 걷기나 계단 이용하기와 같은 가벼운 운동부터 시작해보세요.",
                "medium": "현재 운동량을 조금씩 늘려가는 것이 좋겠습니다.",
                "high": "현재 운동 습관을 잘 유지해주세요."
            },
            "smoking": {
                "active": "금연 클리닉이나 전문가의 도움을 받아보는 것은 어떨까요?",
                "trying": "금연 시도는 훌륭합니다. 스트레스 관리와 함께 진행하면 좋습니다."
            },
            "alcohol": {
                "frequent": "주 2회 이하로 음주 횟수를 줄여보는 것이 좋겠습니다.",
                "moderate": "현재 음주량을 유지하면서 음주 전후 수분 섭취를 늘려보세요."
            }
        }
        
        if lifestyle_type in templates:
            # 답변 내용에 따라 적절한 템플릿 선택
            if "exercise" in lifestyle_type:
                if "안함" in answer_text or "없음" in answer_text:
                    return templates["exercise"]["low"]
                elif "가끔" in answer_text or "주 1-2회" in answer_text:
                    return templates["exercise"]["medium"]
                else:
                    return templates["exercise"]["high"]
            
            elif "smoking" in lifestyle_type:
                if "피움" in answer_text or "흡연" in answer_text:
                    return templates["smoking"]["active"]
                else:
                    return templates["smoking"]["trying"]
            
            elif "alcohol" in lifestyle_type:
                if "자주" in answer_text or "매일" in answer_text:
                    return templates["alcohol"]["frequent"]
                else:
                    return templates["alcohol"]["moderate"]
        
        return "생활습관 개선을 위해 전문가와 상담해보시는 것이 좋겠습니다." 