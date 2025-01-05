from typing import List, Dict, Optional
import uuid
from models.session import Question, Session, AnalysisResult
from core.vector_db.vector_store_manager import ChromaManager
from utils.logger_config import setup_logger

logger = setup_logger('question_generator')

class QuestionGenerator:
    def __init__(self, chroma_manager: ChromaManager):
        self.chroma = chroma_manager
        self.logger = logger

    async def generate_questions(self, session: Session) -> List[Question]:
        """세션 상태에 기반하여 질문을 생성합니다."""
        try:
            questions = []
            
            # 1. 상호작용 관련 질문
            if session.analysis_results and session.analysis_results.interaction_warnings:
                questions.extend(
                    await self._generate_interaction_questions(
                        session.analysis_results.interaction_warnings
                    )
                )
            
            # 2. 건강 상태 관련 질문
            if session.health_data.get("medical_history", {}).get("chronic_conditions"):
                questions.extend(
                    await self._generate_condition_questions(
                        session.health_data["medical_history"]["chronic_conditions"]
                    )
                )
            
            # 3. 생활습관 관련 질문
            if session.health_data.get("lifestyle"):
                questions.extend(
                    await self._generate_lifestyle_questions(
                        session.health_data["lifestyle"]
                    )
                )
            
            # 우선순위 기반 정렬
            return sorted(questions, key=lambda q: q.priority, reverse=True)
            
        except Exception as e:
            self.logger.error(f"질문 생성 중 오류: {str(e)}")
            raise

    async def _generate_interaction_questions(
        self,
        warnings: List[Dict]
    ) -> List[Question]:
        """상호작용 관련 질문을 생성합니다."""
        questions = []
        
        for warning in warnings:
            if warning["source"].startswith("medication_"):
                med_name = warning["source"].replace("medication_", "")
                questions.append(Question(
                    id=str(uuid.uuid4()),
                    text=f"{med_name}을(를) 복용하신지 얼마나 되셨나요? 현재 복용량은 어느 정도인가요?",
                    context=f"medication_interaction_{med_name}",
                    priority=2 if warning["severity"] == "high" else 1,
                    interaction_check=True
                ))
            
            elif warning["source"].startswith("condition_"):
                condition = warning["source"].replace("condition_", "")
                questions.append(Question(
                    id=str(uuid.uuid4()),
                    text=f"{condition} 관련하여 현재 어떤 치료를 받고 계신가요?",
                    context=f"condition_interaction_{condition}",
                    priority=2 if warning["severity"] == "high" else 1,
                    interaction_check=True
                ))
        
        return questions

    async def _generate_condition_questions(
        self,
        conditions: List[str]
    ) -> List[Question]:
        """건강 상태 관련 질문을 생성합니다."""
        questions = []
        
        for condition in conditions:
            # 관련 질문 검색
            results = await self.chroma.similarity_search(
                query=f"{condition} patient assessment questions",
                collection_name="health_data",
                n_results=2
            )
            
            if results["documents"]:
                questions.append(Question(
                    id=str(uuid.uuid4()),
                    text=f"{condition} 진단을 받으신지 얼마나 되셨나요?",
                    context=f"condition_history_{condition}",
                    priority=1,
                    evidence=results["metadatas"]
                ))
        
        return questions

    async def _generate_lifestyle_questions(
        self,
        lifestyle: Dict
    ) -> List[Question]:
        """생활습관 관련 질문을 생성합니다."""
        questions = []
        
        # 운동 관련 질문
        if lifestyle.get("exercise_frequency", 0) < 3:
            questions.append(Question(
                id=str(uuid.uuid4()),
                text="운동하기 어려운 특별한 이유가 있으신가요?",
                context="lifestyle_exercise_barrier",
                priority=1
            ))
        
        # 수면 관련 질문
        if lifestyle.get("sleep_hours", 0) < 7:
            questions.append(Question(
                id=str(uuid.uuid4()),
                text="수면의 질은 어떠신가요? 자주 깨시거나 잠들기 어려운가요?",
                context="lifestyle_sleep_quality",
                priority=1
            ))
        
        # 스트레스 관련 질문
        if lifestyle.get("stress_level", 0) > 3:
            questions.append(Question(
                id=str(uuid.uuid4()),
                text="스트레스 해소를 위해 현재 하고 계신 활동이 있으신가요?",
                context="lifestyle_stress_management",
                priority=1
            ))
        
        return questions 