from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

class Question(BaseModel):
    id: str
    text: str
    context: str
    priority: int = 1
    evidence: Optional[List[Dict]] = None
    interaction_check: bool = False

class Answer(BaseModel):
    question_id: str
    answer_text: str
    timestamp: datetime = Field(default_factory=datetime.now)

class Recommendation(BaseModel):
    type: str
    name: str
    confidence: float
    reason: str
    evidence: Optional[List[Dict]] = None

class InteractionWarning(BaseModel):
    source: str
    target: str
    severity: str
    description: str
    evidence: Optional[List[Dict]] = None

class AnalysisResult(BaseModel):
    primary_concerns: List[Dict] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    lifestyle_suggestions: List[Dict] = Field(default_factory=list)
    interaction_warnings: List[InteractionWarning] = Field(default_factory=list)
    required_checks: List[str] = Field(default_factory=list)
    evidence: List[Dict] = Field(default_factory=list)
    confidence_levels: Dict[str, float] = Field(default_factory=dict)
    follow_up_questions: List[Question] = Field(default_factory=list)

class Session(BaseModel):
    id: str
    status: str = "created"
    current_step: int = 0
    total_expected_steps: int = 5  # 기본값 설정
    
    health_data: Dict
    current_questions: List[Question] = Field(default_factory=list)
    answers: List[Answer] = Field(default_factory=list)
    analysis_results: Optional[AnalysisResult] = None
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @classmethod
    def create_new(cls, health_data: Dict) -> "Session":
        """새로운 세션을 생성합니다."""
        return cls(
            id=str(uuid.uuid4()),
            health_data=health_data
        )

    def update_status(self, new_status: str) -> None:
        """세션 상태를 업데이트합니다."""
        self.status = new_status
        self.updated_at = datetime.now()

    def add_answer(self, answer: Answer) -> None:
        """답변을 추가합니다."""
        self.answers.append(answer)
        self.current_step += 1
        self.updated_at = datetime.now()

    def update_analysis(self, analysis: AnalysisResult) -> None:
        """분석 결과를 업데이트합니다."""
        self.analysis_results = analysis
        self.updated_at = datetime.now()

    def update_questions(self, questions: List[Question]) -> None:
        """현재 질문 목록을 업데이트합니다."""
        self.current_questions = questions
        self.updated_at = datetime.now() 