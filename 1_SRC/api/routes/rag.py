from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models.session import Session, AnalysisResult
from core.analysis.health_analyzer import EnhancedHealthAnalyzer
from core.analysis.question_generator import QuestionGenerator
from core.vector_db.vector_store_manager import ChromaManager
from utils.logger_config import setup_logger

logger = setup_logger('rag_router')
router = APIRouter(prefix="/api/rag", tags=["rag"])

# 의존성 초기화
chroma_manager = ChromaManager()
health_analyzer = EnhancedHealthAnalyzer(chroma_manager)
question_generator = QuestionGenerator(chroma_manager)

# 요청/응답 모델
class CreateSessionRequest(BaseModel):
    health_data: Dict

class Recommendation(BaseModel):
    type: str
    name: str
    confidence: float
    reason: str
    evidence: Optional[List[Dict]] = None

class InteractionWarning(BaseModel):
    source: str  # 예: "medication", "condition"
    target: str  # 예: "supplement_name"
    severity: str  # 예: "high", "medium", "low"
    description: str
    evidence: Optional[List[Dict]] = None

class FollowUpQuestion(BaseModel):
    id: str
    question: str
    context: str
    priority: int
    interaction_check: bool = False

class InitialRecommendations(BaseModel):
    supplements: List[Recommendation]
    lifestyle_changes: List[Dict]
    confidence_scores: Dict[str, float]

class FollowUp(BaseModel):
    questions: List[FollowUpQuestion]
    potential_interactions: List[InteractionWarning]
    required_confirmations: List[str]

class CreateSessionResponse(BaseModel):
    session_id: str
    initial_recommendations: InitialRecommendations
    follow_up: FollowUp

class SessionStatusResponse(BaseModel):
    session_status: str
    current_step: int
    analysis_progress: float

# 세션 저장소 (임시, 실제로는 데이터베이스 사용 필요)
active_sessions: Dict[str, Session] = {}

@router.post("/create-session", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    """새로운 분석 세션을 생성합니다."""
    try:
        # 1. 세션 생성
        session = Session.create_new(request.health_data)
        
        # 2. 초기 건강 데이터 분석 및 1차 추천
        initial_analysis = await health_analyzer.analyze(request.health_data)
        session.analysis_results = initial_analysis
        
        # 3. 첫 질문 세트 생성 (잠재적 간섭 고려)
        initial_questions = await question_generator.generate_questions(session)
        session.current_questions = initial_questions
        
        # 4. 미들웨어로 전달할 통합 응답
        response = CreateSessionResponse(
            session_id=session.id,
            initial_recommendations=InitialRecommendations(
                supplements=initial_analysis.recommendations,
                lifestyle_changes=initial_analysis.lifestyle_suggestions,
                confidence_scores=initial_analysis.confidence_levels
            ),
            follow_up=FollowUp(
                questions=initial_questions,
                potential_interactions=initial_analysis.interaction_warnings,
                required_confirmations=initial_analysis.required_checks
            )
        )
        
        # 5. 세션 저장
        active_sessions[session.id] = session
        
        return response
        
    except Exception as e:
        logger.error(f"세션 생성 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="세션 생성 중 오류가 발생했습니다."
        )

@router.get("/session/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """세션의 현재 상태를 조회합니다."""
    try:
        # 1. 세션 존재 여부 확인
        session = active_sessions.get(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="세션을 찾을 수 없습니다."
            )
        
        # 2. 진행 상태 계산
        total_steps = session.total_expected_steps
        current_step = session.current_step
        progress = (current_step / total_steps) if total_steps > 0 else 0.0
        
        return SessionStatusResponse(
            session_status=session.status,
            current_step=current_step,
            analysis_progress=progress
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"세션 상태 조회 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="세션 상태 조회 중 오류가 발생했습니다."
        ) 