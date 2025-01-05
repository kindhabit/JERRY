import uuid
from datetime import datetime
from typing import Dict, Optional, List
from models.session import Session, SessionStatus, Question, Answer, AnalysisResult
from utils.logger_config import setup_logger

logger = setup_logger('session_manager')

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        
    async def create_session(self, health_data: Dict) -> Session:
        """새로운 분석 세션을 생성합니다."""
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            health_data=health_data
        )
        self._sessions[session_id] = session
        logger.info(f"새로운 세션 생성: {session_id}")
        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """세션 ID로 세션을 조회합니다."""
        session = self._sessions.get(session_id)
        if not session:
            logger.warning(f"세션을 찾을 수 없음: {session_id}")
            return None
        return session

    async def update_session_status(
        self,
        session_id: str,
        status: SessionStatus,
        step: Optional[str] = None
    ) -> Optional[Session]:
        """세션 상태를 업데이트합니다."""
        session = await self.get_session(session_id)
        if not session:
            return None
            
        session.status = status
        session.updated_at = datetime.now()
        if step:
            session.current_step = step
            
        logger.info(f"세션 상태 업데이트: {session_id} -> {status}")
        return session

    async def add_analysis_result(
        self,
        session_id: str,
        result: AnalysisResult
    ) -> Optional[Session]:
        """분석 결과를 세션에 추가합니다."""
        session = await self.get_session(session_id)
        if not session:
            return None
            
        session.analysis_results = result
        session.updated_at = datetime.now()
        logger.info(f"분석 결과 추가: {session_id}")
        return session

    async def add_questions(
        self,
        session_id: str,
        questions: List[Question]
    ) -> Optional[Session]:
        """질문 목록을 세션에 추가합니다."""
        session = await self.get_session(session_id)
        if not session:
            return None
            
        session.current_questions.extend(questions)
        session.updated_at = datetime.now()
        session.status = SessionStatus.WAITING_ANSWER
        logger.info(f"질문 추가: {session_id}, {len(questions)}개")
        return session

    async def add_answer(
        self,
        session_id: str,
        answer: Answer
    ) -> Optional[Session]:
        """사용자 답변을 세션에 추가합니다."""
        session = await self.get_session(session_id)
        if not session:
            return None
            
        session.answers.append(answer)
        session.updated_at = datetime.now()
        logger.info(f"답변 추가: {session_id}, question_id: {answer.question_id}")
        return session

    async def get_session_state(self, session_id: str) -> Dict:
        """세션의 현재 상태 정보를 반환합니다."""
        session = await self.get_session(session_id)
        if not session:
            return {"error": "세션을 찾을 수 없습니다."}
            
        return {
            "status": session.status,
            "current_step": session.current_step,
            "has_analysis": bool(session.analysis_results),
            "questions_count": len(session.current_questions),
            "answers_count": len(session.answers),
            "last_updated": session.updated_at.isoformat()
        }

    async def cleanup_session(self, session_id: str) -> bool:
        """세션을 정리합니다."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"세션 정리 완료: {session_id}")
            return True
        return False 