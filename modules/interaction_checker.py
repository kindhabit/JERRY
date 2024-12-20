from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import logging
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# Logging 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ChromaDB 설정
CHROMA_COLLECTION_NAME = "supplement_interactions"
CHROMA_DIR = "./chroma_db"
QUESTION_LIMIT = 3  # 사용자당 최대 질문 제한
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# FastAPI 애플리케이션
app = FastAPI()


# Pydantic 데이터 모델
class UserHealthData(BaseModel):
    health_conditions: list[str]


class UserResponse(BaseModel):
    conflicting_supplement: str
    action: str  # "대체 성분 찾기" 또는 "기존 성분 유지"


# ChromaDB 초기화
def initialize_chroma_db():
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        db = Chroma(persist_directory=CHROMA_DIR, collection_name=CHROMA_COLLECTION_NAME, embedding_function=embeddings)
        logger.info("ChromaDB initialized successfully.")
        return db
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {e}")
        raise


# 간섭 탐지 함수
def detect_conflicts(user_health_data, db):
    try:
        results = db.similarity_search(user_health_data)
        if not results:
            logger.info("No conflicts detected.")
            return []

        conflicts = []
        for result in results:
            conflicts.append({
                "supplement": result.metadata.get("supplement"),
                "indicator": result.metadata.get("indicator"),
                "effect": result.metadata.get("effect"),
                "severity": result.metadata.get("conflict_severity", "unknown"),
                "details": result.metadata.get("details", "No additional details available.")
            })

        logger.info(f"Conflicts detected: {conflicts}")
        return conflicts
    except Exception as e:
        logger.error(f"Error detecting conflicts: {e}")
        return []


# 대체 성분 추천 함수
def suggest_alternatives(conflicts, db):
    try:
        suggestions = []
        for conflict in conflicts:
            query = f"Find alternatives for {conflict['supplement']} with minimal conflict."
            results = db.similarity_search(query)

            alternatives = [
                result.metadata.get("supplement") for result in results[:3]
                if result.metadata.get("supplement")
            ]
            if alternatives:
                suggestions.append({
                    "original": conflict["supplement"],
                    "alternatives": alternatives,
                    "reason": f"Conflict detected for {conflict['supplement']}. Alternatives: {', '.join(alternatives)}."
                })
            else:
                suggestions.append({
                    "original": conflict["supplement"],
                    "alternatives": ["No suitable alternative found"],
                    "reason": f"No alternatives found for {conflict['supplement']} in the database."
                })

        logger.info(f"Suggestions: {suggestions}")
        return suggestions
    except Exception as e:
        logger.error(f"Error suggesting alternatives: {e}")
        return []


# FastAPI 엔드포인트: 간섭 탐지 및 대안 제안
@app.post("/detect_conflicts/")
async def detect_and_suggest(user_health_data: UserHealthData):
    """
    사용자 건강 데이터를 기반으로 간섭 탐지 및 대안 제안
    """
    try:
        db = initialize_chroma_db()

        # 1. 간섭 탐지
        conflicts = detect_conflicts(user_health_data.health_conditions, db)
        if not conflicts:
            return {"status": "success", "message": "No conflicts detected. All supplements are safe to recommend."}

        # 2. 대체 성분 추천
        suggestions = suggest_alternatives(conflicts, db)

        return {
            "status": "success",
            "conflicts": conflicts,
            "suggestions": suggestions
        }
    except Exception as e:
        logger.error(f"Error in detect_and_suggest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# FastAPI 엔드포인트: 사용자 응답 처리
@app.post("/user_response/")
async def handle_user_response(user_response: UserResponse):
    """
    사용자 응답 처리 및 대체 성분 추천 업데이트
    """
    try:
        db = initialize_chroma_db()
        if user_response.action == "대체 성분 찾기":
            # 대체 성분 추천
            alternatives = suggest_alternatives([{"supplement": user_response.conflicting_supplement}], db)
            return {"status": "success", "alternatives": alternatives}
        else:
            return {"status": "success", "message": "Original supplement retained."}
    except Exception as e:
        logger.error(f"Error in handle_user_response: {e}")
        raise HTTPException(status_code=500, detail=str(e))
