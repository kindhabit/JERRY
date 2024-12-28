#!/usr/bin/env python3
import os
import logging
import pandas as pd
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config.config_loader import CONFIG, OPENAI_API_KEY

# 로그 설정
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs"))
log_file = os.path.join(log_dir, "medical_terms_init.log")
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(log_file)
console_handler = logging.StreamHandler()
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def initialize_medical_terms_db():
    """
    의학 용어를 위한 ChromaDB 초기화
    기존 건기식 DB와 별도의 컬렉션 사용
    """
    try:
        # OpenAI 임베딩 초기화 (기존 시스템과 동일한 임베딩 사용)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # 의학 용어 전용 ChromaDB 초기화
        db = Chroma(
            persist_directory=CONFIG["chroma"]["medical_terms_dir"],
            collection_name=CONFIG["chroma"]["medical_terms_collection"],
            embedding_function=embeddings
        )
        logger.info("의학 용어 ChromaDB 초기화 완료")
        return db
    except Exception as e:
        logger.error(f"ChromaDB 초기화 실패: {e}")
        raise

def load_initial_medical_terms():
    """
    기본 의학 용어 데이터 로드
    데이터 소스: CSV, API, 또는 수동 입력 데이터
    """
    try:
        # 기본 데이터 파일 경로
        data_path = Path(CONFIG["medical_data"]["initial_data_path"])
        
        if data_path.exists():
            # CSV 파일에서 데이터 로드
            df = pd.read_csv(data_path)
            logger.info(f"기초 의학 용어 {len(df)}개 로드 완료")
            return df.to_dict('records')
        else:
            # 기본 의학 용어 데이터 (CSV 없을 경우)
            return [
                {
                    "korean": "코로나19",
                    "english": "COVID-19",
                    "category": "virus",
                    "references": ["WHO", "CDC"],
                    "verified": True
                },
                # ... 더 많은 기본 데이터
            ]
    except Exception as e:
        logger.error(f"의학 용어 데이터 로드 실패: {e}")
        return []

def store_medical_terms(vectorstore, terms):
    """
    의학 용어를 ChromaDB에 저장
    """
    for term in terms:
        try:
            # 한글 용어 저장
            vectorstore.add_texts(
                texts=[term["korean"]],
                metadatas=[{
                    "korean": term["korean"],
                    "english": term["english"],
                    "category": term["category"],
                    "references": term.get("references", []),
                    "verified": term.get("verified", False),
                    "language": "korean"
                }],
                ids=[f"kr_{term['korean']}"]
            )
            
            # 영문 용어 저장 (역방향 검색 지원)
            vectorstore.add_texts(
                texts=[term["english"]],
                metadatas=[{
                    "korean": term["korean"],
                    "english": term["english"],
                    "category": term["category"],
                    "references": term.get("references", []),
                    "verified": term.get("verified", False),
                    "language": "english"
                }],
                ids=[f"en_{term['english']}"]
            )
            
            logger.info(f"용어 저장 완료: {term['korean']} -> {term['english']}")
        except Exception as e:
            logger.error(f"용어 저장 실패 ({term['korean']}): {e}")

def main():
    """
    의학 용어 데이터베이스 초기화 메인 함수
    """
    try:
        # 1. ChromaDB 초기화
        vectorstore = initialize_medical_terms_db()
        
        # 2. 초기 데이터 로드
        terms = load_initial_medical_terms()
        logger.info(f"총 {len(terms)}개의 의학 용어 로드됨")
        
        # 3. 데이터 저장
        store_medical_terms(vectorstore, terms)
        logger.info("의학 용어 데이터베이스 초기화 완료")
        
    except Exception as e:
        logger.error(f"초기화 과정 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main() 