#!/usr/bin/env python3
# scripts/add_pubmed_to_chroma.py
import os
import logging
from datetime import datetime
from modules.pubMed import PubMedClient
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from nlp_tools import extract_interaction_data, filter_by_date
from config.config_loader import CONFIG, OPENAI_API_KEY  # 공통 설정 모듈에서 가져옴
from dataclasses import dataclass
from typing import List, Dict, Optional
import re
from enum import Enum
from modules.supplement_types import *
from modules.text_analysis import StudyAnalyzer
from modules.pubmed import PubMedClient
from modules.chroma_db import ChromaDBClient

# 로그 설정
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs"))
log_file = os.path.join(log_dir, "app.log")

# 로그 디렉토리 생성
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class EvidenceLevel(Enum):
    HIGH = "A"    # RCT, 메타분석
    MEDIUM = "B"  # 관찰연구, 코호트
    LOW = "C"     # 사례연구, 전문가 의견
    
@dataclass
class StudyMetrics:
    sample_size: int
    duration_weeks: Optional[int]
    p_value: Optional[float]
    confidence_interval: Optional[str]

@dataclass
class SupplementEffect:
    effect_type: str
    description: str
    mechanism: Optional[str]
    dosage: Optional[str]
    confidence: float

class SupplementAnalyzer:
    def __init__(self):
        self.study_patterns = {
            "RCT": r"random(ized|ised)\s+control(led)?\s+trial",
            "meta_analysis": r"meta[\-\s]analysis",
            "cohort": r"cohort\s+study",
            "case_control": r"case[\-\s]control",
        }
        
    def extract_study_details(self, abstract: str) -> Dict:
        """논문에서 연구 세부사항 추출"""
        # ... (이전 코드와 동일)
        
    def analyze_supplement_effects(self, abstract: str, supplement: str) -> List[SupplementEffect]:
        """보충제 효과 분석"""
        # ... (이전 코드와 동일)

def initialize_chroma_db():
    """ChromaDB 클라이언트를 초기화합니다."""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        db = Chroma(
            persist_directory=CONFIG["chroma"]["persist_directory"],
            collection_name=CONFIG["chroma"]["collection_name"],
            embedding_function=embeddings
        )
        logger.info("ChromaDB initialized successfully.")
        return db
    except Exception as e:
        logger.error(f"Could not connect to Chroma server: {e}")
        raise RuntimeError("Failed to initialize ChromaDB. Please ensure the Chroma server is running.")

import json

def sanitize_metadata(metadata):
    """
    ChromaDB 저장을 위한 메타데이터 정제
    """
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, dict):
            sanitized[key] = json.dumps(value)  # dict를 JSON 문자열로 변환
        else:
            sanitized[key] = str(value)  # 기타 타입을 문자열로 변환
    return sanitized



def disable_existing_record(pmid, vectorstore, new_metadata):
    """
    기존 데이터를 Disable 처리하고 Remark로 기록합니다.
    """
    try:
        existing = vectorstore.similarity_search(pmid, k=1)
        if not existing:
            return False

        # 기존 데이터 비활성화
        old_metadata = existing[0].metadata
        old_metadata["status"] = "disabled"
        old_metadata["remark"] = f"Disabled on {datetime.now().isoformat()}. Updated with new data."

        vectorstore.update_texts(
            ids=[pmid],
            metadatas=[old_metadata]
        )
        logger.info(f"PMID {pmid} 기록이 비활성화 처리되었습니다.")
        return True
    except Exception as e:
        logger.error(f"Error disabling existing record in ChromaDB: {e}")
        return False


def generate_vector_from_field_data(field_data):
    """
    주어진 필드 데이터를 기반으로 벡터 데이터를 생성합니다.
    """
    vector_data = {}

    # 필드별 위험 수준 계산 및 추천 성분 추가 (근거 포함)
    vector_data["blood_pressure_risk"] = (
        "high" if field_data.get("systolic_bp", 0) >= 130 or field_data.get("diastolic_bp", 0) >= 80 else "normal"
    )
    vector_data["blood_pressure_supplements"] = {
        "Omega-3": "Helps reduce systolic and diastolic blood pressure.",
        "Magnesium": "Relaxes blood vessels and improves blood flow."
    }

    vector_data["blood_sugar_risk"] = (
        "high" if field_data.get("fasting_blood_sugar", 0) >= 100 else "normal"
    )
    vector_data["blood_sugar_supplements"] = {
        "Chromium": "Improves insulin sensitivity and glucose metabolism.",
        "Alpha Lipoic Acid": "Reduces oxidative stress and improves blood sugar control."
    }

    vector_data["liver_function_risk"] = (
        "high" if field_data.get("sgotast", 0) > 40
        or field_data.get("sgptalt", 0) > 40
        or field_data.get("gammagtp", 0) > 60 else "normal"
    )
    vector_data["liver_function_supplements"] = {
        "Milk Thistle": "Supports liver detoxification and reduces inflammation.",
        "Vitamin E": "Protects liver cells from oxidative stress."
    }

    vector_data["cholesterol_risk"] = (
        "high" if field_data.get("total_cholesterol", 0) > 200 else "normal"
    )
    vector_data["cholesterol_supplements"] = {
        "Red Yeast Rice": "Lowers LDL cholesterol by inhibiting cholesterol synthesis.",
        "Fish Oil": "Increases HDL cholesterol and reduces triglycerides."
    }

    vector_data["kidney_function_risk"] = (
        "high" if field_data.get("gfr", 0) < 60 else "normal"
    )
    vector_data["kidney_function_supplements"] = {
        "Astragalus": "Improves kidney function and reduces proteinuria.",
        "CoQ10": "Supports cellular energy production and kidney health."
    }

    vector_data["bmi_risk"] = (
        "high" if field_data.get("bmi", 0) >= 25 else "normal"
    )
    vector_data["bmi_supplements"] = {
        "Green Tea Extract": "Enhances metabolism and supports weight loss.",
        "CLA": "Helps reduce body fat while preserving muscle mass."
    }

    return vector_data


def fetch_and_store_data(vectorstore, supplements, health_keywords, pubmed_client):
    """
    1. PubMed에서 논문 검색
    2. 초록 데이터 추출
    3. NLP로 상호작용 데이터 분석
    4. ChromaDB에 저장
    """
    for supplement in supplements:
        for keyword in health_keywords:
            query = f"{supplement} {keyword}"
            logger.info(f"PubMed 검색 시작: {query}")

            # PubMed 데이터 검색
            pmids = pubmed_client.search_pmids(query, retmax=10)
            if not pmids:
                logger.info(f"{query}에 대한 PMID를 찾을 수 없습니다.")
                continue

            abstracts = pubmed_client.fetch_abstracts(pmids)
            if not abstracts:
                logger.info(f"{query}에 대한 초록을 가져올 수 없습니다.")
                continue

            # 초록 데이터 처리
            for pmid, abstract, title in abstracts:
                # NLP로 간섭 데이터 및 부작용 추출
                interaction, side_effects = extract_interaction_data(abstract)

                # ChromaDB 업로드
                try:
                    vectorstore.add_texts(
                        [abstract],
                        metadatas=[{
                            "pmid": pmid,
                            "query": query,
                            "title": title,
                            "supplement": supplement,
                            "health_keyword": keyword,
                            "interaction": interaction,
                            "side_effects": side_effects,
                            "details": "Positive or negative effect on health metrics.",
                            "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid != "N/A" else None,
                            "status": "active"
                        }],
                        ids=[pmid]
                    )
                    logger.info(f"PMID: {pmid} 논문 데이터 저장 완료.")
                except Exception as e:
                    logger.error(f"Chroma 업로드 중 오류 발생: {e}")


async def main():
    pubmed_client = PubMedClient()
    chroma_client = ChromaDBClient()
    
    supplements = ["Milk Thistle", "Probiotics", "Omega-3"]
    health_keywords = ["liver function", "blood pressure", "cholesterol"]
    
    for supplement in supplements:
        for keyword in health_keywords:
            query = pubmed_client.generate_structured_query(supplement, keyword)
            pmids = await pubmed_client.search_pmids(query)
            
            for pmid in pmids:
                abstract_data = await pubmed_client.fetch_structured_abstract(pmid)
                if abstract_data:
                    await chroma_client.add_supplement_data(abstract_data)


if __name__ == "__main__":
    main()