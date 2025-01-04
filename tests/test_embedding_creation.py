import pytest
import asyncio
from typing import Dict, List
from modules.db.chroma_manager import EnhancedRAGSystem
from modules.db.chroma_client import ChromaDBClient
from modules.utils.openai_client import OpenAIClient

# 테스트 데이터
sample_data = {
    "supplements": [
        {
            "name": "오메가3",
            "category": "지방산",
            "primary_effects": ["심혈관 건강", "염증 감소"],
            "mechanism": "EPA/DHA를 통한 항염증 작용",
            "cautions": ["혈액 희석제 복용자 주의"],
            "usage_guide": "식사와 함께 1일 1-2회",
            "evidence_level": "HIGH",
            "references": ["PMID:12345", "PMID:67890"],
            "interactions": [
                {
                    "target": "아스피린",
                    "effect": "혈액 희석 효과 증가",
                    "severity": "MODERATE"
                }
            ],
            "health_metrics": {
                "blood_pressure": {
                    "impact": "POSITIVE",
                    "value_range": "수축기 3-4mmHg 감소",
                    "evidence_level": "MODERATE"
                }
            }
        }
    ],
    "complex_cases": {
        "metabolic_syndrome": {
            "conditions": ["고혈압", "당뇨"],
            "medications": ["메트포민", "암로디핀"],
            "considerations": ["약물 상호작용 주의"]
        }
    },
    "lifestyle_factors": {
        "exercise": "주 3회 이상 운동",
        "diet": "저탄고지",
        "sleep": "6-7시간"
    }
}

@pytest.fixture
async def rag_system():
    chroma_client = ChromaDBClient()
    openai_client = OpenAIClient()
    return EnhancedRAGSystem(chroma_client, openai_client)

@pytest.mark.asyncio
async def test_embedding_creation(rag_system):
    """임베딩 생성 테스트"""
    embeddings = await rag_system.create_comprehensive_embeddings(sample_data)
    
    # 기본 검증
    assert len(embeddings) > 0
    assert all(isinstance(emb, dict) for emb in embeddings)
    assert all("text" in emb and "metadata" in emb for emb in embeddings)
    
    # 메타데이터 타입 검증
    metadata_types = set(emb["metadata"]["type"] for emb in embeddings)
    expected_types = {
        "supplement_basic",
        "supplement_interaction",
        "health_metric_effect",
        "complex_case",
        "lifestyle_factor"
    }
    assert metadata_types.intersection(expected_types) == expected_types

@pytest.mark.asyncio
async def test_specific_embedding_content(rag_system):
    """특정 임베딩 내용 테스트"""
    embeddings = await rag_system.create_comprehensive_embeddings(sample_data)
    
    # 오메가3 기본 정보 임베딩 검증
    omega3_basic = next(
        (emb for emb in embeddings 
         if emb["metadata"]["type"] == "supplement_basic" 
         and emb["metadata"]["name"] == "오메가3"),
        None
    )
    assert omega3_basic is not None
    assert "EPA/DHA" in omega3_basic["text"]
    assert "심혈관 건강" in omega3_basic["text"]

@pytest.mark.asyncio
async def test_embedding_storage(rag_system):
    """임베딩 저장 테스트"""
    embeddings = await rag_system.create_comprehensive_embeddings(sample_data)
    
    # ChromaDB 저장 테스트
    success = await rag_system.store_embeddings(embeddings)
    assert success is True
    
    # 저장된 데이터 검증
    stored_data = await rag_system.chroma.get_collection("supplements_basic").get()
    assert len(stored_data) > 0

@pytest.mark.asyncio
async def test_complex_case_embeddings(rag_system):
    """복합 케이스 임베딩 테스트"""
    embeddings = await rag_system.create_comprehensive_embeddings(sample_data)
    
    # 복합 케이스 임베딩 검증
    complex_case = next(
        (emb for emb in embeddings 
         if emb["metadata"]["type"] == "complex_case"),
        None
    )
    assert complex_case is not None
    assert "metabolic_syndrome" in complex_case["text"]
    assert "메트포민" in complex_case["text"]

@pytest.mark.asyncio
async def test_lifestyle_embeddings(rag_system):
    """생활습관 임베딩 테스트"""
    embeddings = await rag_system.create_comprehensive_embeddings(sample_data)
    
    # 생활습관 임베딩 검증
    lifestyle = next(
        (emb for emb in embeddings 
         if emb["metadata"]["type"] == "lifestyle_factor"),
        None
    )
    assert lifestyle is not None
    assert "운동" in lifestyle["text"]
    assert "저탄고지" in lifestyle["text"]

if __name__ == "__main__":
    asyncio.run(pytest.main(["-v", "test_embedding_creation.py"])) 