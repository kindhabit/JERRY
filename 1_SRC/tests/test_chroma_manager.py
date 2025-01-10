import pytest
import asyncio
from core.vector_db.vector_store_manager import ChromaManager
from utils.openai_client import OpenAIClient

@pytest.fixture
async def chroma_manager():
    manager = ChromaManager()
    yield manager

@pytest.mark.asyncio
async def test_get_supplement_interaction(chroma_manager):
    # 테스트용 건강 데이터
    health_data = {
        "symptoms": ["fatigue", "joint_pain"],
        "health_metrics": {
            "blood_pressure": {"systolic": 120, "diastolic": 80}
        },
        "lifestyle_factors": {
            "exercise": "moderate",
            "sleep": "good"
        }
    }
    current_supplements = ["vitamin_d", "omega_3"]
    
    result = await chroma_manager.get_supplement_interaction(health_data, current_supplements)
    
    assert isinstance(result, dict)
    assert "status" in result
    assert result["status"] in ["success", "insufficient_data", "low_relevance", "error"]

@pytest.mark.asyncio
async def test_get_health_impacts(chroma_manager):
    # 건강 영향 조회 테스트
    supplement = "vitamin_d"
    health_data = {
        "condition": "osteoporosis",
        "severity": "moderate"
    }
    
    impacts = await chroma_manager.get_health_impacts(supplement, health_data)
    
    assert isinstance(impacts, list)
    if impacts:
        assert all(isinstance(impact, dict) for impact in impacts)
        assert all("supplement" in impact for impact in impacts)
        assert all("health_aspect" in impact for impact in impacts)

@pytest.mark.asyncio
async def test_show_stats(chroma_manager):
    # 통계 조회 테스트
    stats = await chroma_manager.show_stats()
    
    assert isinstance(stats, dict)
    for collection_name, collection_stats in stats.items():
        assert "count" in collection_stats
        assert "metadata_fields" in collection_stats
        assert "last_updated" in collection_stats

@pytest.mark.asyncio
async def test_add_paper_to_collection(chroma_manager):
    # 논문 추가 테스트
    test_paper = {
        "pmid": "test123",
        "title": "Test Paper",
        "abstract": "This is a test abstract",
        "authors": ["Test Author"],
        "publication_date": "2024-01-01",
        "journal": "Test Journal",
        "category": "test_category",
        "weight": 1.0,
        "description": "Test description",
        "processed_text": "Test processed text",
        "llm_analysis": "Test analysis"
    }
    
    success = await chroma_manager._add_paper_to_collection("supplements", test_paper)
    assert isinstance(success, bool)

if __name__ == "__main__":
    asyncio.run(pytest.main([__file__])) 