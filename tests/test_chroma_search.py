import pytest
from modules.db.chroma_client import ChromaDBClient

@pytest.mark.asyncio
async def test_similarity_search():
    client = ChromaDBClient()
    
    # 1. 테스트 데이터 추가
    test_texts = [
        "비타민D는 뼈 건강에 중요합니다.",
        "오메가3는 심장 건강에 좋습니다.",
        "마그네슘은 근육 이완에 도움을 줍니다."
    ]
    
    test_metadata = [
        {"type": "vitamin", "category": "bone_health"},
        {"type": "omega", "category": "heart_health"},
        {"type": "mineral", "category": "muscle_health"}
    ]
    
    await client.add_texts(
        texts=test_texts,
        collection_name="supplements",
        metadatas=test_metadata
    )
    
    # 2. 검색 테스트
    results = await client.similarity_search(
        query_text="뼈 건강",
        collection_name="supplements",
        n_results=1
    )
    
    # 3. 결과 검증
    assert len(results["documents"]) == 1
    assert "비타민D" in results["documents"][0]
    assert results["metadatas"][0]["category"] == "bone_health" 