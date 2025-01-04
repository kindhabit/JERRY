import pytest
from modules.db.pattern_learner import DynamicPatternLearner
from modules.db.chroma_manager import ChromaManager
from modules.db.enhanced_rag_system import EnhancedRAGSystem

@pytest.mark.asyncio
async def test_pattern_learning_integration():
    """패턴 학습 통합 테스트"""
    # 1. 시스템 초기화
    chroma_manager = ChromaManager()
    await chroma_manager.reinitialize_database(force=True)
    
    # 2. 테스트 데이터
    test_interaction = {
        "interaction_type": "supplement_interaction",
        "entities": ["비타민D", "칼슘"],
        "effect": "흡수율 증가",
        "confidence": 0.85
    }
    
    # 3. 패턴 학습
    await chroma_manager.pattern_learner.learn_from_interaction(test_interaction)
    
    # 4. 학습된 패턴 검증
    patterns = chroma_manager.pattern_learner.patterns
    assert "supplement_interactions" in patterns
    assert len(patterns["supplement_interactions"]) > 0 