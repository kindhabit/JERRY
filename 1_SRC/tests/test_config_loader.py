import pytest
from config.config_loader import ConfigLoader
from copy import deepcopy

@pytest.fixture
def config_loader():
    return ConfigLoader()

@pytest.fixture
def test_supplement():
    return {
        "name": "테스트 성분",
        "aliases": ["test supplement", "시험용 성분"],
        "category": "테스트",
        "effects": ["면역력 강화", "피로 회복"],
        "mechanism": "테스트 메커니즘",
        "absorption_path": "장내흡수",
        "absorption_competitors": ["비타민C"],
        "evidence_level": "B",
        "safety_rating": "안전",
        "references": ["테스트 참고문헌 1"]
    }

@pytest.fixture
def test_interactions():
    return {
        "competitors": ["비타민D", "마그네슘"],
        "synergistic": ["아연", "셀레늄"],
        "contraindications": ["고혈압약"],
        "timing": {
            "optimal": "식전",
            "avoid": "취침 전"
        }
    }

def test_update_supplement(config_loader, test_supplement):
    # 초기 상태 저장
    original_supplements = deepcopy(config_loader.get_supplements())
    
    try:
        # 새로운 성분 추가
        config_loader.update_supplement(test_supplement)
        
        # 추가된 성분 확인
        supplements = config_loader.get_supplements()
        added = next((s for s in supplements if s["name"] == test_supplement["name"]), None)
        
        assert added is not None
        assert added["aliases"] == test_supplement["aliases"]
        assert added["effects"] == test_supplement["effects"]
        assert added["evidence_level"] == test_supplement["evidence_level"]
        
        # 기존 성분 업데이트
        updated_supplement = test_supplement.copy()
        updated_supplement["effects"].append("새로운 효과")
        config_loader.update_supplement(updated_supplement)
        
        # 업데이트 확인
        supplements = config_loader.get_supplements()
        updated = next((s for s in supplements if s["name"] == test_supplement["name"]), None)
        
        assert "새로운 효과" in updated["effects"]
        
    finally:
        # 테스트 후 원래 상태로 복구
        config_loader.save_supplements(original_supplements)

def test_update_supplement_interactions(config_loader, test_supplement, test_interactions):
    # 초기 상태 저장
    original_supplements = deepcopy(config_loader.get_supplements())
    
    try:
        # 테스트용 성분 추가
        config_loader.update_supplement(test_supplement)
        
        # 상호작용 정보 업데이트
        config_loader.update_supplement_interactions(test_supplement["name"], test_interactions)
        
        # 업데이트 확인
        supplements = config_loader.get_supplements()
        updated = next((s for s in supplements if s["name"] == test_supplement["name"]), None)
        
        assert updated is not None
        assert "interactions" in updated
        assert updated["interactions"]["synergistic"] == test_interactions["synergistic"]
        assert updated["interactions"]["timing"] == test_interactions["timing"]
        assert all(comp in updated["absorption_competitors"] 
                  for comp in test_interactions["competitors"])
        
    finally:
        # 테스트 후 원래 상태로 복구
        config_loader.save_supplements(original_supplements)

def test_update_supplement_validation(config_loader):
    # 필수 필드가 없는 경우
    invalid_supplement = {
        "name": "잘못된 성분"
        # category와 effects가 없음
    }
    
    with pytest.raises(Exception):
        config_loader.update_supplement(invalid_supplement)
    
    # 잘못된 타입의 필드
    invalid_supplement = {
        "name": "잘못된 성분",
        "category": "테스트",
        "effects": "문자열로 된 효과"  # List[str]이 아님
    }
    
    with pytest.raises(Exception):
        config_loader.update_supplement(invalid_supplement)

def test_update_interactions_validation(config_loader, test_supplement):
    # 존재하지 않는 성분에 대한 상호작용 업데이트
    with pytest.raises(Exception):
        config_loader.update_supplement_interactions(
            "존재하지 않는 성분",
            {"competitors": ["테스트"]}
        )
    
    # 잘못된 형식의 상호작용 정보
    invalid_interactions = {
        "competitors": "문자열로 된 경쟁성분"  # List[str]이 아님
    }
    
    with pytest.raises(Exception):
        config_loader.update_supplement_interactions(
            test_supplement["name"],
            invalid_interactions
        ) 