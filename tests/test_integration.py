async def test_full_recommendation_flow():
    """전체 추천 프로세스 테스트"""
    # 테스트 데이터
    health_data = {
        "liver_function": {"AST": 45, "ALT": 50},
        "blood_pressure": {"systolic": 145, "diastolic": 95}
    }
    
    # 1. 초기 분석
    initial_response = await client.post(
        "/api/analyze-health-data",
        json=health_data
    )
    assert initial_response.status_code == 200
    
    # 2. 상호작용 분석
    interaction_response = await client.post(
        "/api/analyze-health-data/interactions",
        json={
            "analysis_id": initial_response.json()["analysis_id"],
            "recommendations": initial_response.json()["field_recommendations"]
        }
    )
    assert interaction_response.status_code == 200
    
    # 3. 최종 추천
    if interaction_response.json()["has_interactions"]:
        final_response = await client.post(
            "/api/analyze-health-data/final",
            json={
                "analysis_id": initial_response.json()["analysis_id"],
                "answers": ["예", "아니���", "가끔"],
                "initial_recommendations": initial_response.json()["field_recommendations"]
            }
        )
        assert final_response.status_code == 200 