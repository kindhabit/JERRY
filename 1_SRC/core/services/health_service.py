from typing import List, Dict, Set, Optional
from models.supplement import Supplement, HealthEffect, Interaction
from models.health_data import HealthData
from config.config_loader import CONFIG
from core.vector_db.vector_store_manager import ChromaManager
from core.services.rag_service import RAGService
from core.analysis.client_health_analyzer import HealthDataAnalyzer
from utils.logger_config import setup_logger
import logging
import json
from datetime import datetime, date
import time

logger = setup_logger('health_service')

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        elif hasattr(obj, 'model_dump'):  # Pydantic v2
            return obj.model_dump()
        elif hasattr(obj, 'dict'):  # Pydantic v1
            return obj.dict()
        elif hasattr(obj, '__dict__'):  # 일반 객체
            return obj.__dict__
        return str(obj)  # 기타 타입은 문자열로 변환

class HealthService:
    def __init__(self, chroma_manager: ChromaManager):
        self.chroma_manager = chroma_manager
        self.rag_service = RAGService(
            chroma_manager=chroma_manager,
            openai_client=chroma_manager.openai_client
        )
        self.health_analyzer = HealthDataAnalyzer()
        self.json_encoder = DateTimeEncoder()

    def _serialize_json(self, data):
        """JSON 직렬화 헬퍼 메서드"""
        try:
            return json.dumps(data, cls=DateTimeEncoder, ensure_ascii=False)
        except Exception as e:
            logger.error(f"JSON 직렬화 중 오류: {str(e)}")
            # 기본값으로 안전하게 변환 시도
            return str(data)

    async def analyze_health_metrics(self, health_data: HealthData) -> Dict:
        """건강 지표 분석"""
        try:
            start_time = time.time()
            logger.info("건강 지표 분석 시작")
            
            # 1. 건강 데이터 상세 분석
            analysis_result = await self.health_analyzer.analyze_health_data(
                health_data.model_dump()
            )
            logger.info(f"건강 데이터 분석 결과: {self._serialize_json(analysis_result)}")
            
            # 2. 1차 추천 (건강 지표별)
            logger.info("1차 추천 요청 시작")
            primary_recs = await self._get_primary_recommendations(
                analysis_result.get("context", {})
            )
            logger.info(f"1차 추천 결과: {self._serialize_json(primary_recs)}")
            
            # 3. 결과 포맷팅
            result = {
                '분석_요약': '현재 건강 데이터를 기반으로 분석했어요',
                '추천': {
                    '영양제': [rec["name"] for rec in primary_recs],
                    '이유': {rec["name"]: rec["reason"] for rec in primary_recs}
                },
                '추가_질문': await self._generate_custom_question(health_data, primary_recs),
                '분석_시간': f'{time.time() - start_time:.2f}초',
                '분석_일시': datetime.now().isoformat()
            }
            
            logger.info(f"최종 분석 결과: {self._serialize_json(result)}")
            return result
            
        except Exception as e:
            logger.error(f"건강 지표 분석 중 오류: {str(e)}")
            raise

    async def generate_interaction_notice(self, analysis: Dict) -> Dict:
        """상호작용 알림 생성"""
        try:
            if not analysis.get("interactions"):
                return {"notice": None}
                
            return {
                "notice": self._create_interaction_notice(
                    analysis["recommendations"],
                    analysis["interactions"]
                )
            }
        except Exception as e:
            logger.error(f"상호작용 알림 생성 중 오류: {str(e)}")
            raise

    async def detailed_interaction_analysis(
        self,
        health_data: HealthData,
        initial_recommendations: Dict
    ) -> Dict:
        """상세 상호작용 분석"""
        try:
            # 1. 상세 컨텍스트 검색
            context = await self._get_detailed_context(
                health_data,
                initial_recommendations
            )
            
            # 2. RAG 기반 상세 분석
            analysis = await self.rag_service.analyze_with_patterns(
                query=self._create_detailed_query(health_data),
                context=context
            )
            
            return {
                "analysis": analysis,
                "evidence": await self._gather_evidence(analysis)
            }
        except Exception as e:
            logger.error(f"상세 분석 중 오류: {str(e)}")
            raise

    def _create_interaction_notice(
        self,
        recommendations: List[str],
        interactions: List[Dict]
    ) -> str:
        return f"""
            건강 상태 개선을 위해 {', '.join(recommendations)}을(를) 추천드립니다.
            
            다만, {interactions[0]['supplements'][0]}와(과) {interactions[0]['supplements'][1]}의 경우
            {interactions[0]['mechanism']} 때문에 주의가 필요합니다.
            
            이러한 영양제들의 간섭도에 대해 몇 가지 여쭤보고 싶은 점이 있습니다.
            답변해 주시면 최적의 복용 조합과 시간을 찾아드리겠습니다.
        """
    
    async def _get_primary_recommendations(self, health_data: Dict) -> List[Dict]:
        """건강 지표별 1차 추천"""
        try:
            logger.info("1차 추천 시작")
            # 1. 건강 데이터를 문자열로 변환
            health_context = json.dumps(health_data, ensure_ascii=False)
            logger.info(f"건강 데이터 컨텍스트: {health_context}")
            
            # 2. 영양제 검색
            supplements_results = await self.chroma_manager.search_supplements(
                query=f"다음 건강 데이터를 바탕으로 적절한 영양제를 추천해주세요: {health_context}",
                n_results=5
            )
            logger.info(f"검색된 영양제 결과: {supplements_results}")
            
            # 3. GPT를 통한 분석
            analysis_prompt = f"""
            다음 건강 데이터와 검색된 영양제 정보를 바탕으로 추천할 영양제를 분석해주세요.
            각 영양제별로 추천 이유를 자세히 설명해주시되, 실제 수치를 포함해서 설명해주세요.
            
            건강 데이터:
            {health_context}
            
            검색된 영양제 정보:
            {json.dumps(supplements_results, ensure_ascii=False)}
            
            다음 형식으로 응답해주세요:
            [
                {{"name": "영양제_이름", "reason": "추천 이유 (수치 기반으로 설명)"}}
            ]
            
            응답은 한국어로 작성하고, 이유는 최대한 자세하게 설명해주세요.
            """
            
            logger.info("GPT 분석 요청 시작")
            analysis = await self.chroma_manager.openai_client.chat_completion(
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            logger.info(f"GPT 분석 결과: {analysis}")
            
            # 4. 결과 파싱
            try:
                recommendations = json.loads(analysis['content'])
                logger.info(f"1차 추천 결과: {recommendations}")
                return recommendations
            except json.JSONDecodeError as e:
                logger.error(f"1차 추천 결과 파싱 실패: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"1차 추천 중 오류: {str(e)}")
            return []
    
    async def _analyze_interactions(
        self,
        recommendations: List[Dict],
        health_data: Optional[Dict],
        user_profile: Optional[Dict]
    ) -> Dict:
        """간섭도 분석"""
        interactions = {
            "supplement_interactions": [],
            "health_condition_impacts": [],
            "medication_interactions": []
        }
        
        # 1. 영양제 간 상호작용
        for i, supp1 in enumerate(recommendations):
            for supp2 in recommendations[i+1:]:
                # 영양제 정보 구성
                supplements_info = {
                    supp1['name']: supp1.get('related', []),
                    supp2['name']: supp2.get('related', [])
                }
                
                try:
                    # 상호작용 분석 요청
                    interaction = await self.chroma_manager.get_supplement_interaction(
                        health_data=health_data if health_data else {},
                        current_supplements=[supp1['name'], supp2['name']]
                    )
                    
                    if interaction and interaction.get("status") == "success":
                        interactions["supplement_interactions"].append({
                            "supplements": [supp1['name'], supp2['name']],
                            "description": interaction.get("description", "상호작용 정보가 없습니다."),
                            "evidence": interaction.get("evidence", [])
                        })
                except Exception as e:
                    logger.error(f"상호작용 분석 중 오류: {str(e)}")
                    interactions["supplement_interactions"].append({
                        "supplements": [supp1['name'], supp2['name']],
                        "description": f"분석 중 오류 발생: {str(e)}",
                        "evidence": []
                    })
        
        # 2. 건강 상태에 미치는 영향 (health_data가 있는 경우에만)
        if health_data:
            for supp in recommendations:
                try:
                    impacts = await self.chroma_manager.get_health_impacts(
                        supplement=supp['name'],
                        health_data=health_data
                    )
                    if impacts:
                        interactions["health_condition_impacts"].extend(impacts)
                except Exception as e:
                    logger.error(f"건강 영향 분석 중 오류: {str(e)}")

        # 3. 약물 상호작용 (사용자 프로필이 있는 경우)
        if user_profile and 'medications' in user_profile:
            for supp in recommendations:
                for med in user_profile['medications']:
                    try:
                        interaction = await self.chroma_manager.get_medication_interaction(
                            supplement=supp['name'],
                            medication=med
                        )
                        if interaction:
                            interactions["medication_interactions"].append(interaction)
                    except Exception as e:
                        logger.error(f"약물 상호작용 분석 중 오류: {str(e)}")
        
        return interactions

    async def analyze_interactions(self, recommendations: Dict[str, List[str]]) -> Dict:
        """영양제 간 상호작용 분석"""
        try:
            # 입력 형식 변환
            formatted_recommendations = [
                {"name": supp_name, "related": related}
                for supp_name, related in recommendations.items()
            ]
            
            # 상호작용 분석
            analysis_result = await self.chroma_manager.get_supplement_interaction(
                health_data={},  # 현재는 빈 딕셔너리 전달
                current_supplements=[rec["name"] for rec in formatted_recommendations]
            )
            
            # 결과 포맷팅
            return {
                "has_interactions": analysis_result["status"] == "success",
                "interactions": [{
                    "status": analysis_result["status"],
                    "message": analysis_result.get("description", ""),
                    "error_details": analysis_result.get("error") if analysis_result["status"] == "error" else None
                }],
                "questions": [
                    "해당 영양제들을 함께 복용하신 적이 있나요?",
                    "복용 시 불편함을 느끼신 적이 있나요?",
                    "현재 다른 약물을 복용 중이신가요?"
                ],
                "evidence": analysis_result.get("evidence", [])
            }
            
        except Exception as e:
            logger.error(f"영양제 상호작용 분석 중 오류: {str(e)}")
            raise

    def _analyze_health_condition(self, field: str, values: Dict) -> Optional[str]:
        """건강 지표를 분석하여 상태를 판단합니다."""
        try:
            if field == "blood_pressure":
                systolic = values.get("systolic", 0)
                diastolic = values.get("diastolic", 0)
                if systolic >= 140 or diastolic >= 90:
                    return "hypertension"
                elif systolic >= 130 or diastolic >= 85:
                    return "prehypertension"
            
            elif field == "cholesterol":
                total = values.get("total", 0)
                ldl = values.get("ldl", 0)
                hdl = values.get("hdl", 0)
                if total > 240 or ldl > 160 or hdl < 40:
                    return "high_cholesterol"
            
            elif field == "blood_sugar":
                fasting = values.get("fasting", 0)
                post_meal = values.get("post_meal", 0)
                if fasting > 100 or post_meal > 140:
                    return "prediabetes"
            
            elif field == "vitamin_d":
                if values < 20:
                    return "vitamin_d_deficiency"
            
            elif field == "omega_3_index":
                if values < 4:
                    return "omega3_deficiency"
            
            return None
            
        except Exception as e:
            logger.error(f"건강 상태 분석 중 오류: {str(e)}")
            return None 

    async def _generate_custom_question(self, health_data: HealthData, recommendations: List[Dict]) -> str:
        """건강 데이터 기반 맞춤 질문 생성"""
        try:
            # 건강 데이터를 안전하게 직렬화
            health_data_dict = health_data.model_dump() if hasattr(health_data, 'model_dump') else health_data.dict()
            
            prompt = f"""
            다음 건강 데이터와 추천된 영양제를 바탕으로, 가장 중요한 하나의 추가 질문을 생성해주세요:
            
            건강 데이터:
            {json.dumps(health_data_dict, cls=DateTimeEncoder, ensure_ascii=False)}
            
            추천된 영양제:
            {json.dumps(recommendations, cls=DateTimeEncoder, ensure_ascii=False)}
            
            질문은:
            1. 현재 건강 상태에서 가장 주의가 필요한 부분에 대해
            2. 친근한 어조로
            3. '-요'로 끝나도록
            4. 30자 이내로 작성해주세요.
            """
            
            response = await self.chroma_manager.openai_client.chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response['content']
            
        except Exception as e:
            logger.error(f"맞춤 질문 생성 중 오류: {str(e)}")
            return "현재 복용 중인 약물이 있으신가요?"  # 기본 질문 

    async def _get_detailed_context(
        self,
        health_data: HealthData,
        initial_recommendations: Dict
    ) -> Dict:
        """상세 컨텍스트 검색"""
        try:
            # 1. 건강 데이터 분석
            analysis_result = await self.health_analyzer.analyze_health_data(health_data.model_dump())
            
            # 2. 초기 추천사항과 분석 결과 통합
            context = {
                "health_analysis": analysis_result,
                "initial_recommendations": initial_recommendations,
                "current_medications": health_data.medical_history.medications if health_data.medical_history else [],
                "chronic_conditions": health_data.medical_history.chronic_conditions if health_data.medical_history else [],
                "lifestyle_factors": health_data.lifestyle.dict() if health_data.lifestyle else {}
            }
            
            # 3. 관련 건강 지표 매핑
            health_metrics = {}
            if health_data.vital_signs:
                health_metrics.update({
                    "blood_pressure": {
                        "systolic": health_data.vital_signs.blood_pressure_systolic,
                        "diastolic": health_data.vital_signs.blood_pressure_diastolic
                    }
                })
            
            if health_data.blood_test:
                health_metrics.update({
                    "cholesterol": {
                        "total": health_data.blood_test.total_cholesterol,
                        "hdl": health_data.blood_test.hdl_cholesterol,
                        "ldl": health_data.blood_test.ldl_cholesterol,
                        "triglycerides": health_data.blood_test.triglycerides
                    },
                    "liver_function": {
                        "alt": health_data.blood_test.alt,
                        "ast": health_data.blood_test.ast
                    },
                    "kidney_function": {
                        "creatinine": health_data.blood_test.creatinine
                    }
                })
            
            context["health_metrics"] = health_metrics
            
            return context
            
        except Exception as e:
            logger.error(f"상세 컨텍스트 검색 중 오류: {str(e)}")
            raise 