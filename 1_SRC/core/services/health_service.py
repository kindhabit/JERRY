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
            logger.info(f"건강 데이터 분석 결과 - def analyze_health_metrics: {self._serialize_json(analysis_result)}")
            
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
            # 입력 데이터 검증
            if not health_data:
                raise ValueError("health_data가 제공되지 않았습니다")
            if not initial_recommendations:
                raise ValueError("initial_recommendations가 제공되지 않았습니다")
            
            # 1. 상세 컨텍스트 검색
            context = await self._get_detailed_context(
                health_data,
                initial_recommendations
            )
            
            # 컨텍스트 검증
            if not context:
                logger.warning("컨텍스트가 비어있습니다")
                context = {
                    "health_analysis": {},
                    "initial_recommendations": initial_recommendations,
                    "warning": "상세 건강 데이터를 찾을 수 없습니다"
                }
            
            # 2. RAG 기반 상세 분석
            analysis = await self.rag_service.analyze_with_patterns(
                query=self._create_detailed_query(health_data),
                context=context
            )
            
            # 분석 결과 검증
            if not isinstance(analysis, dict):
                logger.error(f"예상치 못한 분석 결과 형식: {type(analysis)}")
                raise ValueError("분석 결과가 올바른 형식이 아닙니다")
            
            required_fields = {"status", "description"}
            if not all(field in analysis for field in required_fields):
                logger.error(f"분석 결과에 필수 필드가 누락됨: {analysis}")
                raise ValueError("분석 결과에 필수 필드가 누락되었습니다")
            
            return analysis
            
        except ValueError as e:
            logger.error(f"상세 분석 중 유효성 검증 오류: {str(e)}")
            return {
                "status": "error",
                "description": str(e),
                "error_type": "validation_error"
            }
        except Exception as e:
            logger.error(f"상세 분석 중 오류: {str(e)}")
            return {
                "status": "error",
                "description": "상세 분석 중 오류가 발생했습니다",
                "error_type": "internal_error",
                "error_details": str(e)
            }

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
            logger.info(f"검색된 영양제 결과: {str(supplements_results)[:50]}...")
            
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
            logger.error("-" * 50)  # 구분선 추가
            logger.error(f"1차 추천 생성 중 오류 발생: {str(e)}", exc_info=True)
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
            # 입력 검증
            if not recommendations:
                raise ValueError("영양제 추천 목록이 비어있습니다")
            
            # 입력 형식 변환
            try:
                formatted_recommendations = [
                    {"name": supp_name, "related": related}
                    for supp_name, related in recommendations.items()
                ]
            except Exception as e:
                logger.error(f"추천 데이터 형식 변환 실패: {str(e)}")
                raise ValueError("올바르지 않은 추천 데이터 형식입니다")
            
            # 상호작용 분석
            analysis_result = await self.chroma_manager.get_supplement_interaction(
                health_data={},  # 현재는 빈 딕셔너리 전달
                current_supplements=[rec["name"] for rec in formatted_recommendations]
            )
            
            # 분석 결과 검증
            if not analysis_result:
                logger.warning("상호작용 분석 결과가 비어있습니다")
                return {
                    "has_interactions": False,
                    "interactions": [],
                    "questions": self._get_default_questions(),
                    "evidence": []
                }
            
            # status 필드 안전하게 접근
            status = analysis_result.get("status")
            if not status:
                logger.error("분석 결과에 status 필드가 없습니다")
                raise ValueError("분석 결과 형식이 올바르지 않습니다")
            
            # 결과 포맷팅
            return {
                "has_interactions": status == "success",
                "interactions": [{
                    "status": status,
                    "message": analysis_result.get("description", "상호작용 정보가 없습니다"),
                    "error_details": analysis_result.get("error") if status == "error" else None,
                    "severity": analysis_result.get("severity", "unknown")
                }],
                "questions": self._get_interaction_questions(analysis_result),
                "evidence": analysis_result.get("evidence", [])
            }
            
        except ValueError as e:
            logger.error(f"영양제 상호작용 분석 중 유효성 검증 오류: {str(e)}")
            return {
                "has_interactions": False,
                "interactions": [{
                    "status": "error",
                    "message": str(e),
                    "error_details": "validation_error"
                }],
                "questions": self._get_default_questions(),
                "evidence": []
            }
        except Exception as e:
            logger.error(f"영양제 상호작용 분석 중 오류: {str(e)}")
            return {
                "has_interactions": False,
                "interactions": [{
                    "status": "error",
                    "message": "상호작용 분석 중 오류가 발생했습니다",
                    "error_details": str(e)
                }],
                "questions": self._get_default_questions(),
                "evidence": []
            }

    def _get_default_questions(self) -> List[str]:
        """기본 질문 목록 반환"""
        return [
            "해당 영양제들을 함께 복용하신 적이 있나요?",
            "복용 시 불편함을 느끼신 적이 있나요?",
            "현재 다른 약물을 복용 중이신가요?"
        ]

    def _get_interaction_questions(self, analysis_result: Dict) -> List[str]:
        """분석 결과에 따른 맞춤 질문 생성"""
        try:
            severity = analysis_result.get("severity", "unknown")
            base_questions = self._get_default_questions()
            
            if severity == "high":
                base_questions.append("이전에 비슷한 영양제 조합으로 부작용을 경험하신 적이 있나요?")
            elif severity == "medium":
                base_questions.append("복용 시간을 분리하여 섭취하시는 것이 좋을 것 같은데, 가능하신가요?")
            
            return base_questions
            
        except Exception as e:
            logger.error(f"맞춤 질문 생성 중 오류: {str(e)}")
            return self._get_default_questions()

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
            logger.info("상세 컨텍스트 검색 시작")
            logger.info(f"입력 health_data 타입: {type(health_data)}")
            logger.info(f"입력 health_data 내용: {health_data.model_dump()}")
            logger.info(f"입력 initial_recommendations: {initial_recommendations}")
            
            # 1. 건강 데이터 분석
            analysis_result = await self.health_analyzer.analyze_health_data(health_data.model_dump())
            logger.info(f"건강 데이터 분석 결과 -def _get_detailed_context : {analysis_result}")
            
            # 2. 초기 추천사항과 분석 결과 통합
            try:
                medical_history = health_data.medical_history
                logger.info(f"medical_history 데이터: {medical_history}")
                
                lifestyle = health_data.lifestyle
                logger.info(f"lifestyle 데이터: {lifestyle}")
                
                context = {
                    "health_analysis": analysis_result,
                    "initial_recommendations": initial_recommendations,
                    "current_medications": medical_history.medications if medical_history else [],
                    "chronic_conditions": medical_history.chronic_conditions if medical_history else [],
                    "lifestyle_factors": lifestyle.dict() if lifestyle else {}
                }
                logger.info(f"생성된 context: {context}")
            except Exception as e:
                logger.error(f"컨텍스트 생성 중 에러 - 타입: {type(e).__name__}")
                logger.error(f"에러 메시지: {str(e)}")
                logger.error("스택 트레이스:", exc_info=True)
                raise
            
            # 3. 관련 건강 지표 매핑
            try:
                health_metrics = {}
                if health_data.vital_signs:
                    logger.info(f"vital_signs 데이터: {health_data.vital_signs}")
                    health_metrics.update({
                        "blood_pressure": {
                            "systolic": health_data.vital_signs.blood_pressure_systolic,
                            "diastolic": health_data.vital_signs.blood_pressure_diastolic
                        }
                    })
                
                if health_data.blood_test:
                    logger.info(f"blood_test 데이터: {health_data.blood_test}")
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
                logger.info(f"최종 health_metrics: {health_metrics}")
            except Exception as e:
                logger.error(f"건강 지표 매핑 중 에러 - 타입: {type(e).__name__}")
                logger.error(f"에러 메시지: {str(e)}")
                logger.error("스택 트레이스:", exc_info=True)
                raise
            
            return context
            
        except Exception as e:
            logger.error(f"상세 컨텍스트 검색 중 오류 - 타입: {type(e).__name__}")
            logger.error(f"에러 메시지: {str(e)}")
            logger.error("스택 트레이스:", exc_info=True)
            logger.error(f"전체 health_data: {health_data.model_dump() if health_data else None}")
            raise 

    def _create_detailed_query(self, health_data: HealthData) -> str:
        """건강 데이터를 기반으로 상세 분석을 위한 쿼리를 생성합니다."""
        try:
            query_parts = []
            
            # 1. 기본 건강 정보
            if health_data.vital_signs:
                vital_signs = health_data.vital_signs
                if vital_signs.blood_pressure_systolic and vital_signs.blood_pressure_diastolic:
                    query_parts.append(f"혈압: {vital_signs.blood_pressure_systolic}/{vital_signs.blood_pressure_diastolic}")
            
            # 2. 혈액 검사 결과
            if health_data.blood_test:
                blood_test = health_data.blood_test
                if blood_test.total_cholesterol:
                    query_parts.append(f"총 콜레스테롤: {blood_test.total_cholesterol}")
                if blood_test.hdl_cholesterol:
                    query_parts.append(f"HDL: {blood_test.hdl_cholesterol}")
                if blood_test.ldl_cholesterol:
                    query_parts.append(f"LDL: {blood_test.ldl_cholesterol}")
            
            # 3. 의료 이력
            if health_data.medical_history:
                medical_history = health_data.medical_history
                if medical_history.chronic_conditions:
                    query_parts.append(f"만성질환: {', '.join(medical_history.chronic_conditions)}")
                if medical_history.medications:
                    query_parts.append(f"복용 중인 약물: {', '.join(medical_history.medications)}")
            
            # 4. 생활습관
            if health_data.lifestyle:
                lifestyle = health_data.lifestyle
                if lifestyle.exercise_frequency is not None:
                    query_parts.append(f"운동 빈도: 주 {lifestyle.exercise_frequency}회")
                if lifestyle.smoking:
                    query_parts.append("흡연자")
                if lifestyle.alcohol_consumption:
                    query_parts.append(f"음주: {lifestyle.alcohol_consumption}")
            
            # 쿼리 조합
            return " AND ".join(query_parts) if query_parts else "기본 건강 분석"
            
        except Exception as e:
            logger.error(f"상세 쿼리 생성 중 오류: {str(e)}")
            return "기본 건강 분석"

    async def analyze_supplements(self, health_data: Dict) -> Dict:
        """건강 지표 분석"""
        try:
            logger.info("건강 지표 분석 시작")
            
            # 건강 데이터 분석
            analysis_result = await self._analyze_health_data(health_data)
            logger.info(f"건강 데이터 분석 결과 - analyze_supplements: {analysis_result}")
            
            # 1차 추천
            logger.info("1차 추천 요청 시작")
            recommendations = await self._get_primary_recommendations(analysis_result)
            logger.info(f"1차 추천 결과 타입: {type(recommendations)}")
            logger.info(f"1차 추천 결과 내용: {recommendations}")
            
            try:
                # 최종 분석 결과 생성
                영양제_목록 = [rec["name"] for rec in recommendations]
                logger.info(f"영양제 목록 생성: {영양제_목록}")
                
                이유_매핑 = {rec["name"]: rec["reason"] for rec in recommendations}
                logger.info(f"이유 매핑 생성: {이유_매핑}")
                
                final_result = {
                    "분석_요약": "현재 건강 데이터를 기반으로 분석했어요",
                    "추천": {
                        "영양제": 영양제_목록,
                        "이유": 이유_매핑
                    },
                    "추가_질문": self._get_follow_up_question(analysis_result),
                    "분석_시간": f"{time.time() - self.start_time:.2f}초",
                    "분석_일시": datetime.now().isoformat()
                }
                
                logger.info(f"최종 분석 결과: {final_result}")
                return final_result
                
            except Exception as e:
                logger.error(f"최종 결과 생성 중 에러 발생 - 타입: {type(e).__name__}")
                logger.error(f"에러 메시지: {str(e)}")
                logger.error("스택 트레이스:", exc_info=True)
                logger.error(f"recommendations 데이터: {recommendations}")
                raise
                
        except Exception as e:
            logger.error(f"분석 중 에러 발생 - 타입: {type(e).__name__}")
            logger.error(f"에러 메시지: {str(e)}")
            logger.error("스택 트레이스:", exc_info=True)
            logger.error(f"입력 데이터: {health_data}")
            raise 