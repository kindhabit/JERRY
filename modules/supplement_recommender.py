import os
import json
import logging
from typing import Dict, List
from modules.db.chroma_client import ChromaDBClient
from langchain_openai import ChatOpenAI
from modules.data_parser import HealthData

logger = logging.getLogger(__name__)

TRANSLATION_CACHE_FILE = os.path.join(os.path.dirname(__file__), "../data/translation_cache.json")

class SupplementRecommender:
    def __init__(self):
        self.chroma_client = ChromaDBClient()
        self.llm = ChatOpenAI(model="gpt-4")
        self.translation_cache = self._load_translation_cache()

    def _load_translation_cache(self):
        """번역 캐시를 로드합니다."""
        if os.path.exists(TRANSLATION_CACHE_FILE):
            with open(TRANSLATION_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_translation_cache(self):
        """번역 캐시를 저장합니다."""
        with open(TRANSLATION_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.translation_cache, f, ensure_ascii=False, indent=2)
        logger.info("번역 캐시 저장 완료")

    async def _translate_term(self, term: str) -> str:
        """GPT를 사용하여 용어를 번역"""
        if term in self.translation_cache:
            return self.translation_cache[term]

        prompt = f"Translate this Korean medical/supplement term to English: {term}"
        response = await self.llm.agenerate([prompt])
        translation = response.generations[0].text.strip()
        
        self.translation_cache[term] = translation
        self._save_translation_cache()
        
        return translation

    async def add_supplement_data(self, supplement_name: str, pubmed_client) -> None:
        """새로운 영양제 데이터를 ChromaDB에 추가"""
        try:
            # 영양제 이름 번역
            english_name = await self._translate_term(supplement_name)
            
            # PubMed 검색 쿼리 생성
            query = f"{english_name} supplement effects OR benefits OR safety"
            logger.info(f"PubMed 검색: {query}")

            # PubMed 검색 및 데이터 저장
            pmids = pubmed_client.search_pmids(query, retmax=10)
            abstracts = pubmed_client.fetch_abstracts(pmids)

            for pmid, abstract in abstracts:
                try:
                    self.chroma_client.collections["base"].add_texts(
                        [abstract],
                        metadatas=[{
                            "supplement": supplement_name,
                            "supplement_en": english_name,
                            "pmid": pmid,
                            "query": query
                        }],
                        ids=[f"{supplement_name}_{pmid}"]
                    )
                    logger.info(f"{supplement_name} 데이터 저장 완료: PMID {pmid}")
                except Exception as e:
                    logger.error(f"ChromaDB 저장 실패: {e}")

        except Exception as e:
            logger.error(f"영양제 데이터 추가 실패: {e}")

    async def _create_analysis_prompt(self, data: Dict) -> str:
        """분석 프롬프트 생성"""
        return f"""
        분석할 건강 지표:
        - 신체 정보
          * 키: {data.get('height')}cm
          * 체중: {data.get('weight')}kg
          * 허리둘레: {data.get('waist_circumference')}cm
        - BMI: {data.get('bmi')}
        - 혈압: {data.get('systolic_bp')}/{data.get('diastolic_bp')}
        - 콜레스테롤
          * 총: {data.get('total_cholesterol')}
          * HDL: {data.get('hdl_cholesterol')}
          * LDL: {data.get('ldl_cholesterol')}
          * 중성지방: {data.get('triglyceride')}
        - 공복혈당: {data.get('fasting_blood_sugar')}
        - 간 기능
          * SGOT/AST: {data.get('sgotast')}
          * SGPT/ALT: {data.get('sgptalt')}
          * γ-GTP: {data.get('gammagtp')}
        - 신장 기능
          * 크레아티닌: {data.get('creatinine')}
          * GFR: {data.get('gfr')}
        - 추가 추천
          * 위내시경: {data.get('cancerdata', {}).get('recommendations', '정보 없음')}
        
        위 건강 지표를 분석하여 한글과 영문으로 응답:
        1. 현재 건강 상태
        2. 위험 요인
        3. 권장할 수 있는 영양제와 그 근거
        4. 주의사항
        
        JSON 형식으로 응답:
        {{
            "health_status": {{
                "ko": "현재 건강 상태 설명",
                "en": "Current health status description"
            }},
            "risk_factors": [
                {{
                    "ko": "위험요인1",
                    "en": "Risk factor 1"
                }}
            ],
            "supplement_recommendations": [
                {{
                    "supplement": {{
                        "ko": "영양제명",
                        "en": "Supplement name"
                    }},
                    "reason": {{
                        "ko": "추천 이유",
                        "en": "Recommendation reason"
                    }},
                    "caution": {{
                        "ko": "주의사항",
                        "en": "Precautions"
                    }}
                }}
            ]
        }}
        """

    async def process_health_data(self, data: Dict) -> Dict:
        """건강 데이터 기반 필드별 초기 추천"""
        try:
            # 입력 데이터 로깅
            logger.debug(f"Processing health data: {json.dumps(data, ensure_ascii=False, indent=2)}")
            
            # 필수 필드 체크 및 매핑
            field_mapping = {
                "bmi": "bmi",
                "systolic_bp": "systolic_bp",
                "diastolic_bp": "diastolic_bp",
                "total_cholesterol": "total_cholesterol",
                "hdl_cholesterol": "hdl_cholesterol", 
                "ldl_cholesterol": "ldl_cholesterol",
                "fasting_blood_sugar": "fasting_blood_sugar",
                "sgotast": "sgotast",  # 간 기능
                "sgptalt": "sgptalt",  # 간 기능
                "gammagtp": "gammagtp",  # 간 기능
                "triglyceride": "triglyceride",  # 중성지방
                "creatinine": "creatinine",  # 신장 기능
                "gfr": "gfr"  # 신장 기능
            }
            
            # 데이터 정제
            cleaned_data = {}
            for target_field, source_field in field_mapping.items():
                value = data.get(source_field)
                if value is not None:
                    try:
                        cleaned_data[target_field] = float(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid value for {source_field}: {value}")
                        cleaned_data[target_field] = None
                else:
                    cleaned_data[target_field] = None
            
            # 추가 정보 로깅
            if cancer_data := data.get("cancerdata"):
                logger.info(f"Cancer data found: {cancer_data}")
            
            # 건강 지표 분석
            health_metrics = {
                "bmi": cleaned_data["bmi"],
                "blood_pressure": {
                    "systolic": cleaned_data["systolic_bp"],
                    "diastolic": cleaned_data["diastolic_bp"]
                },
                "cholesterol": {
                    "total": cleaned_data["total_cholesterol"],
                    "hdl": cleaned_data["hdl_cholesterol"],
                    "ldl": cleaned_data["ldl_cholesterol"],
                    "triglyceride": cleaned_data["triglyceride"]
                },
                "blood_sugar": cleaned_data["fasting_blood_sugar"],
                "liver_function": {
                    "sgot": cleaned_data["sgotast"],
                    "sgpt": cleaned_data["sgptalt"],
                    "ggt": cleaned_data["gammagtp"]
                },
                "kidney_function": {
                    "creatinine": cleaned_data["creatinine"],
                    "gfr": cleaned_data["gfr"]
                }
            }
            
            logger.debug(f"Processed health metrics: {json.dumps(health_metrics, ensure_ascii=False, indent=2)}")

            # 2. 프롬프트 생성 및 GPT 분석
            prompt = await self._create_analysis_prompt(data)
            gpt_response = await self.llm.agenerate([prompt])
            gpt_analysis = gpt_response.generations[0][0]
            
            # GPT 응답 전처리 및 로깅
            logger.debug(f"GPT Raw Response: {gpt_analysis.text}")
            
            # 응답 텍스트에서 JSON 부분만 추출
            cleaned_response = gpt_analysis.text.strip()
            json_start = cleaned_response.find('{')
            json_end = cleaned_response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                cleaned_response = cleaned_response[json_start:json_end]
            else:
                logger.error("JSON 형식을 찾을 수 없습니다")
                logger.error(f"전체 응답: {cleaned_response}")
                raise ValueError("Invalid response format: No JSON found")

            logger.debug(f"Cleaned Response: {cleaned_response}")
            
            # JSON 파싱
            try:
                gpt_recommendations = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패: {str(e)}")
                logger.error(f"파싱 시도한 텍스트: {cleaned_response}")
                raise
            
            # 3. ChromaDB 검색 실행
            chroma_results = await self.chroma_client.similarity_search(
                query_text=str(data),
                n_results=5
            )

            # 결과 처리
            research_results = chroma_results

            # 5. GPT 분석 결과 파싱
            try:
                gpt_recommendations = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패: {str(e)}")
                logger.error(f"파싱 시도한 텍스트: {cleaned_response}")
                raise
            
            # 6. 분석 결과 조합
            analysis = {
                "metrics": health_metrics,
                "recommendations": [
                    {
                        "category": rec["supplement"]["ko"],
                        "category_en": rec["supplement"]["en"],
                        "supplements": [rec["supplement"]["ko"]],
                        "supplements_en": [rec["supplement"]["en"]],
                        "reason": rec["reason"]["ko"],
                        "reason_en": rec["reason"]["en"],
                        "caution": rec["caution"]["ko"],
                        "caution_en": rec["caution"]["en"]
                    }
                    for rec in gpt_recommendations["supplement_recommendations"]
                ],
                "evidence": {
                    "research": research_results,
                    "gpt_analysis": gpt_recommendations,
                    "risk_factors": [
                        {"ko": risk["ko"], "en": risk["en"]} 
                        for risk in gpt_recommendations["risk_factors"]
                    ]
                }
            }
            
            # 7. 기존 ��이스 추천도 추가
            if health_metrics["cholesterol"]["total"] > 200:
                analysis["recommendations"].append({
                    "category": "콜레스테롤 관리",
                    "category_en": "Cholesterol Management",
                    "supplements": ["오메가3", "홍국"],
                    "supplements_en": ["Omega-3", "Red Yeast Rice"],
                    "reason": "높은 콜레스테롤 수치 관리",
                    "reason_en": "Management of high cholesterol levels",
                    "caution": "간 기능 모니턴 필요",
                    "caution_en": "Liver function monitoring required"
                })
            
            if health_metrics["blood_pressure"]["systolic"] > 130:
                analysis["recommendations"].append({
                    "category": "혈압 관리",
                    "category_en": "Blood Pressure Management",
                    "supplements": ["마그네슘", "코엔자임Q10"],
                    "supplements_en": ["Magnesium", "Coenzyme Q10"],
                    "reason": "혈압 관리",
                    "reason_en": "Blood pressure management",
                    "caution": "복용량 조절 필요",
                    "caution_en": "Dosage adjustment needed"
                })
            
            # 8. 상호작용 분석 추가
            if len(analysis["recommendations"]) > 1:
                all_supplements = []
                for rec in analysis["recommendations"]:
                    all_supplements.extend(rec["supplements"])
                
                interactions = self.chroma_client.search_interactions(all_supplements)
                if interactions:
                    analysis["interaction_warning"] = {
                        "ko": f"다음 영양제들 간의 상호작용 확인 필요: {', '.join(all_supplements)}",
                        "en": f"Interaction check needed for: {', '.join(all_supplements)}",
                        "reason": interactions
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"건강 데이터 분석 실패: {str(e)}")
            raise

    async def analyze_interactions(self, recommendations: List[Dict]) -> Dict:
        """상호작용 분석 및 질문 생성"""
        try:
            supplements = []
            for rec in recommendations:
                supplements.extend(rec.get("supplements", []))

            if len(supplements) > 1:
                interactions = await self.chroma_client.search_interactions(supplements)
                return {
                    "has_interactions": True,
                    "interactions": interactions,
                    "message": "잠재적 상호작용이 발견되어 추가 확인이 필요합니다."
                }

            return {"has_interactions": False}
            
        except Exception as e:
            logger.error(f"상호작용 분석 실패: {str(e)}")
            raise

    async def process_user_answers(self, answers: List[str], initial_recommendations: Dict) -> Dict:
        """답변 분석 및 최종 추천"""
        try:
            final_analysis = {
                "recommendations": initial_recommendations.copy(),
                "evidence": {},
                "safety_notes": [],
                "safety_notes_en": []
            }
            
            # 사용자 답변에 따른 조정
            for answer in answers:
                if "알레르기" in answer:
                    final_analysis["safety_notes"].append(
                        "알레르기 반응 주의 필요"
                    )
                    final_analysis["safety_notes_en"].append(
                        "Caution required for allergic reactions"
                    )
                if "복용중 약���" in answer:
                    final_analysis["safety_notes"].append(
                        "사와 상담 후 복용 권장"
                    )
                    final_analysis["safety_notes_en"].append(
                        "Consultation with doctor recommended before use"
                    )
                    
            # GPT를 사용한 답변 분석
            analysis_prompt = f"""
            사용자 답변을 분석하여 추가 주의사항 생성:
            답변: {answers}
            기존 추천: {initial_recommendations}
            
            JSON 형식으로 응답:
            {{
                "additional_notes": [
                    {{
                        "ko": "추가 주의사항1",
                        "en": "Additional note 1"
                    }}
                ],
                "adjustments": [
                    {{
                        "supplement": {{
                            "ko": "영양제명",
                            "en": "Supplement name"
                        }},
                        "adjustment": {{
                            "ko": "조정사항",
                            "en": "Adjustment details"
                        }}
                    }}
                ]
            }}
            """
            
            gpt_analysis = await self.llm.agenerate([analysis_prompt])
            analysis_result = json.loads(gpt_analysis.generations[0].text)
            
            # GPT 분석 결과 추가
            for note in analysis_result["additional_notes"]:
                final_analysis["safety_notes"].append(note["ko"])
                final_analysis["safety_notes_en"].append(note["en"])
                
            # 추천 조정사항 반영
            for adj in analysis_result["adjustments"]:
                final_analysis["adjustments"] = {
                    "ko": adj["adjustment"]["ko"],
                    "en": adj["adjustment"]["en"]
                }
                
            return final_analysis
            
        except Exception as e:
            logger.error(f"사용자 답변 처리 실패: {str(e)}")
            raise 

    async def generate_recommendations(
        self,
        health_data: HealthData,
        research_data: Dict
    ) -> Dict:
        """건강 데이터와 연구 결과를 기반으로 추천 생성"""
        try:
            # 1. 건강 지표 기반 1차 추천
            recommendations = await self._get_base_recommendations(health_data)
            
            # 2. 상호작용 검토 필요 여부 확인
            # 모든 추천된 영양제 목록 생성
            all_supplements = []
            for rec in recommendations:
                all_supplements.extend(rec["supplements"])
            
            # 상호작용 검사
            interaction_check = None
            if len(all_supplements) > 1:
                interactions = self.chroma_client.search_interactions(all_supplements)
                if interactions["has_interactions"]:
                    interaction_check = interactions
            
            return {
                "status": "success",
                "recommendations": recommendations,
                "interaction_check": interaction_check
            }
            
        except Exception as e:
            logger.error(f"추천 생성 실패: {str(e)}")
            raise

    async def _analyze_with_gpt(self, health_data: HealthData) -> Dict:
        """GPT를 사용한 건강 데이터 분석"""
        try:
            # 프롬프트 생성
            prompt = await self._create_analysis_prompt(health_data.dict())
            
            # GPT 분석 수행
            gpt_response = await self.llm.agenerate([prompt])
            gpt_analysis = gpt_response.generations[0][0]
            
            # 응답 텍스트에서 JSON 부분만 추출
            cleaned_response = gpt_analysis.text.strip()
            json_start = cleaned_response.find('{')
            json_end = cleaned_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                cleaned_response = cleaned_response[json_start:json_end]
            else:
                logger.error("JSON 형식을 찾을 수 없습니다")
                logger.error(f"전체 응답: {cleaned_response}")
                raise ValueError("Invalid response format: No JSON found")
                
            # JSON 파싱
            try:
                analysis_result = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패: {str(e)}")
                logger.error(f"파싱 시도한 텍스트: {cleaned_response}")
                raise
                
            return analysis_result
            
        except Exception as e:
            logger.error(f"GPT 분석 실패: {str(e)}")
            raise

    def _is_relevant_to_health_data(self, effect: Dict, health_data: HealthData) -> bool:
        """건강 데이터와 연구 결과의 관련성 확인"""
        try:
            content = effect.get("content", "").lower()
            
            # 콜레스테롤 관련
            if health_data.total_cholesterol and health_data.total_cholesterol > 200:
                if "cholesterol" in content:
                    return True
                    
            # 혈압 관련
            if health_data.systolic_bp and health_data.systolic_bp > 130:
                if "blood pressure" in content:
                    return True
                    
            # 혈당 관련
            if health_data.fasting_blood_sugar and health_data.fasting_blood_sugar > 100:
                if "blood sugar" in content:
                    return True
                    
            # 간 기능 관련
            if health_data.sgotast and health_data.sgotast > 40:
                if "liver" in content:
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"관련성 확인 실패: {str(e)}")
            return False

    def _format_recommendation(self, effect: Dict) -> Dict:
        """연구 결과를 추천 형식으로 변환"""
        try:
            return {
                "supplement": {
                    "ko": effect.metadata.get("supplement", ""),
                    "en": effect.metadata.get("supplement_en", "")
                },
                "reason": {
                    "ko": effect.page_content,
                    "en": effect.metadata.get("description_en", "")
                },
                "evidence": {
                    "pmid": effect.metadata.get("pmid", ""),
                    "evidence_level": effect.metadata.get("evidence_level", "")
                }
            }
        except Exception as e:
            logger.error(f"추천 형식 변환 실패: {str(e)}")
            raise 

    def _build_recommendation_prompt(
        self,
        health_data: HealthData,
        base_info: List[Dict],
        interactions: List[Dict],
        health_effects: List[Dict],
        contraindications: List[Dict]
    ) -> str:
        """추천 생성을 위한 프롬프트 구성"""
        return f"""
        건강 데이터와 연구 결과를 기반으로 영양제 추천을 생성해주세요.
        
        건강 데이터:
        {json.dumps(health_data.model_dump(), ensure_ascii=False, indent=2)}
        
        연구 결과:
        1. 기본 정보:
        {json.dumps(base_info, ensure_ascii=False, indent=2)}
        
        2. 상호작용:
        {json.dumps(interactions, ensure_ascii=False, indent=2)}
        
        3. 건강 효과:
        {json.dumps(health_effects, ensure_ascii=False, indent=2)}
        
        4. 금기사항:
        {json.dumps(contraindications, ensure_ascii=False, indent=2)}
        
        다음 형식으로 응답해주세요:
        {
            "recommendations": [
                {
                    "supplement": "영양제명",
                    "reason": "추천 이유",
                    "dosage": "권장 복용량",
                    "caution": "주의사항",
                    "evidence": ["연구 근거1", "연구 근거2"]
                }
            ]
        }
        """

    def _parse_gpt_response(self, response: str) -> Dict:
        """GPT 응답 파싱"""
        try:
            # JSON 부분 추출
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            raise ValueError("JSON not found in response")
        except Exception as e:
            logger.error(f"GPT 응답 파싱 실패: {str(e)}")
            raise 

    async def _get_base_recommendations(self, health_data: HealthData) -> List[Dict]:
        """건강 지표 기반 기본 추천 생성"""
        recommendations = []
        
        # analysisData에서 건강 지표 추출
        if health_data.analysisData:
            metrics = health_data.analysisData.get("healthMetrics", {})
            
            # 건강 지표별 RAG 검색 쿼리 생성
            search_queries = []
            
            # BMI 관련
            if bmi := metrics.get("bodyMetrics", {}).get("BMI"):
                if float(bmi) > 25:
                    search_queries.append({
                        "query": "supplements for weight management and metabolism",
                        "category": "체중 관리",
                        "condition": f"BMI {bmi}"
                    })
            
            # 콜레스테롤 관련
            if chol := metrics.get("lipidProfile", {}).get("totalCholesterol"):
                if int(chol) > 200:
                    search_queries.append({
                        "query": "supplements for cholesterol management",
                        "category": "콜레스테롤 관리",
                        "condition": f"총 콜레스테롤 {chol}"
                    })
            
            # 혈압 관련
            if bp := metrics.get("bloodPressure", {}).get("systolic"):
                if int(bp) > 130:
                    search_queries.append({
                        "query": "supplements for blood pressure management",
                        "category": "혈압 관리",
                        "condition": f"수축기 혈압 {bp}"
                    })
            
            # RAG로 각 쿼리에 대한 추천 검색
            for search_info in search_queries:
                search_result = self.chroma_client.similarity_search(
                    query_text=search_info["query"],
                    n_results=3
                )
                
                if search_result["documents"]:  # 검색 결과가 있는 경우
                    for i, doc in enumerate(search_result["documents"]):
                        recommendations.append({
                            "category": search_info["category"],
                            "supplements": [search_result["metadatas"][i].get("supplement", "")],
                            "reason": f"{search_info['condition']}로 인한 {doc}",
                            "evidence": {
                                "pmid": search_result["metadatas"][i].get("pmid", ""),
                                "evidence_level": search_result["metadatas"][i].get("evidence_level", "")
                            }
                        })
        
        return recommendations 