import re
from typing import Dict, List, Optional
from dataclasses import asdict
from modules.supplement_types import StudyMetrics, SupplementEffect, EvidenceLevel
from config.config_loader import CONFIG
import logging
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class SupplementAnalysis(BaseModel):
    supplements_mentioned: dict = Field(..., description="Mentioned supplements")
    health_effects: dict = Field(..., description="Health effects")
    interactions: dict = Field(..., description="Interaction details")
    safety_profile: dict = Field(..., description="Safety information")

class TextAnalyzer:
    def __init__(self):
        # 기본 분석용 빠른 GPT-4 모델
        self.fast_llm = ChatOpenAI(
            model="gpt-4-1106-preview",
            temperature=0.1,
            max_tokens=1000
        )
        
        # 중요 호작용 분석용 정확한 모델
        self.accurate_llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.2,
            max_tokens=1200
        )
        
        # CONFIG에서 설정 가져오기
        self.supplements = CONFIG["pubmed"]["supplements"]
        self.health_keywords = CONFIG["pubmed"]["health_keywords"]
        
        self.parser = JsonOutputParser(pydantic_object=SupplementAnalysis)
        
    async def analyze_text(self, text: str, pmid: str = None) -> Dict:
        """텍스트 종합 분석"""
        try:
            # 기본 정보 추출
            data = {
                "pmid": pmid,
                "text": text[:2000],  # 텍스트 길이 제한
                "supplements": [],
                "health_effects": {},
                "interactions": {},
                "contraindications": {}
            }
            
            # 건강 키워드 관련 여부 확인
            health_related = any(keyword in text.lower() for keyword in self.health_keywords)
            
            if not health_related:
                return data
                
            # 모든 서플리먼트 관련 정보 추출 (known + discovered)
            supplements_info = await self._extract_all_supplements(text)
            data["supplements"] = supplements_info
            
            # 건강 효과 분석
            data["health_effects"] = await self._analyze_health_effects(text)
            
            # 상호작용 분석
            data["interactions"] = await self._analyze_interactions(text)
            
            # 금기사항 분석  
            data["contraindications"] = await self._analyze_contraindications(text)
            
            return data
            
        except Exception as e:
            logger.error(f"텍스트 분석 실패: {str(e)}")
            return None 

    async def extract_interactions(self, text: str) -> Dict:
        """초록에서 상호작용 정보 추출"""
        try:
            logger.info(f"상호작용 분석 시작 - 텍스트 길이: {len(text)}")
            logger.debug(f"분석할 텍스트 (처음 200자): {text[:200]}...")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a scientific analyzer specialized in supplement interactions."),
                ("user", """
                    Analyze the following abstract and provide ONLY a JSON response about interactions and effects.
                    
                    Abstract:
                    {text}
                    
                    Focus on these specific aspects:
                    1. Supplement-Supplement interactions
                    2. Supplement-Medication interactions
                    3. Supplement-Condition interactions
                    4. Timing and dosage interactions
                    
                    IMPORTANT: Return a JSON with this structure:
                    {{
                        "supplements_mentioned": {{
                            "primary": "main supplement being studied",
                            "others": ["other supplement1", "other supplement2"],
                            "combinations": ["combination1", "combination2"]
                        }},
                        "interactions": {{
                            "supplement_supplement": [],
                            "supplement_medication": [],
                            "supplement_condition": [],
                            "timing_dosage": []
                        }}
                    }}
                    """)
            ])
            
            chain = prompt | self.accurate_llm
            logger.debug("LLM 체인 실행 시작")
            response = await chain.ainvoke({"text": text})
            response_text = response.content
            logger.debug(f"LLM 응답 전문:\n{response_text}")
            
            try:
                result = json.loads(response_text)
                logger.info(f"파싱된 결과: {json.dumps(result, indent=2, ensure_ascii=False)}")
                if not result.get("interactions"):
                    logger.debug("No interactions found in the text")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패:")
                logger.error(f"응답 전문:\n{response_text}")
                logger.error(f"에러 위치: {str(e)}")
                logger.error(f"문제의 라인: {response_text.splitlines()[e.lineno-1]}")
                return {
                    "interactions": None,
                    "health_effects": None
                }
            
        except Exception as e:
            logger.error(f"상호작용 추출 중 예외 발생:", exc_info=True)
            logger.error(f"처리 중이던 텍스트: {text[:500]}...")
            return {
                "has_interactions": False,
                "interactions": [],
                "severity": "unknown",
                "error": str(e)
            } 

    async def _extract_all_supplements(self, text: str) -> Dict:
        """모든 서플리먼트 정보 추출 (알려진 것 + 새로 발견된 것)"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a supplement analysis expert."),
                ("user", """
                    Analyze this text and identify all supplements:
                    {text}
                    
                    Known supplements: {supplements}
                """)
            ])
            
            chain = prompt | self.fast_llm | self.parser
            response = await chain.ainvoke({
                "text": text,
                "supplements": self.supplements
            })
            return response

        except Exception as e:
            logger.error(f"서플리먼트 추출 실패: {str(e)}")
            return {
                "known_supplements": [],
                "discovered_supplements": [],
                "combinations": [],
                "context": {}
            } 