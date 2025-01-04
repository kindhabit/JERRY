from abc import ABC, abstractmethod
from typing import List, Dict, Optional, AsyncGenerator
import aiohttp
from datetime import datetime
from config.config_loader import CONFIG
from utils.logger_config import setup_logger
from utils.openai_client import OpenAIClient
import json
import asyncio

logger = setup_logger('data_source')

class DataSource(ABC):
    """데이터 소스 추상 클래스"""
    
    @abstractmethod
    async def search(self, query: str, max_results: int = None) -> List[Dict]:
        """데이터 검색"""
        pass
    
    @abstractmethod
    async def get_details(self, id: str) -> Dict:
        """상세 데이터 조회"""
        pass

class DataSourceManager:
    """데이터 소스 관리자"""
    
    def __init__(self):
        self.sources = {
            "pubmed": PubMedSource()
        }
        self.openai_client = OpenAIClient()
    
    async def collect_data(
        self,
        source: str,
        query: str,
        max_results: Optional[int] = None
    ) -> List[Dict]:
        """데이터 수집"""
        try:
            if source not in self.sources:
                raise ValueError(f"지원하지 않는 데이터 소스: {source}")
            
            data_source = self.sources[source]
            
            # 1. 검색 수행
            logger.info(f"데이터 수집 시작 - 소스: {source}")
            results = await data_source.search(query, max_results)
            
            # 2. 상세 정보 수집
            detailed_results = []
            for result in results:
                details = await data_source.get_details(result["id"])
                detailed_results.append({
                    **details,
                    "metadata": {
                        "source": source,
                        "query": query,
                        "collected_at": datetime.now().isoformat()
                    }
                })
            
            logger.info(f"데이터 수집 완료: {len(detailed_results)}개 항목")
            return detailed_results
            
        except Exception as e:
            logger.error(f"데이터 수집 실패: {str(e)}")
            raise

class PubMedSource:
    """PubMed 데이터 소스"""
    
    def __init__(self):
        """Initialize data source manager"""
        self.supplements = CONFIG.get_supplements()
        self.categories = CONFIG.get_pubmed_categories()
        self.category_weights = CONFIG.get_pubmed_category_weights()
        self.search_strategies = CONFIG.get_pubmed_search_strategies()
        self.settings = CONFIG.get_pubmed_settings()
        self.base_url = self.settings.get('base_url', 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils')
        self.session = None
        self.openai_client = OpenAIClient()
        
    async def _init_session(self):
        """Initialize aiohttp session if not exists"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
            self.session = None
        
    async def search_supplement(self, supplement_name: str) -> AsyncGenerator[Dict, None]:
        """영양제 관련 PubMed 검색 수행"""
        logger.info(f"영양제 검색 시작: {supplement_name}")
        
        # 영어 이름 가져오기
        english_name = self.supplements.get(supplement_name)
        if not english_name:
            logger.error(f"영양제 {supplement_name}의 영어 이름을 찾을 수 없습니다.")
            return
            
        # 각 카테고리별로 검색 수행
        for category_name, category_info in self.categories.items():
            search_term = category_info['search_term']
            weight = category_info['weight']
            description = category_info['description']
            
            # 검색 쿼리 구성
            query = f"{english_name} {search_term} AND 2022:2025[pdat]"
            logger.info(f"검색 쿼리: {query} (카테고리: {category_name} - {description})")
            
            try:
                # PubMed 검색 수행 및 각 논문 처리
                async for paper in self._search_pubmed(query):
                    try:
                        # 기본 정보 추가
                        paper['category'] = category_name
                        paper['weight'] = weight
                        paper['description'] = description
                        
                        # 논문 처리
                        processed_paper = await self._process_single_paper(paper)
                        if processed_paper is None:  # 처리 실패시
                            logger.error(f"LLM 처리 실패로 인한 전체 프로세스 중단 - PMID: {paper.get('pmid')}")
                            return  # 제너레이터 종료
                            
                        yield processed_paper
                        
                    except Exception as e:
                        logger.error(f"논문 처리 중 오류 발생 - PMID: {paper.get('pmid')}: {str(e)}")
                        return  # 오류 발생시 제너레이터 종료
                        
            except Exception as e:
                logger.error(f"PubMed 검색 중 오류 발생: {str(e)}")
                return  # 오류 발생시 제너레이터 종료
                
    async def _search_pubmed(self, query: str) -> AsyncGenerator[Dict, None]:
        """PubMed API를 통한 검색 수행"""
        try:
            await self._init_session()
            
            # 검색 수행
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": "10",
                "retmode": "json"
            }
            
            if self.settings.get("api_key"):
                params["api_key"] = self.settings["api_key"]
            
            # 검색 요청
            async with self.session.get(f"{self.base_url}/esearch.fcgi", params=params) as response:
                if response.status != 200:
                    logger.error(f"PubMed API 오류: {response.status}")
                    return
                    
                search_result = await response.json()
                id_list = search_result.get("esearchresult", {}).get("idlist", [])
                
                if not id_list:
                    return
                    
                # 각 논문 ID에 대해 순차적으로 처리
                for pmid in id_list:
                    try:
                        logger.info(f"논문 처리 시작 - PMID: {pmid}")
                        
                        # 1. 상세 정보 조회
                        details = await self.get_details(pmid)
                        if not details:
                            continue
                            
                        # 2. 상세 정보 반환
                        yield details
                        
                        # 3. 다음 논문 처리 전 잠시 대기
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"논문 상세 정보 조회 실패 (PMID: {pmid}): {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"PubMed API 호출 중 오류 발생: {str(e)}")
            return
            
    async def get_details(self, pmid: str) -> Dict:
        """상세 데이터 조회"""
        try:
            await self._init_session()
            logger.info(f"=== PubMed 상세 정보 요청 - PMID: {pmid} ===")
            
            # 1. esummary로 기본 정보 가져오기
            summary_params = {
                "db": "pubmed",
                "id": pmid,
                "retmode": "json"
            }
            
            if self.settings.get("api_key"):
                summary_params["api_key"] = self.settings["api_key"]
            
            logger.debug(f"Summary API 요청 URL: {self.base_url}/esummary.fcgi")
            logger.debug(f"Summary API 요청 파라미터: {summary_params}")
            
            async with self.session.get(f"{self.base_url}/esummary.fcgi", params=summary_params) as response:
                if response.status != 200:
                    logger.error(f"PubMed Summary API 오류 - PMID: {pmid}")
                    logger.error(f"상태 코드: {response.status}")
                    logger.error(f"응답 내용: {await response.text()}")
                    return None
                    
                summary_result = await response.json()
                logger.debug("Summary API 응답 전문:")
                logger.debug(json.dumps(summary_result, indent=2, ensure_ascii=False))
                
                paper_info = summary_result["result"][pmid]
                
            # 2. efetch로 초록 가져오기
            await asyncio.sleep(1)  # API 제한 회피
            
            fetch_params = {
                "db": "pubmed",
                "id": pmid,
                "retmode": "xml"
            }
            
            if self.settings.get("api_key"):
                fetch_params["api_key"] = self.settings["api_key"]
            
            logger.debug(f"Fetch API 요청 URL: {self.base_url}/efetch.fcgi")
            logger.debug(f"Fetch API 요청 파라미터: {fetch_params}")
            
            async with self.session.get(f"{self.base_url}/efetch.fcgi", params=fetch_params) as response:
                if response.status != 200:
                    logger.error(f"PubMed Fetch API 오류 - PMID: {pmid}")
                    logger.error(f"상태 코드: {response.status}")
                    logger.error(f"응답 내용: {await response.text()}")
                    return None
                    
                xml_content = await response.text()
                logger.debug("Fetch API 응답 전문:")
                logger.debug(xml_content)
                
                # XML에서 초록 추출
                import xml.etree.ElementTree as ET
                root = ET.fromstring(xml_content)
                abstract_element = root.find(".//Abstract")
                abstract = ""
                if abstract_element is not None:
                    for text in abstract_element.findall(".//AbstractText"):
                        if text.text:
                            abstract += text.text + " "
                abstract = abstract.strip()
                
            # 3. 데이터 통합
            paper_data = {
                "pmid": pmid,
                "title": paper_info.get("title", ""),
                "abstract": abstract,
                "authors": paper_info.get("authors", []),
                "publication_date": paper_info.get("pubdate", ""),
                "journal": paper_info.get("source", "")
            }
            
            logger.info("=== PubMed 데이터 검증 ===")
            logger.info(f"제목 길이: {len(paper_data['title'])}")
            logger.info(f"초록 길이: {len(paper_data['abstract'])}")
            logger.info(f"저자 수: {len(paper_data['authors'])}")
            logger.info(f"저널명: {paper_data['journal']}")
            logger.info(f"출판일: {paper_data['publication_date']}")
            
            if not paper_data['abstract']:
                logger.warning(f"초록이 비어있음 - PMID: {pmid}")
                logger.debug("전체 논문 데이터:")
                logger.debug(json.dumps(paper_data, indent=2, ensure_ascii=False))
            
            return paper_data
            
        except Exception as e:
            logger.error(f"=== PubMed API 호출 실패 - PMID: {pmid} ===")
            logger.error(f"에러 타입: {type(e).__name__}")
            logger.error(f"에러 메시지: {str(e)}")
            logger.error("스택 트레이스:", exc_info=True)
            return None
            
    async def _process_single_paper(self, paper: Dict) -> Dict:
        """단일 논문 처리"""
        author_names = []
        
        try:
            pmid = paper.get('pmid')
            if not pmid:
                logger.error("PMID가 없는 논문 - 원본 데이터:")
                logger.error(json.dumps(paper, indent=2, ensure_ascii=False))
                return None
                
            logger.info(f"=== 논문 처리 시작 - PMID: {pmid} ===")
            logger.info("입력 데이터:")
            logger.info(f"제목: {paper.get('title', '')}")
            logger.info(f"저널: {paper.get('journal', '')}")
            
            # 초록 검증
            abstract = paper.get('abstract', '')
            abstract_length = len(abstract)
            logger.info(f"초록 길이: {abstract_length}")
            
            if abstract_length < 100:  # 초록이 너무 짧거나 없는 경우
                logger.error(f"초록이 너무 짧거나 없는 논문 건너뛰기 - PMID: {pmid}")
                logger.error(f"초록 내용: {abstract}")
                return None
                
            logger.info(f"카테고리: {paper.get('category', '')}")
            logger.info(f"가중치: {paper.get('weight', '')}")
            
            # 기본 텍스트 구성
            text = f"Title: {paper.get('title', '')}\n\n"
            text += f"Abstract: {abstract}\n\n"
            
            # authors 처리 - 첫 번째 저자만 사용
            authors = paper.get('authors', [])
            if not authors:
                logger.error(f"저자 정보가 없는 논문 건너뛰기 - PMID: {pmid}")
                logger.error("전체 데이터:")
                logger.error(json.dumps(paper, indent=2, ensure_ascii=False))
                return None
                
            if isinstance(authors, str):
                author_names = [authors]
                logger.info(f"저자(문자열): {authors}")
            elif isinstance(authors, list):
                first_author = authors[0]
                author_names = [first_author.get('name', '') if isinstance(first_author, dict) else str(first_author)]
                if not author_names[0]:
                    logger.error(f"저자 이름이 비어있는 논문 건너뛰기 - PMID: {pmid}")
                    logger.error(f"전체 저자 데이터: {json.dumps(authors, indent=2, ensure_ascii=False)}")
                    return None
                logger.info(f"첫 번째 저자: {author_names[0]}")
                logger.debug(f"전체 저자 목록: {json.dumps(authors, indent=2, ensure_ascii=False)}")
            
            text += f"Author: {author_names[0]}\n"
            text += f"Journal: {paper.get('journal', '')}\n"
            text += f"Publication Date: {paper.get('publication_date', '')}\n"
            
            # LLM 분석 시작
            logger.info(f"=== LLM 분석 시작 - PMID: {pmid} ===")
            logger.info(f"분석할 텍스트 길이: {len(text)} 자")
            logger.debug("분석할 텍스트 내용:")
            logger.debug(text)
            
            analysis_prompt = f"""Please analyze the following medical research paper and provide a structured analysis in JSON format with the following fields:
            1. key_findings: Main discoveries and conclusions
            2. supplement_effects: Relevance to supplement efficacy and mechanisms
            3. safety_considerations: Any safety concerns or side effects
            4. clinical_significance: Importance for clinical practice
            5. authors_formatted: Format the authors list as a comma-separated string
            6. categories_formatted: Format the search categories as a comma-separated string
            
            Paper:
            {text}
            
            Original authors: {authors}
            Original categories: {paper.get('search_categories', [])}
            
            IMPORTANT: Response MUST be in valid JSON format.
            """
            
            logger.debug("LLM 프롬프트:")
            logger.debug(analysis_prompt)
            
            # LLM 분석 수행
            analysis_response = await self.openai_client.analyze_with_context(analysis_prompt)
            
            if not analysis_response:
                logger.error(f"LLM 응답이 비어있음 - PMID: {pmid}")
                logger.error("입력 프롬프트:")
                logger.error(analysis_prompt)
                return None
                
            logger.info(f"LLM 응답 수신 - 길이: {len(analysis_response)} 자")
            logger.debug("원본 LLM 응답:")
            logger.debug(analysis_response)
            
            # JSON 형식 검증
            try:
                # 코드 블록 제거
                clean_response = analysis_response.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:]
                    logger.debug("코드 블록 시작 마커 제거됨")
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]
                    logger.debug("코드 블록 끝 마커 제거됨")
                clean_response = clean_response.strip()
                
                logger.debug("정제된 JSON 문자열:")
                logger.debug(clean_response)
                
                # JSON 파싱
                parsed_json = json.loads(clean_response)
                logger.info(f"JSON 파싱 성공 - 포함된 키: {list(parsed_json.keys())}")
                logger.debug("파싱된 JSON 내용:")
                logger.debug(json.dumps(parsed_json, indent=2, ensure_ascii=False))
                
                # 필수 필드 검증
                required_fields = ['key_findings', 'supplement_effects', 'safety_considerations', 
                                 'clinical_significance', 'authors_formatted', 'categories_formatted']
                missing_fields = [field for field in required_fields if field not in parsed_json]
                if missing_fields:
                    logger.error(f"필수 필드 누락: {missing_fields}")
                    return None
                
                # 정제된 응답 사용
                paper['processed_text'] = text + f"\nAnalysis:\n{clean_response}\n"
                paper['llm_analysis'] = clean_response
                paper['author_names'] = author_names
                
                logger.info(f"=== 논문 처리 완료 - PMID: {pmid} ===")
                return paper
                
            except json.JSONDecodeError as e:
                logger.error(f"=== JSON 파싱 실패 - PMID: {pmid} ===")
                logger.error(f"에러 메시지: {str(e)}")
                logger.error(f"에러 위치: {e.pos}")
                logger.error("LLM 원본 응답:")
                logger.error(analysis_response)
                logger.error("정제된 응답:")
                logger.error(clean_response)
                logger.error("입력 프롬프트:")
                logger.error(analysis_prompt)
                return None
            
        except Exception as e:
            logger.error(f"=== 처리 실패 - PMID: {paper.get('pmid', 'unknown')} ===")
            logger.error(f"에러 타입: {type(e).__name__}")
            logger.error(f"에러 메시지: {str(e)}")
            logger.error("원본 데이터:")
            logger.error(json.dumps(paper, indent=2, ensure_ascii=False))
            logger.error("스택 트레이스:", exc_info=True)
            return None 