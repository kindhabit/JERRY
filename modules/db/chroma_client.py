import chromadb
import logging
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from chromadb.config import Settings
from config.config_loader import CONFIG
import gc
import openai
import json
import os
import traceback

logger = logging.getLogger(__name__)

class ChromaDBClient:
    def __init__(self):
        try:
            # 세션 ID 생성 및 로깅
            self.session_id = os.getpid()
            logger.info(f"새로운 ChromaDB 세션 시작 - PID: {self.session_id}")
            
            # 상대 경로를 절대 경로로 변환
            persist_dir = os.path.abspath(CONFIG['chroma']['persist_directory'])
            logger.debug(f"ChromaDB 저장 경로 (변환 전): {CONFIG['chroma']['persist_directory']}")
            logger.debug(f"ChromaDB 저장 경로 (변환 후): {persist_dir}")
            logger.info(f"저장 경로: {persist_dir}")
            
            logger.debug(f"전체 설정: {json.dumps(CONFIG, indent=2)}")
            # ChromaDB 클라이언트 초기화
            self.client = chromadb.PersistentClient(
                path=persist_dir
            )
            logger.debug(f"ChromaDB 클라이언트 ID: {id(self.client)}")  # 클라이언트 ID 로깅
            # 임베딩용 모델
            self.embeddings = OpenAIEmbeddings(
                model=CONFIG["openai"]["models"]["embedding"]
            )
            
            # LLM 설정
            self.LLM = ChatOpenAI(
                model=CONFIG["openai"]["models"]["analysis"]["default"],
                **CONFIG["openai"]["models"]["analysis"]["settings"]
            )
            
            # 컬렉션 초기화 - client 포함하여 통일
            self.collections = {
                "supplements_data": Chroma(
                    collection_name="supplements_data",
                    embedding_function=self.embeddings,
                    client=self.client
                ),
                "health_metrics": Chroma(
                    collection_name="health_metrics",
                    embedding_function=self.embeddings,
                    client=self.client
                ),
                "supplement_interactions": Chroma(
                    collection_name="supplement_interactions",
                    embedding_function=self.embeddings,
                    client=self.client
                )
            }
            logger.info("ChromaDB 컬렉션 초기화 완료")
            
            # RAG 인 기화
            self.retrieval_chain = create_retrieval_chain(
                self.collections["supplements_data"].as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                ),
                create_stuff_documents_chain(
                    self.LLM,
                    ChatPromptTemplate.from_template(self.get_rag_prompt_template())
                )
            )
            
        except Exception as e:
            logger.error(f"ChromaDB 초기화 실패: {e}")
            raise

    def __del__(self):
        """소멸자: 객체가 삭제될 때 메모리 정리 수행"""
        self.cleanup_memory()

    def similarity_search(self, query_text: str, n_results: int = 5):
        """ChromaDB 검색 수행"""
        try:
            logger.debug(f"Similarity Search - Query: {query_text}, N Results: {n_results}")
            
            # RAG 체인 사용
            response = self.retrieval_chain.invoke({
                "question": query_text,
                "context": self.collections["supplements_data"]
                    .similarity_search(query_text, k=n_results)
            })
            
            results = {
                "answer": response["answer"],
                "source_documents": response["source_documents"],
                "metadata": [doc.metadata for doc in response["source_documents"]]
            }
            
            if not results:  # 결과가 없는 경우
                logger.debug("No search results found")
                return {
                    "documents": [],
                    "metadatas": [],
                    "distances": []
                }
                
            # 결과 로깅
            logger.debug(f"Found {len(results)} results")
            for i, doc in enumerate(results):
                logger.debug(f"Result {i+1}:")
                logger.debug(f"  Content: {doc.page_content[:100]}...")
                logger.debug(f"  Metadata: {doc.metadata}")
                
            return {
                "documents": [doc.page_content for doc in results],
                "metadatas": [doc.metadata for doc in results],
                "distances": []
            }
        except Exception as e:
            logger.error(f"검색 실패: {str(e)}")
            return {
                "documents": [],
                "metadatas": [],
                "distances": []
            }

    def search_interactions(self, supplements: List[str]) -> Dict:
        """영양제 간 상호작용 검색"""
        try:
            interactions = []
            for i in range(len(supplements)):
                for j in range(i + 1, len(supplements)):
                    # 상호작용 분석을 위한 프롬프트 템플릿
                    interaction_prompt = self.get_interaction_prompt_template()
                    query = f"{supplements[i]} {supplements[j]} interaction effects"
                    
                    results = self.collections["supplement_interactions"].similarity_search_with_score(
                        query=query,
                        k=3
                    )
                    
                    if results:
                        analysis_prompt = interaction_prompt.format(
                            supplements=[supplements[i], supplements[j]],
                            context=[doc.page_content for doc, _ in results]
                        )
                        
                        analysis = self._analyze_with_LLM(analysis_prompt)
                        doc, score = results[0]
                        
                        interactions.append({
                            "supplements": [supplements[i], supplements[j]],
                            "interaction": {
                                "mechanism": analysis.get("mechanism", ""),
                                "description": doc.page_content
                            },
                            "evidence": {
                                "scientific_basis": analysis.get("scientific_basis", ""),
                                "metadata": doc.metadata,
                                "references": [d.metadata.get("pmid") for d, _ in results],
                                "confidence": analysis.get("confidence", 0.0)
                            },
                            "severity": analysis.get("severity", "unknown"),
                            "recommendations": analysis.get("recommendations", [])
                        })
                        logger.info(f"Found interaction between {supplements[i]} and {supplements[j]}")
            
            return {
                "has_interactions": bool(interactions),
                "interactions": interactions,
                "supplements_checked": supplements
            }
            
        except Exception as e:
            logger.error(f"상호작용 검색 실패: {str(e)}")
            raise

    def add_supplement_data(self, data: Dict):
        """영양제 데이터 추가"""
        try:
            supplement_name = data['supplement_name']
            pmid = data.get('pmid', 'N/A')
            
            # OpenAI API 할당량 초과 체크
            if "insufficient_quota" in str(data.get("error", "")):
                logger.error("OpenAI API 할당량 초과. 작업을 일시 중단합니다.")
                return {
                    "status": "error",
                    "error": "API_QUOTA_EXCEEDED",
                    "message": "OpenAI API 할당량이 초과되었습니다. 나중에 다시 시도해주세요."
                }
            
            def safe_value(value, default="not specified"):
                return value if value is not None else default
            
            # 저장 전 검증용 ID 생성
            doc_id = f"{supplement_name}_{pmid}"
            
            # 중복 체크
            existing = self.collections["supplements_data"].get(
                ids=[doc_id]
            )
            if existing and existing["ids"]:
                logger.info(f"기존 문서 발견 - ID: {doc_id}")
            
            try:
                # 기본 보충제 정보 저장
                self.collections["supplements_data"].add_texts(
                    texts=[data["abstract"]],
                    metadatas=[{
                        "supplement": safe_value(data["supplement_name"]),
                        "type": "base_info",
                        "pmid": safe_value(data["pmid"]),
                        "title": safe_value(data["title"])
                    }],
                    ids=[doc_id]
                )
            except openai.RateLimitError as e:
                logger.error(f"OpenAI API 할당량 초과: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"데이터 저장 실패: {str(e)}")
                return False

            # 즉시 저장 확인
            immediate_check = self.collections["supplements_data"].get(
                where={
                    "$and": [
                        {"supplement": {"$eq": supplement_name}},
                        {"pmid": {"$eq": pmid}}
                    ]
                }
            )
            logger.info(f"즉시 저장 확인 결과: {immediate_check}")

            # 저장 확인
            saved_data = self.collections["supplements_data"].get(
                where={
                    "$and": [
                        {"supplement": {"$eq": supplement_name}},
                        {"pmid": {"$eq": pmid}}
                    ]
                }
            )
            if not saved_data or not saved_data["ids"]:
                logger.error(f"데이터 저장 실패 [{supplement_name}] PMID: {pmid}")
                logger.error(f"- 제목: {data.get('title', 'N/A')[:100]}...")  # 제목 앞부분만
                
                # ChromaDB 컬렉션 상태도 확인
                try:
                    collection_info = self.client.get_collection("supplements_data")
                    logger.error(f"- 현재 컬렉션 문서 수: {collection_info.count()}")
                except Exception as e:
                    logger.error(f"- 컬렉션 상태 확인 실패: {str(e)}")
                return False

            # 상호작용 정보가 있으면 저장
            if interactions := data.get("interactions"):
                try:
                    interaction_text = (
                        f"Type: {safe_value(interactions.get('type'))}, "
                        f"Severity: {safe_value(interactions.get('severity'))}, "
                        f"Description: {safe_value(interactions.get('description'))}"
                    )
                    
                    interaction_id = f"interaction_{doc_id}"
                    self.collections["supplement_interactions"].add_texts(
                        texts=[interaction_text],
                        metadatas=[{
                            "supplement": data["supplement_name"],
                            "interaction_type": safe_value(interactions.get("type")),
                            "severity": safe_value(interactions.get("severity")),
                            "pmid": safe_value(data["pmid"]),
                            "title": safe_value(data["title"])
                        }],
                        ids=[interaction_id]
                    )
                except openai.RateLimitError as e:
                    logger.error(f"OpenAI API 할당량 초과: {str(e)}")
                    raise
                except Exception as e:
                    logger.warning(f"상호작용 데이터 저장 실패: {str(e)}")
            
            logger.info(f"✓ {supplement_name} (PMID: {pmid}) - 저장 완료")
            return True
            
        except Exception as e:
            logger.error(f"데이터 추가 실패: {str(e)}")
            logger.error(f"실패한 데이터: supplement={supplement_name}, pmid={pmid}")
            raise

    def reset_collection(self, force=False):
        """컬렉션 초기화 - 데이터 완전 삭제 및 재생성"""
        try:
            if not force:
                logger.error("force=True 없이 reset_collection이 호출됨")
                return
            
            logger.warning("=== 데이터베이스 초기화 시작 ===")
            logger.warning("이 작업은 모든 데이터를 제합니다!")
            
            # 컬렉션 삭제
            for name, collection in self.collections.items():
                logger.warning(f"컬렉션 '{name}' 삭제 중...")
                try:
                    if name in [c.name for c in self.client.list_collections()]:
                        self.client.delete_collection(name=name)
                except Exception as e:
                    logger.warning(f"컬렉션 '{name}' 삭제 중 오류 발생 (무시됨): {e}")
            
            # 컬렉션 재생성
            self.collections = {
                "supplements_data": Chroma(
                    collection_name="supplements_data",
                    embedding_function=self.embeddings,
                    client=self.client
                ),
                "health_metrics": Chroma(
                    collection_name="health_metrics",
                    embedding_function=self.embeddings,
                    client=self.client
                ),
                "supplement_interactions": Chroma(
                    collection_name="supplement_interactions",
                    embedding_function=self.embeddings,
                    client=self.client
                )
            }
            logger.warning("=== 데이터베이스 초기화 완료 ===")
            
        except Exception as e:
            logger.error(f"컬렉션 초기화 실패: {str(e)}")
            raise

        logger.warning(f"reset_collection 호출됨 - 호출 스택:\n{traceback.format_stack()}")

    def cleanup_memory(self):
        """메모리 리소스 정리 - 가지 컬렉션만 수행"""
        try:
            gc.collect()
            logger.info("메모리 리소스 정리 완료")
        except Exception as e:
            logger.error(f"메모리 정리 실패: {str(e)}")

    def _analyze_with_LLM(self, prompt: str) -> Dict:
        """LLM을 사용하여 분석 수행"""
        try:
            response = self.LLM.predict(prompt)
            # JSON 형식으로 응답 파싱
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                logger.error(f"LLM 응답 파싱 실패: {response}")
                return {}
        except Exception as e:
            logger.error(f"LLM 분석 실패: {str(e)}")
            return {}

    def search_comprehensive(self, query: str, context: Dict) -> Dict:
        """다층 구조에서 종합적 색 수행"""
        try:
            logger.debug(f"Comprehensive Search - Query: {query}")
            
            # 1단계: 관련 문서 검색
            search_results = self.collections["supplements_data"].similarity_search_with_score(
                query=f"""
                Based on these supplements and health conditions:
                - Supplements: {context.get('supplements', [])}
                - Health Metrics: {context.get('health_metrics', [])}
                
                Analyze potential interactions, mechanisms, and safety concerns.
                """,
                k=5
            )
            
            # 2단계: LLM을 기반으로 상세 분석
            analysis_prompt = f"""
            Analyze the potential interactions and safety concerns for the following supplements and health conditions.
            Provide your response in JSON format.
            
            Context:
            - Supplements: {context.get('supplements', [])}
            - Health conditions: {context.get('health_metrics', [])}
            
            Research findings:
            {[f"Document {i+1}: {doc.page_content}" for i, (doc, score) in enumerate(search_results)]}
            
            Format your response as JSON with the following structure:
            {{
                "needs_check": true/false,
                "reasoning": "detailed scientific explanation",
                "confidence": 0.0-1.0,
                "mechanisms": ["mechanism1", "mechanism2"],
                "severity_indication": "low/medium/high",
                "recommended_checks": ["check1", "check2"]
            }}
            """
            
            analysis_result = self._analyze_with_LLM(analysis_prompt)
            
            # 3단계: 결과 구조화
            assessment = {
                "needs_interaction_check": analysis_result.get("needs_check", False),
                "reasoning": {
                    "scientific_basis": analysis_result.get("reasoning", ""),
                    "mechanisms": analysis_result.get("mechanisms", []),
                    "confidence": analysis_result.get("confidence", 0.0)
                },
                "evidence": {
                    "supporting_documents": [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "relevance_score": 1-score
                        } for doc, score in search_results
                    ],
                    "pmids": list(set(doc.metadata.get("pmid") for doc, _ in search_results))
                },
                "recommendations": {
                    "additional_checks": analysis_result.get("recommended_checks", []),
                    "priority_level": self._calculate_priority(analysis_result)
                }
            }
            
            return assessment

        except Exception as e:
            logger.error(f"Comprehensive search failed: {str(e)}")
            raise

    def _calculate_priority(self, analysis_result: Dict) -> str:
        """우선순위 레벨 계산"""
        confidence = analysis_result.get("confidence", 0.0)
        severity = analysis_result.get("severity_indication", "low")
        mechanism_count = len(analysis_result.get("mechanisms", []))
        
        if confidence > 0.8 and severity == "high":
            return "HIGH"
        elif confidence > 0.6 and mechanism_count > 2:
            return "MEDIUM"
        return "LOW"

    def get_data_stats(self) -> Dict:
        """데이터베이스 통계 조회"""
        try:
            logger.info(f"데이터 조회 세션 - PID: {self.session_id}")
            collection_stats = {}
            
            # 디버그 로그 추가
            logger.debug("=== ChromaDB 컬렉션 목록 ===")
            for collection in self.client.list_collections():
                logger.debug(f"컬렉션 이름: {collection.name}")
                logger.debug(f"컬렉션 메데이터: {collection.metadata}")
                try:
                    items = collection.get()
                    logger.debug(f"컬렉션 아이템 수: {len(items['ids']) if items else 0}")
                except Exception as e:
                    logger.error(f"컬렉션 조회 실패: {e}")

            # 각 컬렉션의 데��터 수 확인
            for name, collection in self.collections.items():
                try:
                    # 기존 인스턴스 사용
                    results = collection.get()
                    count = len(results.get("ids", [])) if results else 0
                    logger.debug(f"컬렉션 '{name}' 문서 수: {count}")

                    collection_stats[name] = {
                        "count": count,
                        "status": "active" if count >= 0 else "error"
                    }
                    
                    # 샘플 데이터 추가 (최대 3개)
                    if count > 0:
                        collection_stats[name]["samples"] = [{
                            "id": results["ids"][i],
                            "metadata": results["metadatas"][i]
                        } for i in range(min(3, count))]
                        
                except Exception as e:
                    logger.error(f"컬렉션 '{name}' 상태 확인 실패: {e}")
                    collection_stats[name] = {"status": "error", "error": str(e)}

            stats = {
                "total_documents": sum(c["count"] for c in collection_stats.values() if c.get("count", 0) > 0),
                "collections": collection_stats,
                "supplements": {},
                "interactions": {
                    "total": collection_stats.get("supplement_interactions", {}).get("count", 0),
                    "by_type": {}
                }
            }
            
            # 보충제별 통계
            if collection_stats["supplements_data"]["status"] == "active":
                results = self.collections["supplements_data"].get()
                if results and results.get("metadatas"):
                    for metadata in results["metadatas"]:
                        supplement = metadata.get("supplement")
                        if supplement:
                            if supplement not in stats["supplements"]:
                                stats["supplements"][supplement] = {
                                    "count": 0,
                                    "pmids": set()
                                }
                            stats["supplements"][supplement]["count"] += 1
                            if metadata.get("pmid"):
                                stats["supplements"][supplement]["pmids"].add(metadata.get("pmid"))

            # set을 list로 변환 (JSON 직화를 위해)
            for supplement in stats["supplements"].values():
                supplement["pmids"] = list(supplement["pmids"])

            logger.info(f"데이터베이스 통계: {json.dumps(stats, indent=2, ensure_ascii=False)}")
            return stats
            
        except Exception as e:
            logger.error(f"통계 조회 실패: {str(e)}")
            raise

    async def format_docs(self, input_dict: dict) -> str:
        docs = input_dict["context"]
        question = input_dict["question"]
        return f"""Question: {question}
        
        Context: {docs}
        
        Answer: """ 

    def get_rag_prompt_template(self) -> str:
        """RAG 시스템 기반으로 한 프롬프트 템플릿"""
        return """당신은 영양제와 건강 관련 전문가입니다.
        주어진 문맥을 기반으로 질문에 답변해주세요.
        
        문맥: {context}
        
        질문: {question}
        
        답변 형식:
        1. 과학적 근거 설명
        2. 주의사항
        3. 권장사항
        
        답변:""" 

    def get_interaction_prompt_template(self) -> str:
        """상호작용 분석을 위한 프롬프트 템플릿"""
        return """당신은 영양제 상호작용 분석 전가입니다.
        
        음 영양제들 간의 상호작용을 석해주세요:
        영양제: {supplements}
        
        문맥: {context}
        
        다음 형식으로 JSON 응답을 제공해주세요:
        {
            "mechanism": "상호작용 메커니즘",
            "severity": "low/medium/high",
            "scientific_basis": "과학적 근거",
            "confidence": 0.0-1.0,
            "recommendations": ["권장사항1", "권장사항2"]
        }
        """ 