import argparse
import asyncio
import logging
from typing import List, Dict, Set, Optional, Any
from utils.logger_config import setup_logger
from config.config_loader import CONFIG
from core.vector_db.embedding_creator import EmbeddingCreator
from datetime import datetime
from utils.openai_client import OpenAIClient
from models.health_data import HealthData
from core.data_source.data_source_manager import DataSourceManager, PubMedSource
import os
import chromadb
import json
from config.config_loader import ConfigLoader
import uuid

logger = setup_logger('vector_store')

class ChromaManager:
    """ChromaDB 관리자"""
    
    COLLECTIONS_STRUCTURE = {
        'supplements': {
            'description': '영양제 기본 정보',
            'metadata': {
                'type': 'supplement',
                'name': '',
                'category': '',
                'analysis_method': '',
                'analysis_time': '',
                'confidence': ''
            }
        },
        'interactions': {
            'description': '영양제 간 상호작용',
            'metadata': {
                'type': 'interaction',
                'supplements': [],
                'interaction_type': '',
                'severity': ''
            }
        },
        'health_data': {
            'description': '건강 관련 데이터',
            'metadata': {
                'type': 'health_data',
                'category': '',
                'keywords': [],
                'source': ''
            }
        },
        'health_metrics': {
            'description': '건강 지표',
            'metadata': {
                'type': 'health_metric',
                'metric_name': '',
                'category': '',
                'related_factors': []
            }
        },
        'medical_terms': {
            'description': '의학 용어 사전',
            'metadata': {
                'type': 'medical_term',
                'term_ko': '',
                'term_en': '',
                'category': ''
            }
        }
    }
    
    def __init__(self):
        """ChromaManager 초기화"""
        self.client = None
        self.collections = {}
        self.embedding_creator = None
        self.openai_client = None
        self.config = ConfigLoader()
        
        try:
            self.client = self._initialize_chroma_client()
            self.embedding_creator = EmbeddingCreator()
            self.openai_client = OpenAIClient()
            # 기존 컬렉션 로드
            self.collections = {
                coll.name: coll 
                for coll in self.client.list_collections()
            }
            logger.info("ChromaManager 기본 초기화 완료")
            logger.debug(f"임베딩 생성기 초기화 상태: {self.embedding_creator.get_cache_stats()}")
        except chromadb.errors.ChromaError as e:
            logger.error(f"ChromaDB 초기화 실패: {str(e)}")
            raise

    @staticmethod
    def _initialize_chroma_client():
        """ChromaDB 클라이언트 초기화"""
        try:
            chroma_settings = CONFIG.get_service_settings()["chroma"]
            logger.info(f"ChromaDB 서버 연결: {chroma_settings['host']}:{chroma_settings['port']}")
            
            settings = chromadb.Settings(
                chroma_api_impl=chroma_settings["chroma_api_impl"],
                **chroma_settings.get("settings", {})
            )
            
            return chromadb.HttpClient(
                host=chroma_settings["host"],
                port=chroma_settings["port"],
                settings=settings
            )
        except Exception as e:
            logger.error(f"ChromaDB 클라이언트 초기화 실패: {str(e)}")
            raise

    async def _initialize_collections(self):
        """컬렉션 초기화"""
        try:
            collections = {}
            
            # 1. 기존 컬렉션 삭제
            try:
                current_collections = self.client.list_collections()
                for collection in current_collections:
                    try:
                        logger.info(f"{collection.name} 컬렉션 삭제 중...")
                        self.client.delete_collection(collection.name)
                        logger.info(f"{collection.name} 컬렉션 삭제 완료")
                    except Exception as e:
                        logger.warning(f"{collection.name} 컬렉션 삭제 실패: {str(e)}")
            except Exception as e:
                logger.warning(f"기존 컬렉션 삭제 중 오류 발생: {str(e)}")
            
            # 2. 새 컬렉션 생성
            for name, info in self.COLLECTIONS_STRUCTURE.items():
                try:
                    logger.info(f"{name} 컬렉션 생성 중... ({info['description']})")
                    # 혹시 모를 기존 컬렉션 삭제 재시도
                    try:
                        self.client.delete_collection(name)
                    except:
                        pass
                    
                    # 임베딩 함수 설정
                    embedding_function = self.embedding_creator
                    logger.debug(f"{name} 컬렉션에 임베딩 함수 설정: {embedding_function}")
                    
                    collections[name] = self.client.create_collection(
                        name=name,
                        metadata={"description": info['description']},
                        embedding_function=embedding_function
                    )
                    logger.info(f"{name} 컬렉션 생성 완료")
                except Exception as e:
                    logger.error(f"{name} 컬렉션 생성 실패: {str(e)}")
                    raise
            
            if not collections:
                raise Exception("컬렉션 생성 실패: 생성된 컬렉션이 없습니다")
            
            logger.info(f"컬렉션 초기화 완료: {list(collections.keys())}")
            return collections
            
        except Exception as e:
            logger.error(f"컬렉션 초기화 실패: {str(e)}")
            raise

    async def reinitialize_database(self, force: bool = False):
        """데이터베이스 재초기화"""
        try:
            if not force:
                raise ValueError("force 옵션이 필요합니다")
            
            logger.info("\n=== 데이터베이스 재초기화 시작 ===")
            logger.warning("⚠️  주의: 이 작업은 모든 기존 데이터를 삭제합니다!")
            
            # 사용자 확인
            confirmation = input("\n정말로 모든 데이터를 삭제하시겠습니까? (yes/no): ")
            if confirmation.lower() != 'yes':
                logger.info("작업이 취소되었습니다.")
                return False
            
            # 비밀번호 확인
            password = input("\n관리자 비밀번호를 입력하세요: ")
            if password != "ckrgkstmqrhks!23":
                logger.error("비밀번호가 올바르지 않습니다.")
                return False
            
            logger.info("\n인증 성공. 재초기화를 시작합니다...")
            
            # 1. 기존 컬렉션 상태 확인 및 초기화
            self.collections = await self._initialize_collections()
            
            # 2. 데이터 초기화 및 임베딩
            await self.initialize_data()
            
            logger.info("=== 데이터베이스 재초기화 완료 ===")
            return True
            
        except Exception as e:
            logger.error(f"DB 재초기화 실패: {str(e)}")
            return False

    async def update_database(self, collection_limits: Dict[str, int] = None):
        """데이터베이스 업데이트 - 기존 데이터 유지하며 새로운 데이터만 추가"""
        try:
            logger.info("=== 데이터베이스 업데이트 시작 ===")
            
            if collection_limits:
                logger.info("=== 컬렉션별 처리 제한 설정 ===")
                for coll, limit in collection_limits.items():
                    if limit:
                        logger.info(f"- {coll}: {limit}개 제한")
            
            # 1. 기존 PMID 수집
            existing_pmids = set()
            pmid_stats = {}  # 컬렉션별 PMID 통계
            
            for collection_name in self.COLLECTIONS_STRUCTURE.keys():
                try:
                    collection = self.client.get_collection(collection_name)
                    pmids = set(collection.get()["ids"])
                    existing_pmids.update(pmids)
                    pmid_stats[collection_name] = len(pmids)
                    logger.info(f"{collection_name} 컬렉션의 기존 PMID 수: {len(pmids)}")
                except Exception as e:
                    logger.error(f"{collection_name} 컬렉션 PMID 수집 실패: {str(e)}")
                    continue
            
            logger.info("=== PMID 현황 요약 ===")
            logger.info(f"전체 기존 PMID 수: {len(existing_pmids)}")
            for coll, count in pmid_stats.items():
                logger.info(f"- {coll}: {count}개")
            
            # 2. 업데이트 시작
            update_stats = {
                "total_checked": 0,
                "existing": 0,
                "new": 0,
                "failed": 0
            }
            
            await self.initialize_data(
                existing_pmids=existing_pmids,
                is_update=True,
                update_stats=update_stats,
                collection_limits=collection_limits
            )
            
            # 3. 업데이트 결과 로깅
            logger.info("\n=== 업데이트 결과 ===")
            logger.info(f"검사한 총 PMID 수: {update_stats['total_checked']}")
            logger.info(f"기존 PMID 수: {update_stats['existing']}")
            logger.info(f"새로 추가된 PMID 수: {update_stats['new']}")
            logger.info(f"처리 실패 PMID 수: {update_stats['failed']}")
            
            logger.info("=== 데이터베이스 업데이트 완료 ===")
            return True
            
        except Exception as e:
            logger.error(f"DB 업데이트 실패: {str(e)}")
            return False

    async def initialize_data(
        self,
        existing_pmids: Set[str] = set(),
        is_update: bool = False,
        update_stats: Dict = None,
        collection_limits: Dict[str, int] = None
    ):
        """데이터 초기화 및 임베딩 메인 프로세스"""
        try:
            logger.info("=== initialize_data 메서드 시작 ===")
            mode = "업데이트" if is_update else "초기화"
            logger.info(f"실행 모드: {mode}")
            
            # PubMed 소스 초기화
            pubmed_source = PubMedSource()
            logger.info("PubMed 소스 초기화 완료")
            
            try:
                # 1. Supplements 처리
                supplements = CONFIG.get_supplements()
                if not isinstance(supplements, dict):
                    raise ValueError(f"supplements가 딕셔너리가 아닙니다. 현재 타입: {type(supplements)}")
                
                logger.info(f"영양제 데이터 {mode} 시작 (총 {len(supplements)}개)")
                supplements_limit = collection_limits.get("supplements") if collection_limits else None
                supplements_count = 0
                
                # supplements 처리 여부 결정
                should_process_supplements = supplements_limit is None or supplements_limit > 0
                if not should_process_supplements:
                    logger.info("supplements 컬렉션 처리 건너뜀 (제한: 0)")
                
                if should_process_supplements:
                    for ko_name, en_name in supplements.items():
                        try:
                            logger.info(f"\n=== 영양제 처리 시작: {ko_name} (영문: {en_name}) ===")
                            async for paper_data in pubmed_source.search_supplement(ko_name):
                                try:
                                    if update_stats is not None:
                                        update_stats['total_checked'] += 1
                                    
                                    pmid = paper_data.get('pmid')
                                    if is_update and pmid in existing_pmids:
                                        logger.info(f"기존 PMID 스킵: {pmid}")
                                        if update_stats is not None:
                                            update_stats['existing'] += 1
                                        continue
                                    
                                    logger.info(f"새로운 PMID 처리 시작: {pmid}")
                                    success = await self._add_paper_to_collection("supplements", paper_data)
                                    
                                    if success:
                                        logger.info(f"새로운 PMID 처리 완료: {pmid}")
                                        if update_stats is not None:
                                            update_stats['new'] += 1
                                        supplements_count += 1
                                    else:
                                        logger.warning(f"새로운 PMID 처리 실패: {pmid}")
                                        if update_stats is not None:
                                            update_stats['failed'] += 1
                                    
                                    if supplements_limit and supplements_count >= supplements_limit:
                                        logger.info(f"supplements 컬렉션 제한 수({supplements_limit}개) 도달")
                                        break
                                        
                                except Exception as e:
                                    logger.error(f"논문 처리 실패 - PMID: {paper_data.get('pmid', 'unknown')}: {str(e)}")
                                    if update_stats is not None:
                                        update_stats['failed'] += 1
                                    continue
                                
                            if supplements_limit and supplements_count >= supplements_limit:
                                break
                                
                        except Exception as e:
                            logger.error(f"영양제 처리 실패 ({ko_name}): {str(e)}")
                            continue
                
                # 2. Interactions 처리
                interactions_limit = collection_limits.get("interactions") if collection_limits else None
                interactions_count = 0
                
                # interactions 처리 여부 결정
                should_process_interactions = interactions_limit is None or interactions_limit > 0
                if not should_process_interactions:
                    logger.info("interactions 컬렉션 처리 건너뜀 (제한: 0)")
                
                if should_process_interactions:
                    logger.info(f"\n=== 상호작용 데이터 {mode} 시작 ===")
                    for ko_name, en_name in supplements.items():
                        try:
                            async for paper_data in pubmed_source.search_interactions(ko_name):
                                try:
                                    if update_stats is not None:
                                        update_stats['total_checked'] += 1
                                    
                                    pmid = paper_data.get('pmid')
                                    if is_update and pmid in existing_pmids:
                                        logger.info(f"기존 PMID 스킵: {pmid}")
                                        if update_stats is not None:
                                            update_stats['existing'] += 1
                                        continue
                                    
                                    logger.info(f"새로운 PMID 처리 시작: {pmid}")
                                    success = await self._add_paper_to_collection("interactions", paper_data)
                                    
                                    if success:
                                        logger.info(f"새로운 PMID 처리 완료: {pmid}")
                                        if update_stats is not None:
                                            update_stats['new'] += 1
                                        interactions_count += 1
                                    else:
                                        logger.warning(f"새로운 PMID 처리 실패: {pmid}")
                                        if update_stats is not None:
                                            update_stats['failed'] += 1
                                    
                                    if interactions_limit and interactions_count >= interactions_limit:
                                        logger.info(f"interactions 컬렉션 제한 수({interactions_limit}개) 도달")
                                        break
                                        
                                except Exception as e:
                                    logger.error(f"논문 처리 실패 - PMID: {paper_data.get('pmid', 'unknown')}: {str(e)}")
                                    if update_stats is not None:
                                        update_stats['failed'] += 1
                                    continue
                                
                            if interactions_limit and interactions_count >= interactions_limit:
                                break
                                
                        except Exception as e:
                            logger.error(f"상호작용 처리 실패 ({ko_name}): {str(e)}")
                            continue
                
                # 3. Health Data 처리
                health_data_limit = collection_limits.get("health_data") if collection_limits else None
                health_data_count = 0
                
                # health_data 처리 여부 결정
                should_process_health_data = health_data_limit is None or health_data_limit > 0
                if not should_process_health_data:
                    logger.info("health_data 컬렉션 처리 건너뜀 (제한: 0)")
                
                if should_process_health_data:
                    logger.info(f"\n=== 건강 데이터 {mode} 시작 ===")
                    for ko_name, en_name in supplements.items():
                        try:
                            async for paper_data in pubmed_source.search_health_data(ko_name):
                                try:
                                    if update_stats is not None:
                                        update_stats['total_checked'] += 1
                                    
                                    pmid = paper_data.get('pmid')
                                    if is_update and pmid in existing_pmids:
                                        logger.info(f"기존 PMID 스킵: {pmid}")
                                        if update_stats is not None:
                                            update_stats['existing'] += 1
                                        continue
                                    
                                    logger.info(f"새로운 PMID 처리 시작: {pmid}")
                                    success = await self._add_paper_to_collection("health_data", paper_data)
                                    
                                    if success:
                                        logger.info(f"새로운 PMID 처리 완료: {pmid}")
                                        if update_stats is not None:
                                            update_stats['new'] += 1
                                        health_data_count += 1
                                    else:
                                        logger.warning(f"새로운 PMID 처리 실패: {pmid}")
                                        if update_stats is not None:
                                            update_stats['failed'] += 1
                                    
                                    if health_data_limit and health_data_count >= health_data_limit:
                                        logger.info(f"health_data 컬렉션 제한 수({health_data_limit}개) 도달")
                                        break
                                        
                                except Exception as e:
                                    logger.error(f"논문 처리 실패 - PMID: {paper_data.get('pmid', 'unknown')}: {str(e)}")
                                    if update_stats is not None:
                                        update_stats['failed'] += 1
                                    continue
                                
                            if health_data_limit and health_data_count >= health_data_limit:
                                break
                                
                        except Exception as e:
                            logger.error(f"건강 데이터 처리 실패 ({ko_name}): {str(e)}")
                            continue
                
            finally:
                await pubmed_source.close()
            
            # 최종 처리 결과 로깅
            logger.info(f"\n=== initialize_data 메서드 완료 ({mode}) ===")
            if collection_limits:
                if supplements_limit is not None:
                    status = "건너뜀" if supplements_limit == 0 else f"{supplements_count}/{supplements_limit}"
                    logger.info(f"supplements 컬렉션: {status}")
                if interactions_limit is not None:
                    status = "건너뜀" if interactions_limit == 0 else f"{interactions_count}/{interactions_limit}"
                    logger.info(f"interactions 컬렉션: {status}")
                if health_data_limit is not None:
                    status = "건너뜀" if health_data_limit == 0 else f"{health_data_count}/{health_data_limit}"
                    logger.info(f"health_data 컬렉션: {status}")
            
        except Exception as e:
            logger.error(f"initialize_data 메서드 실패: {str(e)}")
            logger.error(f"에러 발생 위치: {e.__traceback__.tb_frame.f_code.co_name}")
            logger.error(f"에러 발생 라인: {e.__traceback__.tb_lineno}")
            raise

    async def _add_paper_to_collection(self, collection_name: str, paper: Dict) -> bool:
        """논문 데이터를 컬렉션에 추가"""
        try:
            collection = self.client.get_collection(collection_name)
            
            # 임베딩 생성
            embeddings = await self.openai_client.get_embeddings(paper["processed_text"])
            if not embeddings:
                logger.error(f"임베딩 생성 실패 - PMID: {paper.get('pmid')}")
                return False
                
            # numpy 배열로 변환
            import numpy as np
            embeddings_array = np.array(embeddings)
            
            # 저자 정보를 문자열로 변환
            if isinstance(paper["authors"], list):
                if len(paper["authors"]) > 0:
                    if isinstance(paper["authors"][0], dict):
                        authors_str = ", ".join([author.get("name", "") for author in paper["authors"]])
                    else:
                        authors_str = ", ".join(map(str, paper["authors"]))
                else:
                    authors_str = ""
            else:
                authors_str = str(paper["authors"])
            
            # 메타데이터 준비
            metadata = {
                "pmid": paper["pmid"],
                "title": paper["title"],
                "abstract": paper["abstract"],
                "authors": authors_str,
                "publication_date": paper["publication_date"],
                "journal": paper["journal"],
                "category": paper["category"],
                "weight": paper["weight"],
                "description": paper["description"],
                "llm_analysis": paper["llm_analysis"]
            }
            
            # 데이터 추가
            collection.add(
                embeddings=embeddings_array,
                documents=[paper["processed_text"]],
                metadatas=[metadata],
                ids=[paper["pmid"]]
            )
            
            logger.info(f"논문 데이터 저장 완료 - PMID: {paper['pmid']}")
            return True
            
        except Exception as e:
            logger.error(f"논문 저장 실패 - PMID: {paper.get('pmid')}: {str(e)}")
            logger.error("저장 실패한 데이터 구조: " + json.dumps({
                "pmid": "str",
                "title": "str",
                "abstract": "str",
                "authors": "list",
                "publication_date": "str",
                "journal": "str",
                "category": "str",
                "weight": "float",
                "description": "str",
                "processed_text": "str",
                "llm_analysis": "str",
                "author_names": "list"
            }, indent=2))
            return False

    async def show_stats(self) -> Dict:
        """컬렉션 통계 조회"""
        try:
            stats = {}
            for name, collection in self.collections.items():
                result = collection.get()
                stats[name] = {
                    "count": len(result['ids']) if result['ids'] else 0,
                    "metadata_fields": list(set().union(*[set(m.keys()) for m in result['metadatas']])) if result['metadatas'] else [],
                    "last_updated": datetime.now().isoformat()
                }
            return stats
        except Exception as e:
            logger.error(f"통계 조회 실패: {str(e)}")
            raise

    @classmethod
    async def create(cls):
        """비동기 팩토리 메소드"""
        self = cls()
        try:
            logger.info("ChromaManager 초기화 시작")
            return self
        except Exception as e:
            logger.error(f"ChromaManager 초기화 실패: {str(e)}")
            raise

    @classmethod
    async def main(cls):
        """메인 실행 함수"""
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("--action", choices=["reinit", "update", "stats"], required=True)
            parser.add_argument("--force", action="store_true")
            parser.add_argument("--debug", action="store_true")
            parser.add_argument("--supplements-limit", type=int, help="영양제 데이터 처리 제한 수")
            parser.add_argument("--interactions-limit", type=int, help="상호작용 데이터 처리 제한 수")
            parser.add_argument("--health-data-limit", type=int, help="건강 데이터 처리 제한 수")
            args = parser.parse_args()

            manager = ChromaManager()
            if args.action == "reinit":
                await manager.reinitialize_database(force=args.force)
            elif args.action == "update":
                limits = {
                    "supplements": args.supplements_limit,
                    "interactions": args.interactions_limit,
                    "health_data": args.health_data_limit
                }
                await manager.update_database(collection_limits=limits)
            elif args.action == "stats":
                stats = await manager.show_stats()
                logger.info(f"\n=== Vector Store 상태 ===\n{json.dumps(stats, indent=2, ensure_ascii=False)}")

        except Exception as e:
            logger.error(f"작업 실패: {str(e)}")
            raise

    async def get_supplement_interaction(self, health_data: Dict[str, Any], current_supplements: List[str]) -> Dict[str, Any]:
        # 최소 필요 문서 수와 관련성 임계값 설정
        MIN_REQUIRED_DOCS = 3
        MINIMUM_RELEVANCE_THRESHOLD = 0.7

        try:
            if not isinstance(health_data, dict):
                raise ValueError("health_data must be a dictionary")

            # 건강 데이터에서 필요한 정보 추출
            symptoms = health_data.get('symptoms', [])
            health_metrics = health_data.get('health_metrics', {})
            lifestyle_factors = health_data.get('lifestyle_factors', {})

            logger.info(f"영양제 상호작용 분석 시작: {current_supplements}")
            logger.debug(f"건강 데이터: {json.dumps(health_data, ensure_ascii=False)}")

            # 검색 쿼리 구성
            query = f"""
            증상: {', '.join(symptoms)}
            현재 복용 중인 영양제: {', '.join(current_supplements)}
            건강 지표: {json.dumps(health_metrics, ensure_ascii=False)}
            생활습관: {json.dumps(lifestyle_factors, ensure_ascii=False)}
            위 정보와 관련된 영양제 상호작용 및 추천 연구
            """

            logger.info("ChromaDB 검색 수행 중...")
            # interactions 컬렉션에서 검색 수행
            collection = self.client.get_collection("interactions")
            results = collection.query(
                query_texts=[query],
                n_results=5
            )

            # 검색 결과 검증
            if not results['documents'] or len(results['documents'][0]) < MIN_REQUIRED_DOCS:
                logger.warning(f"불충분한 데이터: {len(results['documents'][0]) if results['documents'] else 0}/{MIN_REQUIRED_DOCS}")
                return {
                    "status": "insufficient_data",
                    "message": "죄송합니다. 현재 해당 건강 정보와 영양제 조합에 대한 충분한 연구 데이터가 없습니다.",
                    "available_data": {
                        "found_documents": len(results['documents'][0]) if results['documents'] else 0,
                        "required_minimum": MIN_REQUIRED_DOCS,
                        "suggestion": "더 많은 데이터가 수집된 후에 다시 시도해주세요."
                    }
                }

            # 검색 결과의 관련성 검증
            if results['distances'] and max(results['distances'][0]) < MINIMUM_RELEVANCE_THRESHOLD:
                logger.warning(f"낮은 관련성: {max(results['distances'][0])}/{MINIMUM_RELEVANCE_THRESHOLD}")
                return {
                    "status": "low_relevance",
                    "message": "입력하신 건강 정보와 직접적으로 관련된 연구 결과를 찾지 못했습니다.",
                    "suggestion": "좀 더 일반적인 건강 지표나 증상으로 다시 검색해보시겠습니까?"
                }

            # LLM 프롬프트 구성
            context = "\n".join(results['documents'][0])
            logger.info("LLM 분석 시작...")
            
            # 응답 형식 템플릿
            response_format = {
                "recommendations": {
                    "영양제_이름": {
                        "confidence": "high/medium/low",
                        "reason": "추천 이유",
                        "evidence": "연구 근거",
                        "interactions": "상호작용 정보",
                        "precautions": "주의사항"
                    }
                },
                "data_quality": {
                    "confidence_level": "high/medium/low",
                    "limitations": ["고려해야 할 제한사항들"]
                }
            }

            # 프롬프트 구성
            prompt = (
                "다음 사용자의 건강 정보와 검색된 연구 결과를 바탕으로 영양제 추천 분석을 수행해주세요:\n\n"
                f"사용자 건강 정보:\n"
                f"- 증상: {symptoms}\n"
                f"- 현재 복용 중: {current_supplements}\n"
                f"- 건강 지표: {health_metrics}\n"
                f"- 생활습관: {lifestyle_factors}\n\n"
                f"검색된 연구 결과:\n{context}\n\n"
                "다음 사항들을 고려하여 응답해주세요:\n"
                "1. 확실한 근거가 있는 내용만 포함할 것\n"
                "2. 데이터가 불충분한 경우 명확히 표시할 것\n"
                "3. 추측이나 일반화된 조언은 피할 것\n"
                "4. 각 추천에 대한 신뢰도 수준을 명시할 것\n\n"
                f"응답 형식:\n{json.dumps(response_format, indent=2, ensure_ascii=False)}"
            )

            # LLM을 통한 분석 수행
            response = await self.openai_client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            logger.info("LLM 분석 완료")
            analysis_result = json.loads(response.choices[0].message.content)
            
            return {
                "status": "success",
                "analysis_id": str(uuid.uuid4()),
                **analysis_result
            }
            
        except ValueError as ve:
            logger.error(f"잘못된 입력 데이터: {str(ve)}")
            return {
                "status": "error",
                "message": f"잘못된 입력 데이터: {str(ve)}"
            }
        except Exception as e:
            logger.error(f"영양제 상호작용 분석 중 오류 발생: {str(e)}")
            return {
                "status": "error",
                "message": "분석 중 오류가 발생했습니다.",
                "error_details": str(e)
            }

    async def get_health_impacts(self, supplement: str, health_data: Dict) -> List[Dict]:
        """영양제가 건강 상태에 미치는 영향 조회"""
        try:
            # health_data 컬렉션에서 검색
            collection = self.client.get_collection("health_data")
            query = f"{supplement} health effects"
            
            results = collection.query(
                query_texts=[query],
                n_results=3
            )
            
            impacts = []
            for i, doc in enumerate(results['documents']):
                if doc:
                    impacts.append({
                        "supplement": supplement,
                        "health_aspect": results['metadatas'][i].get('category', 'general'),
                        "impact": doc,
                        "evidence": {
                            "source": "PubMed",
                            "pmid": results['ids'][i],
                            "summary": doc
                        }
                    })
            
            return impacts
            
        except Exception as e:
            logger.error(f"건강 영향 검색 실패 ({supplement}): {str(e)}")
            return []

if __name__ == "__main__":
    asyncio.run(ChromaManager.main()) 