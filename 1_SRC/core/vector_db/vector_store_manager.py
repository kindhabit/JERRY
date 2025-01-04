import argparse
import asyncio
import logging
from typing import List, Dict, Set, Optional
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
            
            logger.info("=== 데이터베이스 재초기화 시작 ===")
            
            # 1. 기존 컬렉션 상태 확인 및 초기화
            self.collections = await self._initialize_collections()
            
            # 2. 데이터 초기화 및 임베딩
            await self.initialize_data()
            
            logger.info("=== 데이터베이스 재초기화 완료 ===")
            return True
            
        except Exception as e:
            logger.error(f"DB 재초기화 실패: {str(e)}")
            return False

    async def initialize_data(self):
        """데이터 초기화 및 임베딩 메인 프로세스"""
        try:
            logger.info("=== initialize_data 메서드 시작 ===")
            
            # 1. supplements 데이터 가져오기
            supplements = CONFIG.get_supplements()
            logger.debug(f"supplements 타입: {type(supplements)}")
            logger.debug(f"supplements 내용: {supplements}")
            
            # supplements가 딕셔너리가 아니면 중단
            if not isinstance(supplements, dict):
                raise ValueError(f"supplements가 딕셔너리가 아닙니다. 현재 타입: {type(supplements)}")
            
            logger.info(f"영양제 데이터 임베딩 시작 (총 {len(supplements)}개)")
            
            # 2. PubMed 소스 초기화
            pubmed_source = PubMedSource()
            logger.info("PubMed 소스 초기화 완료")
            
            try:
                # 3. 각 영양제별 처리
                for ko_name, en_name in supplements.items():
                    try:
                        logger.info(f"영양제 처리 시작: {ko_name} (영문: {en_name})")
                        async for paper_data in pubmed_source.search_supplement(ko_name):
                            try:
                                logger.debug(f"논문 데이터 처리 시작 - PMID: {paper_data.get('pmid', 'unknown')}")
                                logger.debug(f"논문 데이터 구조: {paper_data.keys()}")
                                success = await self._add_paper_to_collection("supplements", paper_data)
                                if success:
                                    logger.info(f"논문 데이터 처리 완료 - PMID: {paper_data.get('pmid', 'unknown')}")
                                else:
                                    logger.warning(f"논문 데이터 처리 실패 - PMID: {paper_data.get('pmid', 'unknown')}")
                                    continue
                            except Exception as e:
                                logger.error(f"논문 저장 실패 - 영양제: {ko_name}, 에러: {str(e)}")
                                continue
                    except Exception as e:
                        logger.error(f"영양제 처리 실패 ({ko_name}): {str(e)}")
                        continue
            finally:
                # 4. 세션 정리
                await pubmed_source.close()
            
            logger.info("=== initialize_data 메서드 완료 ===")
            
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
            parser.add_argument('--action', choices=['reinit', 'stats'], required=True, help='2: reinit - 재초기화, 4: stats - 상태 확인')
            parser.add_argument('--force', action='store_true')
            parser.add_argument('--debug', action='store_true')
            args = parser.parse_args()
            
            manager = await cls.create()
            
            if args.action == 'reinit':  # 액션 2: 재초기화
                # 데이터베이스 재초기화
                success = await manager.reinitialize_database(force=True)
                if not success:
                    logger.error("데이터베이스 재초기화 실패")
                    return
                
                # 최종 상태 확인
                logger.info("\n=== 최종 상태 확인 ===")
                final_stats = await manager.show_stats()
                logger.info(f"데이터베이스 초기화 완료\n{json.dumps(final_stats, indent=2, ensure_ascii=False)}")
                
            elif args.action == 'stats':  # 액션 4: 상태 확인
                logger.info("=== 컬렉션 상태 확인 시작 ===")
                
                # 1. 컬렉션 존재 여부 확인
                collections = manager.client.list_collections()
                if not collections:
                    logger.error("컬렉션이 존재하지 않습니다.")
                    return
                
                # 2. 각 컬렉션의 상태 확인
                for collection in collections:
                    logger.info(f"\n=== {collection.name} 컬렉션 상태 ===")
                    result = collection.get()
                    
                    # 데이터 수 확인
                    doc_count = len(result['ids']) if result['ids'] else 0
                    logger.info(f"문서 수: {doc_count}")
                    
                    # 메타데이터 필드 확인
                    if result['metadatas']:
                        metadata_fields = list(set().union(*[set(m.keys()) for m in result['metadatas']]))
                        logger.info(f"메타데이터 필드: {metadata_fields}")
                    
                    # 임베딩 확인
                    if result['embeddings']:
                        embedding_dim = len(result['embeddings'][0])
                        logger.info(f"임베딩 차원: {embedding_dim}")
                
                logger.info("\n=== 컬렉션 상태 확인 완료 ===")
                
        except Exception as e:
            logger.error(f"작업 실패: {str(e)}")
            raise

if __name__ == "__main__":
    asyncio.run(ChromaManager.main()) 