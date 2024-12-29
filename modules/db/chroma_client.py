from typing import List, Dict, Tuple, Optional
import uuid
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from modules.config.settings import CONFIG
from modules.utils.logger import logger

class ChromaDBClient:
    def __init__(self):
        """ChromaDB 클라이언트 초기화"""
        try:
            logger.info("[CHROMA] 클라이언트 초기화 시작")
            
            # 임베딩 생성기 초기화
            logger.info("[CHROMA] 임베딩 생성기 초기화 시작")
            self.openai_embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002"
            )
            logger.info("[CHROMA] 임베딩 생성기 초기화 완료")
            
            # ChromaDB 설정 로드
            logger.info("[CHROMA] ChromaDB 설정 로드 시작")
            chroma_settings = Settings(
                chroma_api_impl="rest",
                chroma_server_host=CONFIG.config['service']['chroma']['host'],
                chroma_server_http_port=CONFIG.config['service']['chroma']['port']
            )
            
            # ChromaDB 클라이언트 생성
            self.client = chromadb.Client(settings=chroma_settings)
            
            # 컬렉션 초기화
            collections = CONFIG.config['service']['chroma']['collections']
            self.initial_collection = self.client.get_or_create_collection(
                name=collections['initial']
            )
            self.interaction_collection = self.client.get_or_create_collection(
                name=collections['interaction']
            )
            self.adjustment_collection = self.client.get_or_create_collection(
                name=collections['adjustment']
            )
            
            logger.info("[CHROMA] 클라이언트 초기화 완료")
            
        except Exception as e:
            logger.error(f"[CHROMA] 초기화 실패: {str(e)}")
            raise

    async def create_enhanced_embeddings(
        self, 
        texts: List[str], 
        metadata: List[Dict] = None
    ) -> Tuple[List[List[float]], List[str], List[Dict]]:
        """텍스트 임베딩 생성"""
        try:
            logger.info("\n=== ChromaDB 임베딩 생성 시작 ===")
            logger.info(f"  - 입력 텍스트 수: {len(texts)}")
            logger.info(f"  - 텍스트 샘플: {texts[:2]}")
            
            # 임베딩 생성
            embeddings = await self.openai_embeddings.aembed_documents(texts)
            
            # ID 생성
            ids = [str(uuid.uuid4()) for _ in texts]
            
            # 메타데이터 처리
            if metadata is None:
                metadata = [{} for _ in texts]
                
            logger.info(f"  - 생성된 임베딩 수: {len(embeddings)}")
            logger.info(f"  - 임베딩 차원: {len(embeddings[0])}")
            logger.info(f"  - 임베딩 타입: {type(embeddings)}")
            logger.info(f"  - 첫 번째 임베딩 타입: {type(embeddings[0])}")
            logger.info(f"  - 첫 번째 임베딩 샘플: {embeddings[0][:5]}...")
            logger.info(f"  - 생성된 ID 수: {len(ids)}")
            logger.info(f"  - 메타데이터 수: {len(metadata)}")
            
            return embeddings, ids, metadata
            
        except Exception as e:
            logger.error("\n=== ChromaDB 임베딩 생성 실패 ===")
            logger.error(f"오류: {str(e)}")
            logger.error("스택 트레이스:", exc_info=True)
            raise

    async def add_supplement_data(
        self, 
        embeddings: List[List[float]], 
        texts: List[str], 
        metadata: List[Dict]
    ) -> bool:
        """영양제 데이터 저장"""
        try:
            logger.info("\n=== ChromaDB 데이터 저장 시작 ===")
            logger.info(f"  - 저장할 데이터 수: {len(texts)}")
            logger.info(f"  - 임베딩 수: {len(embeddings)}")
            logger.info(f"  - 임베딩 타입: {type(embeddings)}")
            logger.info(f"  - 첫 번째 임베딩 타입: {type(embeddings[0])}")
            logger.info(f"  - 메타데이터 수: {len(metadata)}")
            
            # ID 생성
            ids = [str(uuid.uuid4()) for _ in texts]
            
            logger.info("\n초기 데이터 저장 시도:")
            logger.info(f"  - Collection: {self.initial_collection}")
            logger.info(f"  - Collection 타입: {type(self.initial_collection)}")
            
            # 데이터 저장
            self.initial_collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadata,
                ids=ids
            )
            
            logger.info(f"  - {len(texts)}개 데이터 저장 완료")
            return True
            
        except Exception as e:
            logger.error("\n=== ChromaDB 데이터 저장 실패 ===")
            logger.error(f"오류: {str(e)}")
            logger.error("스택 트레이스:", exc_info=True)
            return False

    async def search_interactions(
        self, 
        supplements: List[str], 
        health_conditions: Dict[str, str]
    ) -> Dict:
        """영양제 상호작용 검색"""
        try:
            logger.info("\n=== ChromaDB 상호작용 검색 시작 ===")
            logger.info(f"  - 검색 영양제: {supplements}")
            logger.info(f"  - 건강 상태: {health_conditions}")
            
            # 검색 쿼리 생성
            query = f"Supplements: {', '.join(supplements)}"
            if health_conditions:
                conditions = [f"{cond}({type_})" for cond, type_ in health_conditions.items()]
                query += f" | Conditions: {', '.join(conditions)}"
            logger.info(f"  - 검색 쿼리: {query}")
            
            # 임베딩 생성
            embeddings = await self.openai_embeddings.aembed_query(query)
            logger.info(f"  - 쿼리 임베딩 생성 완료: {len(embeddings)} 차원")
            
            # 검색 실행
            results = self.interaction_collection.query(
                query_embeddings=embeddings,
                n_results=5,
                where={"type": {"$in": supplements}},  # 영양제 타입으로 필터링
                where_document={"$contains": query},   # 문서 내용으로 필터링
                include=['documents', 'metadatas', 'distances']
            )
            logger.info(f"  - 검색 결과: {len(results['documents'])} 건")
            
            return {
                'query': query,
                'documents': results['documents'],
                'metadatas': results['metadatas'],
                'distances': results['distances']
            }
            
        except Exception as e:
            logger.error("\n=== ChromaDB 상호작용 검색 실패 ===")
            logger.error(f"오류: {str(e)}")
            logger.error("스택 트레이스:", exc_info=True)
            raise

    async def get_data_stats(self) -> Dict:
        """데이터베이스 통계 정보 조회"""
        try:
            logger.info("\n=== ChromaDB 통계 정보 조회 시작 ===")
            
            # 컬렉션별 문서 수 확인
            collections = await self.list_collections()
            collection_stats = {
                coll['name']: {
                    'count': coll['count'],
                    'metadata': coll['metadata']
                } for coll in collections
            }
            logger.info(f"  - 컬렉션별 통계: {collection_stats}")
            
            # 메타데이터 분석
            metadata_stats = {}
            for name, coll in [
                ('initial', self.initial_collection),
                ('interaction', self.interaction_collection),
                ('adjustment', self.adjustment_collection)
            ]:
                try:
                    metadata = coll.get()['metadatas']
                    types = set(meta.get('type', 'unknown') for meta in metadata if meta)
                    categories = set(meta.get('category', 'unknown') for meta in metadata if meta)
                    
                    metadata_stats[name] = {
                        'types': list(types),
                        'categories': list(categories)
                    }
                    logger.info(f"  - {name} 메타데이터: {metadata_stats[name]}")
                    
                except Exception as e:
                    logger.warning(f"  - {name} 메타데이터 분석 실패: {str(e)}")
            
            return {
                'collections': collection_stats,
                'metadata': metadata_stats,
                'total_documents': sum(coll['count'] for coll in collections)
            }
            
        except Exception as e:
            logger.error("\n=== ChromaDB 통계 정보 조회 실패 ===")
            logger.error(f"오류: {str(e)}")
            logger.error("스택 트레이스:", exc_info=True)
            raise 