import asyncio
import logging
from datetime import datetime
from typing import Dict, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import CONFIG
from modules.db.chroma_client import ChromaDBClient
from langchain_openai import OpenAIEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSimulation:
    def __init__(self):
        self.chroma_client = ChromaDBClient()
        self.test_results = []
        
    async def run_all_tests(self):
        """모든 테스트 실행"""
        try:
            # 1. 설정 검증
            await self.run_config_test()
            
            # 2. 임베딩 테스트
            test_data = self._get_test_data()
            embedding_result = await self.run_embedding_test(test_data)
            
            # 3. 데이터 저장 테스트
            await self.run_storage_test()
            
            # 4. 상호작용 검색 테스트
            await self.run_interaction_search_test()
            
            # 5. 통계 및 관리 테스트
            await self.run_management_test()
            
        except Exception as e:
            logger.error(f"테스트 실행 실패: {str(e)}")
            raise
        finally:
            self.print_test_summary()

    def _get_test_data(self) -> Dict:
        """테스트용 데이터 생성"""
        return {
            'supplement_name': 'Omega-3',
            'abstract': '''
            Clinical studies have shown that omega-3 fatty acids can significantly 
            reduce blood pressure in hypertensive patients. The effect appears to 
            be dose-dependent, with higher doses showing stronger effects.
            ''',
            'health_metrics': {'blood_pressure': 'high'},
            'category': 'cardiovascular',
            'pmid': 'TEST001',
            'drug_interactions': ['blood_thinners', 'statins'],
            'condition_concerns': ['bleeding_risk', 'cardiovascular_disease'],
            'dosage': '1000mg',
            'timing': 'daily',
            'evidence_level': 'high',
            'safety_category': 'generally_safe'
        }

    async def run_config_test(self):
        """설정 검증 테스트"""
        try:
            logger.info("\n=== ChromaDB 설정 검증 시작 ===")
            
            # ChromaDB 서버 설정 출력
            logger.info("[ChromaDB 서버 설정]")
            logger.info(f"  - 호스트: {CONFIG.config['service']['chroma']['host']}")
            logger.info(f"  - 포트: {CONFIG.config['service']['chroma']['port']}")
            logger.info(f"  - API 구현: {CONFIG.config['service']['chroma']['chroma_api_impl']}")
            
            # ChromaDB 연결 및 컬렉션 확인
            logger.info("\n[ChromaDB 연결 테스트]")
            collections = await self.chroma_client.list_collections()
            logger.info(f"  - 연결 상태: 성공")
            logger.info(f"  - 컬렉션 목록: {[c.name for c in collections]}")
            
            # 설정 검증
            logger.info("\n[필수 설정 검증]")
            required_configs = [
                ('service', '서비스 설정'),
                ('chroma', 'ChromaDB 설정', 'service'),
                ('data_sources', '데이터 소스 설정'),
                ('pubmed', 'PubMed 설정', 'data_sources')
            ]
            
            for config in required_configs:
                if len(config) == 2:
                    key, desc = config
                    assert key in CONFIG.config, f"{desc} 누락"
                    logger.info(f"  - {desc}: ✅")
                else:
                    key, desc, parent = config
                    assert key in CONFIG.config[parent], f"{desc} 누락"
                    logger.info(f"  - {desc}: ✅")
            
            self._add_test_result('config_validation', 'success')
            logger.info("\n=== ChromaDB 설정 검증 완료 ===")
            
        except Exception as e:
            logger.error(f"\n[오류] 설정 검증 실패: {str(e)}")
            self._add_test_result('config_validation', 'failed', str(e))
            raise

    async def run_embedding_test(self, test_data: Dict):
        """임베딩 생성 테스트"""
        try:
            logger.info("임베딩 테스트 시작...")
            texts = [
                "비타민 D는 뼈 건강에 중요합니다.",
                "오메가3는 심장 건강에 좋습니다.",
            ]
            
            # 임베딩만 받도록 수정
            embeddings = await self.chroma_client.create_enhanced_embeddings(texts)
            
            assert embeddings is not None
            assert len(embeddings) > 0
            
            self._add_test_result('embedding_generation', 'success')
            logger.info("임베딩 테스트 완료")
            
            return embeddings
            
        except Exception as e:
            self._add_test_result('embedding_generation', 'failed', str(e))
            raise

    async def run_storage_test(self):
        """저장 테스트"""
        try:
            logger.info("저장 테스트 시작...")
            
            # 테스트 데이터 준비
            texts = [
                "비타민 D는 뼈 건강에 중요합니다.",
                "오메가3는 심장 건강에 좋습니다."
            ]
            
            # 임베딩 생성 (튜플로 반환됨)
            embeddings, ids, _ = await self.chroma_client.create_enhanced_embeddings(texts)
            logger.info(f"  - 생성된 임베딩: 타입={type(embeddings)}, 크기={len(embeddings)}")
            
            # 메타데이터 준비
            metadata = [
                {"type": "vitamin", "category": "bone_health"},
                {"type": "omega", "category": "heart_health"}
            ]
            
            # ChromaDB에 저장
            success = await self.chroma_client.add_supplement_data(
                embeddings=embeddings,  # 튜플의 첫 번째 요소만 전달
                texts=texts,
                metadata=metadata
            )
            
            assert success, "데이터 저장 실패"
            
            self._add_test_result('storage_and_retrieval', 'success')
            logger.info("저장 테스트 완료")
            
        except Exception as e:
            self._add_test_result('storage_and_retrieval', 'failed', str(e))
            raise

    async def run_interaction_search_test(self):
        """상호작용 검색 테스트"""
        try:
            logger.info("상호작용 검색 테스트 시작...")
            
            # 상호작용 검색
            results = await self.chroma_client.search_interactions(
                supplements=['Omega-3', 'Vitamin D'],
                health_conditions={'cardiovascular': 'condition'}
            )
            
            assert results is not None
            
            self._add_test_result('interaction_search', 'success')
            logger.info("상호작용 검색 테스트 완료")
            
        except Exception as e:
            self._add_test_result('interaction_search', 'failed', str(e))
            raise

    async def run_management_test(self):
        """관리 기능 테스트"""
        try:
            logger.info("관리 기능 테스트 시작...")
            
            # 통계 정보 조회
            stats = await self.chroma_client.get_data_stats()
            
            # 검증
            assert stats is not None
            assert 'collections' in stats
            assert 'total_documents' in stats
            assert 'metadata' in stats
            assert stats['total_documents'] > 0
            
            # 컬렉션별 검증
            for collection_name in ['supplements_initial', 'supplements_interaction', 'supplements_adjustment']:
                assert collection_name in stats['collections']
                assert 'count' in stats['collections'][collection_name]
                
            # 메타데이터 검증
            for collection_type in ['initial', 'interaction', 'adjustment']:
                assert collection_type in stats['metadata']
                assert 'types' in stats['metadata'][collection_type]
                assert 'categories' in stats['metadata'][collection_type]
            
            self._add_test_result('management_functions', 'success')
            logger.info("관리 기능 테스트 완료")
            
        except Exception as e:
            self._add_test_result('management_functions', 'failed', str(e))
            raise

    def _add_test_result(self, test_name: str, status: str, error: str = None):
        """테스트 결과 추가"""
        result = {
            'test': test_name,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        if error:
            result['error'] = error
        self.test_results.append(result)

    def print_test_summary(self):
        """테스트 결과 요약 출력"""
        print("\n=== 테스트 결과 요약 ===")
        for result in self.test_results:
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"{status_icon} {result['test']}: {result['status']}")
            if 'error' in result:
                print(f"   Error: {result['error']}")
            print(f"   Time: {result['timestamp']}")
        print("=====================\n")

async def main():
    simulator = TestSimulation()
    await simulator.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 