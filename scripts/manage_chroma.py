#!/usr/bin/env python3
import argparse
import asyncio
import logging
from typing import List, Dict
from modules.db.chroma_client import ChromaDBClient
from modules.utils.logger import setup_logger

logger = setup_logger('chroma.manager')

class ChromaManager:
    def __init__(self):
        self.client = ChromaDBClient()
        
    async def reinitialize_database(self, force: bool = False) -> bool:
        """데이터베이스 초기화"""
        try:
            logger.info("\n=== ChromaDB 초기화 시작 ===")
            
            if not force:
                logger.warning("force 옵션이 필요합니다")
                return False
                
            # 1. 기존 데이터 백업
            stats = await self.client.get_data_stats()
            logger.info(f"현재 데이터 상태: {stats}")
            
            # 2. 컬렉션 삭제 및 재생성
            collections = self.client.client.list_collections()
            for collection in collections:
                logger.info(f"컬렉션 삭제: {collection.name}")
                self.client.client.delete_collection(collection.name)
            
            # 3. 컬렉션 재생성
            logger.info("컬렉션 재생성 중...")
            await self.client.__init__()
            
            logger.info("데이터베이스 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {str(e)}")
            logger.error("상세 정보:", exc_info=True)
            return False
            
    async def update_database(self) -> bool:
        """데이터베이스 업데이트"""
        try:
            logger.info("\n=== ChromaDB 업데이트 시작 ===")
            
            # 1. 현재 상태 확인
            stats = await self.client.get_data_stats()
            logger.info(f"현재 데이터 상태: {stats}")
            
            # 2. 새 데이터 로드
            test_data = self._get_test_data()  # 실제 데이터 로드 함수로 교체 필요
            
            # 3. 임베딩 생성
            embeddings, ids, metadata = await self.client.create_enhanced_embeddings(
                texts=[d['text'] for d in test_data],
                metadata=[{
                    'type': d['type'],
                    'category': d['category']
                } for d in test_data]
            )
            
            # 4. 데이터 저장
            success = await self.client.add_supplement_data(
                embeddings=embeddings,
                texts=[d['text'] for d in test_data],
                metadata=metadata
            )
            
            if success:
                logger.info("데이터베이스 업데이트 완료")
                return True
            else:
                logger.error("데이터 저장 실패")
                return False
                
        except Exception as e:
            logger.error(f"데이터베이스 업데이트 실패: {str(e)}")
            logger.error("상세 정보:", exc_info=True)
            return False
            
    def _get_test_data(self) -> List[Dict]:
        """테스트 데이터 생성"""
        return [
            {
                'text': 'Omega-3 fatty acids can help reduce inflammation and support heart health.',
                'type': 'omega',
                'category': 'heart_health'
            },
            {
                'text': 'Vitamin D is essential for calcium absorption and bone health.',
                'type': 'vitamin',
                'category': 'bone_health'
            }
        ]

async def main():
    parser = argparse.ArgumentParser(description='ChromaDB 관리 도구')
    parser.add_argument('--action', choices=['reinit', 'update'], required=True,
                      help='수행할 작업 (reinit: 초기화, update: 업데이트)')
    parser.add_argument('--force', action='store_true',
                      help='강제 실행 (초기화 시 필요)')
    parser.add_argument('--debug', action='store_true',
                      help='디버그 모드 활성화')
                      
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    manager = ChromaManager()
    
    if args.action == 'reinit':
        success = await manager.reinitialize_database(args.force)
    else:
        success = await manager.update_database()
        
    if not success:
        exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 