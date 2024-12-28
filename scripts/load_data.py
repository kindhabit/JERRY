import asyncio
import logging
from modules.chroma_database import manage_chroma_database
from modules.structured_data import load_structured_data

logger = logging.getLogger(__name__)

async def main():
    """데이터 적재 메인 함수"""
    logger.info("데이터 적재 시작")
    
    try:
        # ChromaDB 초기화
        await manage_chroma_database(action="reinit", force=True)
        
        # 구조화된 데이터 적재
        await load_structured_data()
        
        logger.info("데이터 적재 완료")
        
    except Exception as e:
        logger.error(f"데이터 적재 실패: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 