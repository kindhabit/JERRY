import asyncio
import logging
from core.vector_db.embedding_creator import EmbeddingCreator
from utils.openai_client import OpenAIClient

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_single_text_embedding():
    """단일 텍스트 임베딩 테스트"""
    client = OpenAIClient()
    creator = EmbeddingCreator(client=client)
    
    text = "테스트 텍스트입니다."
    embeddings = await creator(text)
    
    logger.info(f"단일 텍스트 임베딩 결과:")
    logger.info(f"입력 텍스트: {text}")
    logger.info(f"임베딩 차원: {len(embeddings[0])}")
    logger.info(f"임베딩 생성 성공")

async def test_multiple_text_embedding():
    """여러 텍스트 임베딩 테스트"""
    client = OpenAIClient()
    creator = EmbeddingCreator(client=client)
    
    texts = ["첫 번째 텍스트", "두 번째 텍스트", "세 번째 텍스트"]
    embeddings = await creator(texts)
    
    logger.info(f"다중 텍스트 임베딩 결과:")
    logger.info(f"입력 텍스트 수: {len(texts)}")
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        logger.info(f"텍스트 {i+1}: {text}")
        logger.info(f"임베딩 차원: {len(embedding)}")
    logger.info(f"임베딩 생성 성공")

async def test_cache_functionality():
    """캐시 기능 테스트"""
    client = OpenAIClient()
    creator = EmbeddingCreator(client=client)
    
    text = "캐시 테스트 텍스트"
    
    # 첫 번째 호출
    await creator(text)
    initial_stats = creator.get_cache_stats()
    
    # 두 번째 호출 (캐시 사용)
    await creator(text)
    final_stats = creator.get_cache_stats()
    
    logger.info(f"캐시 테스트 결과:")
    logger.info(f"초기 캐시 상태: {initial_stats}")
    logger.info(f"최종 캐시 상태: {final_stats}")
    logger.info(f"캐시 히트 증가: {final_stats['cache_hits'] - initial_stats['cache_hits']}")

async def test_error_handling():
    """에러 처리 테스트"""
    client = OpenAIClient()
    creator = EmbeddingCreator(client=client)
    
    text = "x" * 10000  # 매우 긴 텍스트로 에러 유발
    embeddings = await creator(text)
    
    logger.info(f"에러 처리 테스트 결과:")
    logger.info(f"입력 텍스트 길이: {len(text)}")
    logger.info(f"임베딩 생성 결과: {'성공' if len(embeddings[0]) == 1536 else '실패'}")
    if all(x == 0.0 for x in embeddings[0]):
        logger.info("에러 처리: 0으로 채워진 벡터 반환 확인")

async def main():
    """모든 테스트 실행"""
    logger.info("=== 임베딩 생성 테스트 시작 ===")
    
    logger.info("\n>>> 단일 텍스트 임베딩 테스트")
    await test_single_text_embedding()
    
    logger.info("\n>>> 다중 텍스트 임베딩 테스트")
    await test_multiple_text_embedding()
    
    logger.info("\n>>> 캐시 기능 테스트")
    await test_cache_functionality()
    
    logger.info("\n>>> 에러 처리 테스트")
    await test_error_handling()
    
    logger.info("\n=== 임베딩 생성 테스트 완료 ===")

if __name__ == "__main__":
    asyncio.run(main()) 