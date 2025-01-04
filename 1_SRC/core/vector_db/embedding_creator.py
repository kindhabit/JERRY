import numpy as np
from typing import List, Dict, Any
from utils.openai_client import OpenAIClient
from utils.logger_config import setup_logger

logger = setup_logger('embedding')

class EmbeddingCreator:
    """임베딩 생성기"""
    
    def __init__(self):
        """임베딩 생성기 초기화"""
        self.client = OpenAIClient()
        self._cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def __call__(self, input: str | List[str]) -> List[List[float]]:
        """임베딩 생성
        
        Args:
            input: 임베딩할 텍스트 또는 텍스트 리스트
            
        Returns:
            임베딩 벡터 리스트
        """
        try:
            # 단일 텍스트인 경우 리스트로 변환
            if isinstance(input, str):
                texts = [input]
            else:
                texts = input
            
            # 각 텍스트에 대해 임베딩 생성
            embeddings = []
            for text in texts:
                # 캐시된 임베딩이 있으면 재사용
                if text in self._cache:
                    self.cache_hits += 1
                    embeddings.append(self._cache[text])
                    continue
                    
                # 새로운 임베딩 생성
                self.cache_misses += 1
                embedding = self.client.create_embedding(text)
                self._cache[text] = embedding
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {str(e)}")
            # 에러 발생 시 0으로 채워진 임베딩 반환
            return [[0.0] * 1536] * (1 if isinstance(input, str) else len(input))
        
    def get_cache_stats(self) -> dict:
        """캐시 통계 반환"""
        return {
            "cache_size": len(self._cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses
        }
