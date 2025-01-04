from openai import AsyncOpenAI
from typing import List, Dict, Any
from utils.logger_config import setup_logger
from config.config_loader import CONFIG

logger = setup_logger('openai_client')

class OpenAIClient:
    """OpenAI API 클라이언트"""
    
    def __init__(self):
        """OpenAI 클라이언트 초기화"""
        self.client = AsyncOpenAI(api_key=CONFIG._api_keys['openai'])
        self.settings = CONFIG.get_openai_settings()
        logger.info("OpenAI 클라이언트 초기화 완료")
        
    async def create_embedding(self, text: str) -> List[float]:
        """텍스트의 임베딩 벡터 생성
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터
        """
        try:
            response = await self.client.embeddings.create(
                model=self.settings['embedding']['model'],
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {str(e)}")
            logger.error(f"입력 텍스트 길이: {len(text)}")
            return [0.0] * 1536
            
    async def analyze_with_context(self, prompt: str, context: str = None) -> str:
        """컨텍스트를 포함한 프롬프트 분석
        
        Args:
            prompt: 분석할 프롬프트
            context: 추가 컨텍스트 (선택사항)
            
        Returns:
            분석 결과 텍스트
        """
        try:
            messages = []
            if context:
                messages.append({"role": "system", "content": context})
            messages.append({"role": "user", "content": prompt})
            
            response = await self.client.chat.completions.create(
                model=self.settings['chat']['model'],
                messages=messages,
                temperature=self.settings['chat']['temperature'],
                max_tokens=1000
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"분석 실패: {str(e)}")
            return ""
            
    async def get_embeddings(self, text: str) -> List[float]:
        """텍스트의 임베딩 벡터를 생성"""
        return await self.create_embedding(text) 