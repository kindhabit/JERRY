import os
import yaml
import logging
from typing import Dict, List
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)

def load_config() -> Dict:
    """설정 파일 로드"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
            # .env에서 환경 변수들 가져오기
            config['openai']['api_key'] = os.getenv('OPENAI_API_KEY')
            config['chroma']['persist_directory'] = os.path.join(
                os.getenv('PYTHONPATH', '/workspace/chroma_project'),
                'data/chroma'
            )
            
            logger.debug(f"설정 로드 완료: {config}")
            return config
    except Exception as e:
        logger.error(f"설정 로드 실패: {str(e)}")
        raise

def get_health_keywords() -> List[str]:
    """건강 관련 키워드 목록 반환"""
    try:
        return CONFIG["pubmed"]["health_keywords"]
    except KeyError:
        logger.error("health_keywords 설정을 찾을 수 없습니다")
        return []

def get_supplements() -> List[str]:
    """영양제 성분 목록 반환"""
    try:
        return CONFIG["pubmed"]["supplements"]
    except KeyError:
        logger.error("supplements 설정을 찾을 수 없습니다")
        return []

# 전역 설정 객체
CONFIG = load_config() 