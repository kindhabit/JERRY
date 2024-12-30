import os
import yaml
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
from pathlib import Path

logger = logging.getLogger('config.loader')

class ConfigManager:
    def __init__(self):
        # .env 파일 로드
        env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(env_path)
        
        self.config = self.load_config()
        
    def load_config(self) -> Dict:
        """설정 파일 로드"""
        try:
            config_path = Path(__file__).parent / 'config.yaml'
            logger.info(f"[CONFIG] 설정 파일 경로: {config_path}")
                
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config
                
        except Exception as e:
            logger.error(f"[CONFIG] 설정 로드 실패: {str(e)}")
            raise

    def get_health_keywords(self) -> List[Dict]:
        """건강 관련 키워드 목록 반환
        Returns:
            List[Dict]: [
                {
                    'category': str,
                    'display_name': str,
                    'search_terms': List[str],
                    'conditions': List[Dict]
                }
            ]
        """
        try:
            return self.config["data_sources"]["pubmed"]["health_keywords"]
        except KeyError:
            logger.error("health_keywords 설정을 찾을 수 없습니다")
            return []

    def get_supplements(self) -> List[Dict]:
        """영양제 성분 목록 반환
        Returns:
            List[Dict]: [
                {
                    'name': str,
                    'aliases': List[str]
                }
            ]
        """
        try:
            return self.config["data_sources"]["pubmed"]["supplements"]
        except KeyError:
            logger.error("supplements 설정을 찾을 수 없습니다")
            return []

# 전역 설정 객체
CONFIG = ConfigManager() 