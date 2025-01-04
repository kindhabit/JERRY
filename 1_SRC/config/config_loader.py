from typing import Dict, List, Set, Optional
import yaml
import os
import logging
from pathlib import Path
from utils.logger_config import setup_logger
from utils.translation_manager import TranslationManager
from dotenv import load_dotenv

logger = setup_logger('config_loader')

class ConfigLoader:
    _instance = None
    _config = None
    _openai_settings = None
    _service_settings = None
    _pubmed_settings = None
    _health_mapping = None
    _api_keys = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize configuration and load settings"""
        # 1. 환경 변수 로드
        load_dotenv()
        self._api_keys = {
            'openai': os.getenv('OPENAI_API_KEY'),
            'pubmed': os.getenv('PUBMED_API_KEY', '')  # PubMed API 키가 없으면 빈 문자열 사용
        }
        
        # 2. 서비스 설정 로드 (config.yaml)
        self._config = self._load_config()
        
        # 3. 건강/영양제 데이터 로드 (health_mapping.yaml)
        self._health_mapping = self._load_health_mapping()
        
        # 4. OpenAI 설정
        self._openai_settings = {
            'api_key': self._api_keys['openai'],
            'chat': {
                'model': self._config.get('service', {}).get('openai', {}).get('chat', {}).get('model', 'gpt-4-turbo-preview'),
                'temperature': self._config.get('service', {}).get('openai', {}).get('chat', {}).get('temperature', 0.1)
            },
            'embedding': {
                'model': self._config.get('analysis', {}).get('openai', {}).get('models', {}).get('embedding', {}).get('default', 'text-embedding-3-small')
            }
        }
        
        # 5. 서비스 설정
        self._service_settings = self._config.get('service', {})
        
        # 6. PubMed 설정 (config.yaml의 data_sources.pubmed + pubmed_settings)
        pubmed_settings = self._config.get('pubmed_settings', {}).copy()
        if self._api_keys['pubmed']:  # API 키가 있는 경우에만 설정
            pubmed_settings['api_key'] = self._api_keys['pubmed']
            
        self._pubmed_settings = {
            **pubmed_settings,
            **self._config.get('data_sources', {}).get('pubmed', {}),
            'categories': self._health_mapping.get('pubmed', {}).get('categories', {}),
            'category_weights': self._health_mapping.get('pubmed', {}).get('category_weights', {})
        }

    def _load_config(self):
        """Load service configuration from config.yaml"""
        try:
            # config.yaml 로드
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                logger.info(f"[CONFIG] 설정 파일 경로: {config_path}")
            return config
            
        except FileNotFoundError as e:
            logger.error(f"[CONFIG] 설정 파일을 찾을 수 없습니다: {str(e)}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"[CONFIG] YAML 파싱 오류: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"[CONFIG] 설정 로드 중 오류 발생: {str(e)}")
            raise

    def _load_health_mapping(self):
        """Load health mapping from health_mapping.yaml"""
        try:
            mapping_path = os.path.join(os.path.dirname(__file__), 'health_mapping.yaml')
            with open(mapping_path, 'r', encoding='utf-8') as file:
                health_mapping = yaml.safe_load(file)
                logger.info(f"[CONFIG] 건강 매핑 파일 경로: {mapping_path}")
            return health_mapping
            
        except FileNotFoundError as e:
            logger.error(f"[CONFIG] 건강 매핑 파일을 찾을 수 없습니다: {str(e)}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"[CONFIG] YAML 파싱 오류: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"[CONFIG] 건강 매핑 로드 중 오류 발생: {str(e)}")
            raise

    def get_analysis_settings(self):
        """Get analysis settings"""
        return self._config.get('analysis', {})

    def get_service_settings(self):
        """서비스 설정을 반환합니다."""
        return self._service_settings

    def get_openai_settings(self):
        """OpenAI 설정을 반환합니다."""
        return self._openai_settings

    def get_pubmed_settings(self):
        """PubMed 설정을 반환합니다."""
        return self._pubmed_settings

    def get_health_keywords(self):
        """건강 키워드를 반환합니다."""
        keywords = {}
        for category_id, category in self._health_mapping.get('categories', {}).items():
            keywords[category_id] = {
                'name': category.get('name', ''),
                'display_name': category.get('display_name', ''),
                'description': category.get('description', ''),
                'search_terms': category.get('search_terms', []),
                'medical_terms': category.get('medical_terms', {})
            }
        return keywords

    def get_health_metrics(self):
        """건강 지표를 반환합니다."""
        metrics = {}
        for category_id, category in self._health_mapping.get('categories', {}).items():
            metrics[category_id] = category.get('related_metrics', [])
        return metrics

    def get_supplements(self):
        """영양제 정보를 반환합니다."""
        return self._health_mapping.get('supplements', {}).get('names', {})

    def get_pubmed_categories(self):
        """PubMed 검색 카테고리를 반환합니다."""
        return self._health_mapping.get('pubmed', {}).get('categories', {})

    def get_pubmed_category_weights(self):
        """PubMed 카테고리 가중치를 반환합니다."""
        return self._health_mapping.get('pubmed', {}).get('category_weights', {})

    def get_pubmed_search_strategies(self):
        """PubMed 검색 전략을 반환합니다."""
        return self._config.get('data_sources', {}).get('pubmed', {}).get('search_strategies', {})

    def get_reference_ranges(self):
        """참조 범위를 반환합니다."""
        ranges = {}
        for category_id, category in self._health_mapping.get('categories', {}).items():
            if 'reference_ranges' in category:
                ranges[category_id] = category['reference_ranges']
        return {'ranges': ranges}

CONFIG = ConfigLoader() 