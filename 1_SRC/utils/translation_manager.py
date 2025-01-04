import yaml
from pathlib import Path
from typing import Dict, Optional, List
from utils.logger_config import setup_logger

logger = setup_logger('translation')

class TranslationManager:
    """한글-영어 변환 관리자"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TranslationManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.mapping = self._load_mapping()
            self._initialized = True
        
    def _load_mapping(self) -> Dict:
        """매핑 파일 로드"""
        try:
            mapping_path = Path(__file__).parent.parent / 'config' / 'health_mapping.yaml'
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping = yaml.safe_load(f)
            logger.info("번역 매핑 파일 로드 완료")
            return mapping
        except Exception as e:
            logger.error(f"번역 매핑 파일 로드 실패: {str(e)}")
            raise
    
    def get_english(self, korean: str, category: str, subcategory: str) -> Optional[str]:
        """한글을 영어로 변환
        
        Args:
            korean (str): 한글 텍스트
            category (str): 카테고리 (예: supplements, categories 등)
            subcategory (str): 하위 카테고리 (예: names, effects 등)
            
        Returns:
            Optional[str]: 영어 텍스트 또는 None
        """
        try:
            if category == 'medical_terms':
                for cat in self.mapping['categories'].values():
                    if korean in cat.get('medical_terms', {}):
                        return cat['medical_terms'][korean]
                return None
            return self.mapping[category][subcategory].get(korean)
        except KeyError:
            logger.warning(f"매핑을 찾을 수 없음: {category}.{subcategory}.{korean}")
            return None
    
    def get_korean(self, english: str, category: str, subcategory: str) -> Optional[str]:
        """영어를 한글로 변환
        
        Args:
            english (str): 영어 텍스트
            category (str): 카테고리
            subcategory (str): 하위 카테고리
            
        Returns:
            Optional[str]: 한글 텍스트 또는 None
        """
        try:
            if category == 'medical_terms':
                for cat in self.mapping['categories'].values():
                    medical_terms = cat.get('medical_terms', {})
                    for k, v in medical_terms.items():
                        if v == english:
                            return k
                return None
            
            mapping = self.mapping[category][subcategory]
            for k, v in mapping.items():
                if v == english:
                    return k
            return None
        except KeyError:
            logger.warning(f"매핑을 찾을 수 없음: {category}.{subcategory}.{english}")
            return None
    
    def translate_supplement_info(self, info: Dict) -> Dict:
        """영양제 정보 전체 번역
        
        Args:
            info (Dict): 영양제 정보 딕셔너리
            
        Returns:
            Dict: 번역된 영양제 정보
        """
        translated = {}
        
        # 이름 번역
        if 'name' in info:
            translated['name'] = self.get_english(info['name'], 'supplements', 'names') or info['name']
        
        # 카테고리 번역
        if 'category' in info:
            translated['category'] = self.get_english(info['category'], 'supplements', 'categories') or info['category']
        
        # 효과 번역
        if 'effects' in info:
            translated['effects'] = [
                self.get_english(effect, 'supplements', 'effects') or effect
                for effect in info['effects']
            ]
        
        # 나머지 필드는 그대로 복사
        for k, v in info.items():
            if k not in translated:
                translated[k] = v
        
        return translated
    
    def translate_health_metric(self, metric: Dict) -> Dict:
        """건강 지표 정보 전체 번역
        
        Args:
            metric (Dict): 건강 지표 정보 딕셔너리
            
        Returns:
            Dict: 번역된 건강 지표 정보
        """
        translated = {}
        
        # 이름 번역
        if 'name' in metric:
            translated['name'] = self.get_english(metric['name'], 'health_metrics', 'names') or metric['name']
        
        # 관련 값 번역
        if 'related_values' in metric:
            translated['related_values'] = [
                self.get_english(value, 'health_metrics', 'values') or value
                for value in metric['related_values']
            ]
        
        # 상호작용 경고 번역
        if 'interaction_warnings' in metric:
            translated['interaction_warnings'] = [
                self.get_english(warning, 'interactions', 'warnings') or warning
                for warning in metric['interaction_warnings']
            ]
        
        # 나머지 필드는 그대로 복사
        for k, v in metric.items():
            if k not in translated:
                translated[k] = v
        
        return translated 
    
    def get_all_terms(self) -> List[Dict]:
        """모든 의학 용어 반환
        
        Returns:
            List[Dict]: 의학 용어 목록 [{ko: str, en: str, category: str}]
        """
        try:
            terms = []
            for category_name, category in self.mapping['categories'].items():
                if 'medical_terms' in category:
                    for ko, en in category['medical_terms'].items():
                        terms.append({
                            'ko': ko,
                            'en': en,
                            'category': category_name
                        })
            return terms
        except Exception as e:
            logger.error(f"의학 용어 목록 조회 실패: {str(e)}")
            return [] 
    
    def get_english_term(self, korean: str) -> Optional[str]:
        """한글 용어를 영어로 변환 (모든 카테고리 검색)
        
        Args:
            korean (str): 한글 텍스트
            
        Returns:
            Optional[str]: 영어 텍스트 또는 None
        """
        try:
            # 1. 영양제 이름 검색
            if 'supplements' in self.mapping and 'names' in self.mapping['supplements']:
                result = self.mapping['supplements']['names'].get(korean)
                if result:
                    return result
            
            # 2. 의학 용어 검색
            if 'medical_terms' in self.mapping:
                for category in self.mapping['medical_terms']:
                    result = self.mapping['medical_terms'][category].get(korean)
                    if result:
                        return result
            
            # 3. 건강 지표 검색
            if 'health_metrics' in self.mapping and 'names' in self.mapping['health_metrics']:
                result = self.mapping['health_metrics']['names'].get(korean)
                if result:
                    return result
            
            logger.warning(f"'{korean}'에 대한 영어 번역을 찾을 수 없습니다.")
            return None
            
        except Exception as e:
            logger.error(f"영어 용어 검색 실패: {str(e)}")
            return None 