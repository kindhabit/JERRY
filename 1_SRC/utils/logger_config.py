import logging
import os
from pprint import pformat
from typing import Any, Dict
from datetime import datetime

def setup_logging():
    """중앙 로깅 설정"""
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 공통 포맷터
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 파일 핸들러
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "server.log"))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

def setup_logger(name: str) -> logging.Logger:
    """기존 방식의 로거 설정 (하위 호환성 유지)"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 이미 핸들러가 있다면 추가하지 않음
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 파일 핸들러
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """서비스별 로거 가져오기"""
    logger = logging.getLogger(name)
    logger.propagate = True
    return logger

class PrettyLogger:
    def __init__(self, name: str):
        self.logger = get_logger(name)
        
    def _format_data(self, data: Any, max_length: int = 200) -> str:
        """데이터를 보기 좋게 포맷팅"""
        if isinstance(data, (dict, list)):
            formatted = pformat(data, indent=2, width=80)
            if len(formatted) > max_length:
                lines = formatted.split('\n')
                return '\n'.join(lines[:5]) + '\n... [truncated]'
            return formatted
        return str(data)

    def info(self, message: str, data: Any = None, step: str = None):
        """정보 레벨 로깅"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'message': message
        }
        if data is not None:
            log_entry['data'] = data
            
        self.logger.info('\n' + self._format_data(log_entry))

    def error(self, message: str, error: Exception = None, data: Any = None):
        """에러 레벨 로깅"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'error_type': type(error).__name__ if error else None,
            'error_message': str(error) if error else None
        }
        if data is not None:
            log_entry['data'] = data
            
        self.logger.error('\n' + self._format_data(log_entry))

    def debug(self, message: str, data: Any = None):
        """디버그 레벨 로깅"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message
        }
        if data is not None:
            log_entry['data'] = data
            
        self.logger.debug('\n' + self._format_data(log_entry))

    def warning(self, message: str, data: Any = None):
        """경고 레벨 로깅"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message
        }
        if data is not None:
            log_entry['data'] = data
            
        self.logger.warning('\n' + self._format_data(log_entry)) 