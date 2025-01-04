from utils.logger_config import setup_logger
import logging
import os
from logging.handlers import RotatingFileHandler
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import base64
import json

logger = setup_logger('system_utils')

class SystemUtils:
    """통합된 시스템 유틸리티 클래스"""
    
    @staticmethod
    def generate_key(api_token: str) -> bytes:
        """API 토큰을 32바이트 AES 키로 변환"""
        if len(api_token) < 32:
            return api_token.zfill(32).encode()
        return api_token[:32].encode()

    @staticmethod
    def decrypt_data(encrypted_data: str, api_token: str) -> str:
        """데이터 복호화"""
        try:
            aes_key = SystemUtils.generate_key(api_token)
            encrypted_bytes = base64.b64decode(encrypted_data)
            
            aes_iv = encrypted_bytes[:16]
            encrypted_message = encrypted_bytes[16:]
            
            cipher = Cipher(
                algorithms.AES(aes_key), 
                modes.CBC(aes_iv), 
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            decrypted_padded = decryptor.update(encrypted_message) + decryptor.finalize()
            
            unpadder = padding.PKCS7(128).unpadder()
            decrypted_data = unpadder.update(decrypted_padded) + unpadder.finalize()
            
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"복호화 실패: {str(e)}")
            raise ValueError(f"복호화 실패: {str(e)}")

    @staticmethod
    def decrypt_request_data(encrypted_data: dict) -> dict:
        """요청 데이터 복호화"""
        try:
            from config.config_loader import CONFIG
            api_token = CONFIG["security"]["api_token"]
            decrypted_str = SystemUtils.decrypt_data(
                encrypted_data["data"], 
                api_token
            )
            return json.loads(decrypted_str)
            
        except Exception as e:
            logger.error(f"Request decryption failed: {str(e)}")
            raise 