from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import base64
import os
import logging
import json
from config.config_loader import CONFIG

logger = logging.getLogger(__name__)

class CommonUtils:
    """공통 유틸리티 클래스"""
    
    @staticmethod
    def generate_key(api_token: str) -> bytes:
        """API 토큰을 32바이트 AES 키로 변환"""
        if len(api_token) < 32:
            return api_token.zfill(32).encode()  # 길이가 부족하면 0으로 채움
        return api_token[:32].encode()  # 32바이트로 자름

    @staticmethod
    def decrypt_data(encrypted_data: str, api_token: str) -> str:
        """데이터 복호화"""
        try:
            # 키 생성
            aes_key = CommonUtils.generate_key(api_token)

            # Base64 디코딩
            logger.debug("Base64 디코딩 시작")
            encrypted_bytes = base64.b64decode(encrypted_data)
            logger.debug(f"Base64 디코딩 완료: {len(encrypted_bytes)} bytes")

            # IV와 암호문 분리 (IV는 첫 16바이트)
            aes_iv = encrypted_bytes[:16]
            encrypted_message = encrypted_bytes[16:]
            logger.debug(f"IV 길이: {len(aes_iv)}, 암호문 길이: {len(encrypted_message)}")

            # 복호화
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(aes_iv), backend=default_backend())
            decryptor = cipher.decryptor()
            decrypted_padded = decryptor.update(encrypted_message) + decryptor.finalize()

            # PKCS7 패딩 제거
            unpadder = padding.PKCS7(128).unpadder()
            decrypted_data = unpadder.update(decrypted_padded) + unpadder.finalize()

            # UTF-8 디코딩
            result = decrypted_data.decode('utf-8')
            logger.debug("복호화 성공")
            return result

        except Exception as e:
            logger.error(f"복호화 실패: {str(e)}")
            raise ValueError(f"복호화 실패: {str(e)}")

    @staticmethod
    def decrypt_request_data(encrypted_data: dict) -> dict:
        """요청 데이터 복호화"""
        try:
            # API 토큰 가져오기
            api_token = CONFIG["security"]["api_token"]
            
            # 암호화된 데이터 복호화
            decrypted_str = CommonUtils.decrypt_data(encrypted_data["data"], api_token)
            
            # JSON 파싱
            return json.loads(decrypted_str)
            
        except Exception as e:
            logger.error(f"Request decryption failed: {str(e)}")
            raise 