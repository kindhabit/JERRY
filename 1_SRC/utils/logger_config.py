import logging
import os

def setup_logger(name):
    logger = logging.getLogger(name)
    
    # 기본 로그 레벨을 INFO로 설정
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # 콘솔에는 INFO 레벨 이상만 출력
        
        # 간단한 포맷으로 변경
        formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 파일 핸들러 설정
        log_dir = os.path.abspath("./logs")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        file_handler.setLevel(logging.DEBUG)  # 파일에는 모든 로그 저장
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 