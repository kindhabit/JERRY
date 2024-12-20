#!/usr/bin/env python3
# scripts/view_chroma_data.py

import os
import yaml
import logging
from chromadb import HttpClient
from langchain_chroma import Chroma

# 스크립트의 현재 디렉토리를 기반으로 config 파일의 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '../config/config.yaml')

# config.yaml 파일 로드
with open(config_path, 'r') as config_file:
    config_dict = yaml.safe_load(config_file)

# 로그 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def main():
    # Chroma HttpClient 초기화
    chroma_client = HttpClient(
        host=config_dict['chroma']['server_host'],
        port=config_dict['chroma']['server_port']
    )

    # Chroma 벡터스토어에 연결
    collection_name = config_dict['chroma']['collection_name']
    logger.info(f"'{collection_name}' 컬렉션 데이터 확인 중...")

    # Chroma 컬렉션 불러오기
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name
    )

    # 데이터 조회
    results = vectorstore.get()
    if results and 'documents' in results:
        for idx, doc in enumerate(results['documents']):
            logger.info(f"Document {idx + 1}: {doc}")
            logger.info(f"Metadata: {results['metadatas'][idx]}")
    else:
        logger.info("컬렉션에 데이터가 없습니다.")

if __name__ == "__main__":
    main()
