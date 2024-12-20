#!/usr/bin/env python3
# scripts/add_pubmed_to_chroma.py

import os
import yaml
import logging
from modules.pubmed import PubMedClient
from chromadb import HttpClient
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from modules.input_data_manager import translate_to_english_with_cache

# 설정 파일 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '../config/config.yaml')

with open(config_path, 'r') as config_file:
    config_dict = yaml.safe_load(config_file)

# 로그 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join(current_dir, '../logs/app.log'))
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def is_in_whitelist(title, whitelist):
    """
    논문 제목이 화이트리스트에 포함된 키워드를 갖고 있는지 확인합니다.
    """
    return any(keyword.lower() in title.lower() for keyword in whitelist)


# 번역 함수 (단순 테스트용)
def mock_translate_function(term):
    """
    임의 번역 함수 (테스트용)
    """
    return f"{term}_translated"


def main():
    # 설정 값 로드
    SUPPLEMENTS = config_dict['pubmed']['supplements']
    HEALTH_KEYWORDS = config_dict['pubmed']['health_keywords']

    logger.info(f"SUPPLEMENTS: {SUPPLEMENTS}")
    logger.info(f"HEALTH_KEYWORDS: {HEALTH_KEYWORDS}")

    # 번역 함수 정의 및 전달
    translate_function = mock_translate_function
    translated_supplements = translate_to_english_with_cache(SUPPLEMENTS, translate_function)
    translated_keywords = translate_to_english_with_cache(HEALTH_KEYWORDS, translate_function)
    logger.info(f"번역된 SUPPLEMENTS: {translated_supplements}")
    logger.info(f"번역된 HEALTH_KEYWORDS: {translated_keywords}")

    # ChromaDB 및 OpenAI Embeddings 초기화
    chroma_client = HttpClient(
        host=config_dict['chroma']['server_host'],
        port=config_dict['chroma']['server_port']
    )
    embeddings = OpenAIEmbeddings(openai_api_key=config_dict['openai']['api_key'])
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=config_dict['chroma']['collection_name'],
        embedding_function=embeddings
    )
    logger.info("Chroma 벡터스토어 초기화 완료.")

    # PubMed 클라이언트 초기화
    pubmed_client = PubMedClient(config_dict)

    # 번역된 supplements와 keywords로 PubMed 검색
    for supplement in translated_supplements:
        for keyword in translated_keywords:
            query = f"{supplement} {keyword}"
            logger.info(f"PubMed 검색 시작: {query}")

            pmids = pubmed_client.search_pmids(query, retmax=5)
            if not pmids:
                logger.info(f"{query}에 대한 PMID를 찾을 수 없습니다.")
                continue

            abstracts = pubmed_client.fetch_abstracts(pmids)
            if not abstracts:
                logger.info(f"{query}에 대한 초록을 가져올 수 없습니다.")
                continue

            for item in abstracts:
                if len(item) == 2:
                    pmid, abstract = item
                    title = "제목 없음"
                elif len(item) == 3:
                    pmid, abstract, title = item
                else:
                    logger.warning(f"알 수 없는 형식의 데이터 건너뜀: {item}")
                    continue

                # PubMed 링크 생성
                pubmed_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid != "N/A" else None

                # 화이트리스트 필터링
                if not is_in_whitelist(title, SUPPLEMENTS):
                    logger.info(f"화이트리스트에 없는 논문 건너뜀: {title}")
                    continue

                try:
                    # ChromaDB에 텍스트 추가
                    vectorstore.add_texts(
                        [abstract],
                        metadatas=[
                            {
                                "pmid": pmid,
                                "query": query,
                                "title": title,
                                "supplement": supplement,
                                "health_keyword": keyword,
                                "link": pubmed_link
                            }
                        ],
                        ids=[pmid]
                    )
                    logger.info(f"PMID: {pmid} 논문 저장 완료: {pubmed_link}")
                except Exception as e:
                    logger.error(f"Chroma에 업서트 중 오류 발생 (PMID: {pmid}): {e}")

    logger.info("모든 PubMed 데이터 업서트 완료.")


if __name__ == "__main__":
    main()
