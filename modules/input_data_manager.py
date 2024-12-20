import os
import json
import logging
from modules.pubmed import PubMedClient
from chromadb import HttpClient

logger = logging.getLogger(__name__)

# 번역 캐시 파일 경로
TRANSLATION_CACHE_FILE = os.path.join(os.path.dirname(__file__), "../data/translation_cache.json")

def load_translation_cache():
    """
    번역 캐시를 로드합니다.
    """
    if os.path.exists(TRANSLATION_CACHE_FILE):
        with open(TRANSLATION_CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_translation_cache(cache):
    """
    번역 캐시를 저장합니다.
    """
    logger.info(f"Saving cache to: {TRANSLATION_CACHE_FILE}")
    directory = os.path.dirname(TRANSLATION_CACHE_FILE)
    if not os.path.exists(directory):
        logger.info(f"Creating directory for cache: {directory}")
        os.makedirs(directory, exist_ok=True)

    try:
        with open(TRANSLATION_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        logger.info("Cache saved successfully.")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")
        raise


def translate_to_english_with_cache(korean_terms, translate_function):
    """
    한국어 키워드를 영어로 번역하고 캐시에 저장합니다.
    """
    cache = load_translation_cache()
    translated_terms = []

    for term in korean_terms:
        if term in cache:
            logger.info(f"캐시에서 번역 가져옴: {term} -> {cache[term]}")
            translated_terms.append(cache[term])
        else:
            english_term = translate_function(term)  # 번역 API 호출
            cache[term] = english_term
            translated_terms.append(english_term)

    save_translation_cache(cache)
    return translated_terms

def add_new_condition_to_db(condition_name, pubmed_client, vectorstore):
    """
    새로운 질병 또는 바이러스 데이터를 ChromaDB에 추가합니다.
    """
    query = f"{condition_name} treatment or adverse effects"
    logger.info(f"새로운 조건 추가: {query}")

    pmids = pubmed_client.search_pmids(query, retmax=10)
    abstracts = pubmed_client.fetch_abstracts(pmids)

    for pmid, abstract in abstracts:
        try:
            metadata = {"condition": condition_name, "pmid": pmid, "query": query}
            vectorstore.add_texts([abstract], metadatas=[metadata], ids=[pmid])
            logger.info(f"{condition_name} 저장됨: PMID {pmid}")
        except Exception as e:
            logger.error(f"저장 실패 (PMID {pmid}): {e}")

# 예제 사용
if __name__ == "__main__":
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    import yaml

    # 설정 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "../config/config.yaml")

    with open(config_path, 'r') as config_file:
        config_dict = yaml.safe_load(config_file)

    # ChromaDB 및 PubMed 초기화
    chroma_client = HttpClient(
        host=config_dict['chroma']['server_host'],
        port=config_dict['chroma']['server_port']
    )
    embeddings = OpenAIEmbeddings(openai_api_key=config_dict['openai']['api_key'])
    vectorstore = Chroma(client=chroma_client,
                         collection_name=config_dict['chroma']['collection_name'],
                         embedding_function=embeddings)

    pubmed_client = PubMedClient(config_dict)

    # 번역 예제
    def mock_translate_function(term):
        return f"{term}_translated"

    korean_terms = ["위염", "간질환"]
    translated = translate_to_english_with_cache(korean_terms, mock_translate_function)
    logger.info(f"번역 결과: {translated}")

    # 새로운 질병 추가 예제
    add_new_condition_to_db("Gastritis", pubmed_client, vectorstore)
