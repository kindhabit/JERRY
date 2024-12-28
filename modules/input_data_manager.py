import os
import json
import logging

logger = logging.getLogger(__name__)

TRANSLATION_CACHE_FILE = os.path.join(os.path.dirname(__file__), "../data/translation_cache.json")


def load_translation_cache():
    """번역 캐시를 로드합니다."""
    if os.path.exists(TRANSLATION_CACHE_FILE):
        with open(TRANSLATION_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_translation_cache(cache):
    """번역 캐시를 저장합니다."""
    with open(TRANSLATION_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    logger.info("번역 캐시 저장 완료.")


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
            english_term = translate_function(term)
            cache[term] = english_term
            translated_terms.append(english_term)

    save_translation_cache(cache)
    return translated_terms


def add_new_condition_to_db(condition_name, pubmed_client, vectorstore):
    """
    새로운 질병 데이터를 ChromaDB에 추가합니다.
    """
    query = f"{condition_name} treatment or adverse effects"
    logger.info(f"새로운 조건 추가: {query}")

    pmids = pubmed_client.search_pmids(query, retmax=10)
    abstracts = pubmed_client.fetch_abstracts(pmids)

    for pmid, abstract in abstracts:
        try:
            vectorstore.add_texts(
                [abstract],
                metadatas=[{"condition": condition_name, "pmid": pmid, "query": query}],
                ids=[pmid]
            )
            logger.info(f"{condition_name} 데이터 저장 완료: PMID {pmid}")
        except Exception as e:
            logger.error(f"ChromaDB 저장 실패: {e}")