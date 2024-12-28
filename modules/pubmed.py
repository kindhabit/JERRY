# modules/pubMed.py

import logging
from aiohttp import ClientSession
from lxml import etree
from typing import List, Tuple, Optional, Dict
from config.config_loader import CONFIG
import aiohttp

logger = logging.getLogger(__name__)

class PubMedClient:
    def __init__(self):
        pubmed_config = CONFIG["pubmed"]
        self.search_url = pubmed_config["search_url"]
        self.fetch_url = pubmed_config["fetch_url"]
        self.supplements = pubmed_config["supplements"]
        self.health_keywords = pubmed_config["health_keywords"]
        self.max_results = pubmed_config.get("max_results", 10)
        self.session = None
        logger.info("PubMed 클라이언트 초기화 완료")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    async def _ensure_session(self):
        """세션이 없으면 새로 생성"""
        if self.session is None:
            self.session = ClientSession()

    async def search_pmids(self, query: str, retmax: int = 5) -> List[str]:
        """PubMed에서 PMID 검색"""
        await self._ensure_session()
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": retmax,
        }
        try:
            async with self.session.get(self.search_url, params=params) as response:
                data = await response.json()
                pmids = data.get("esearchresult", {}).get("idlist", [])
                logger.info(f"검색된 PMID 수: {len(pmids)} for query: {query}")
                return pmids
        except Exception as e:
            logger.error(f"PubMed 검색 중 오류 발생: {e}")
            return []

    async def fetch_abstracts(self, pmids: List[str]) -> List[Dict]:
        """PMID로 초록 이터 가져오기"""
        if not pmids:
            return []
            
        await self._ensure_session()
        
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        try:
            async with self.session.get(self.fetch_url, params=params) as response:
                content = await response.text()
                root = etree.fromstring(content.encode('utf-8'))
                abstracts = []
                
                for article in root.xpath("//PubmedArticle"):
                    abs_text_elems = article.xpath(".//AbstractText")
                    full_abstract = " ".join(
                        [a.text for a in abs_text_elems if a.text]) if abs_text_elems else "No Abstract"
                    pmid = article.xpath(".//PMID")[0].text if article.xpath(".//PMID") else "N/A"
                    title = article.xpath(".//ArticleTitle")[0].text if article.xpath(".//ArticleTitle") else "No Title"
                    
                    # 딕셔너리로 반환
                    abstracts.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": full_abstract
                    })
                    
                logger.info(f"수집된 초록 수: {len(abstracts)}")
                return abstracts
                
        except Exception as e:
            logger.error(f"PubMed 초록 가져오는 중 오류 발생: {e}")
            return []

    def get_queries(self):
        """검색 쿼리 생성기"""
        for supplement in self.supplements:
            for health_kw in self.health_keywords:
                yield f"{supplement} {health_kw}"

    async def close(self):
        """세션 종료"""
        if self.session:
            await self.session.close()
            self.session = None

    async def fetch_structured_abstract(self, pmid: str) -> Optional[Dict]:
        """단일 PMID에 대한 구조화된 초록 데이터 가져오기"""
        abstracts = await self.fetch_abstracts([pmid])
        if abstracts:
            return abstracts[0]
        return None
