# modules/pubmed.py

import requests
from lxml import etree
import logging

logger = logging.getLogger(__name__)

class PubMedClient:
    def __init__(self, config):
        self.search_url = config['pubmed']['search_url']
        self.fetch_url = config['pubmed']['fetch_url']
        self.supplements = config['pubmed']['supplements']
        self.health_keywords = config['pubmed']['health_keywords']
        self.session = self._get_session()

    def _get_session(self):
        session = requests.Session()
        retries = requests.adapters.Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def search_pmids(self, query, retmax=5):
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": retmax,
        }
        try:
            response = self.session.get(self.search_url, params=params)
            response.raise_for_status()
            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            logger.info(f"검색된 PMID 수: {len(pmids)} for query: {query}")
            return pmids
        except requests.RequestException as e:
            logger.error(f"PubMed 검색 중 오류 발생: {e}")
            return []

    def fetch_abstracts(self, pmids):
        if not pmids:
            return []
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        try:
            response = self.session.get(self.fetch_url, params=params)
            response.raise_for_status()
            root = etree.fromstring(response.content)
            abstracts = []
            for article in root.xpath("//PubmedArticle"):
                abs_text_elems = article.xpath(".//AbstractText")
                if abs_text_elems:
                    full_abstract = " ".join([a.text for a in abs_text_elems if a.text])
                    pmid = article.xpath(".//PMID")[0].text if article.xpath(".//PMID") else "N/A"
                    abstracts.append((pmid, full_abstract))
            logger.info(f"수집된 초록 수: {len(abstracts)}")
            return abstracts
        except (requests.RequestException, etree.XMLSyntaxError) as e:
            logger.error(f"PubMed 초록 가져오는 중 오류 발생: {e}")
            return []

    def get_queries(self):
        for supplement in self.supplements:
            for health_kw in self.health_keywords:
                yield f"{supplement} {health_kw}"
