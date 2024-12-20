# modules/embeddings.py

from langchain.embeddings import OpenAIEmbeddings
import logging

logger = logging.getLogger(__name__)

class EmbeddingsClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        logger.info("OpenAI Embeddings 초기화 완료.")

    def get_embeddings(self):
        return self.embeddings
