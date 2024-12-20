from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field
import yaml
import logging
from langchain_chroma import Chroma  # 최신 Chroma import


logger = logging.getLogger(__name__)

def load_config():
    """설정 파일(config.yaml)을 로드합니다."""
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class ChromaDBSettings(BaseModel):
    chroma_api_impl: str = Field(default="rest")
    chroma_server_host: str
    chroma_server_http_port: int
    persist_directory: str = Field(default="./chroma_persistence")

class ChromaDBClient:
    def __init__(self, config, embeddings):
        chroma_config = config
        self.settings = ChromaDBSettings(
            chroma_api_impl=chroma_config.get("chroma_api_impl", "rest"),
            chroma_server_host=chroma_config['server_host'],
            chroma_server_http_port=chroma_config['server_port'],
            persist_directory=chroma_config.get("persist_directory", "./chroma_persistence")
        )
        self.collection_name = chroma_config['collection_name']
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=self.settings.persist_directory
        )
        logger.info(f"Chroma 클라이언트가 초기화되었습니다. 컬렉션: {self.collection_name}")

    def search(self, query, k=5):
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Chroma 검색 중 오류 발생: {e}")
            return []
