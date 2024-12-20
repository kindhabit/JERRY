import uvicorn
from fastapi import FastAPI, HTTPException, Request
from modules.chroma_db import ChromaDBClient  # ChromaDBClient 불러오기
from langchain_openai import OpenAIEmbeddings  # OpenAIEmbeddings import
import yaml
import os

# 설정 파일 로드 함수
def load_config():
    """설정 파일(config.yaml)을 로드합니다."""
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# 설정 로드
config = load_config()

# OpenAI API 키 설정
openai_api_key = config["openai"]["api_key"]
os.environ["OPENAI_API_KEY"] = openai_api_key  # 환경 변수에 API 키 설정

# ChromaDB 클라이언트 초기화
chroma_config = config["chroma"]  # Chroma 관련 설정
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)  # OpenAI Embeddings 객체 생성
chroma_client = ChromaDBClient(chroma_config, embeddings)  # ChromaDBClient 초기화

# FastAPI 애플리케이션 초기화
app = FastAPI()

@app.get("/")
def read_root():
    """루트 엔드포인트."""
    return {"status": "Chroma Project API is running"}

@app.post("/search")
async def search_chroma(request: Request):
    """ChromaDB 유사성 검색 엔드포인트."""
    try:
        # 요청 데이터 파싱
        data = await request.json()
        query = data.get("query", "")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        # ChromaDB 검색 수행
        results = chroma_client.search(query)
        if not results:
            return {"references": []}

        # 검색 결과 반환
        return {"references": [{"title": r.metadata.get("pmid", ""), "authors": "N/A", "source": "ChromaDB"} for r in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=chroma_config["server_host"],  # 설정 파일에서 가져온 host
        port=chroma_config["server_port"],  # 설정 파일에서 가져온 port
        reload=True
    )
