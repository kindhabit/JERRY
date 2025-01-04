# JERRY 프로젝트

## 프로젝트 개요
JERRY는 영양제 추천 및 상호작용 분석을 위한 지능형 시스템입니다.

## 기술 문서

- [임베딩 시스템 기술 문서](0_docs/embedding.md)
  - 영양제 관련 논문 데이터 벡터화
  - ChromaDB 저장 및 검색 시스템
  - 임베딩 프로세스 및 아키텍처

## 시스템 구조

### 주요 컴포넌트
- ChromaManager: ChromaDB 연결 및 컬렉션 관리
- EmbeddingCreator: OpenAI API를 사용한 임베딩 생성
- DataSourceManager: PubMed API를 통한 논문 데이터 수집
- OpenAIClient: OpenAI API 연동 및 임베딩 생성

### 데이터베이스
ChromaDB에는 다음과 같은 컬렉션들이 존재합니다:
1. supplements: 영양제 기본 정보
2. interactions: 영양제 간 상호작용
3. health_data: 건강 관련 데이터
4. health_metrics: 건강 지표
5. medical_terms: 의학 용어 사전

## 설치 및 실행

### 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv v_jerry
source v_jerry/bin/activate  # Linux/Mac
v_jerry\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 환경 변수 설정
`.env` 파일에 다음 내용을 추가:
```
OPENAI_API_KEY=your_api_key
PUBMED_API_KEY=your_api_key
PYTHONPATH=/workspace/JERRY/1_SRC
```

### 실행 방법
1. 데이터베이스 초기화:
```bash
python 1_SRC/core/vector_db/vector_store_manager.py --action reinit --force --debug
```

2. API 서버 실행:
```bash
python 1_SRC/main/app.py
```

## 개발 가이드
자세한 개발 가이드는 [임베딩 시스템 기술 문서](0_docs/embedding.md)를 참조하세요. 