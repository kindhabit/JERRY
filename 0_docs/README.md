# JERRY 시스템 문서 (2025-01-02)

## 프로젝트 개요
JERRY는 건강검진 데이터 기반의 지능형 영양제 추천 시스템입니다. 벡터 데이터베이스와 RAG(Retrieval-Augmented Generation) 기술을 활용하여 개인화된 영양제 추천과 건강 상담을 제공합니다.

## 시스템 구조
```
📁 jerry_project/
├── 📁 config/
│   ├── config.yaml              # 시스템 설정
│   └── config_loader.py         # 설정 로더
│
├── 📁 core/
│   ├── 📁 vector_db/           # 벡터 DB 관련
│   │   ├── vector_store_manager.py  # 벡터 DB 생성/관리
│   │   └── embedding_creator.py     # 임베딩 생성
│   │
│   ├── 📁 services/            # 핵심 서비스
│   │   ├── rag_service.py          # RAG 기반 검색/생성
│   │   ├── pattern_service.py      # 패턴 학습 및 강화
│   │   └── health_service.py       # 통합된 건강 서비스
│   │
│   └── 📁 analysis/
│       └── client_health_analyzer.py # 클라이언트 건강데이터 분석
│
├── 📁 models/                   # 데이터 모델
│   ├── health_data.py          # 건강 데이터 모델
│   └── supplement.py           # 영양제 관련 모델
│
├── 📁 utils/                    # 유틸리티
│   └── system_utils.py         # 통합된 유틸리티
│
└── 📁 external/                 # 외부 연동
    └── pubmed_client.py        # PubMed API 클라이언트
```

## 벡터 DB 구조

### 컬렉션 구조
```python
collections = {
    "supplements": {
        # 영양제 정보 컬렉션
        "embeddings": List[float],  # 1536차원 벡터
        "metadata": {
            "name": str,            # 영양제 이름
            "category": str,        # 분류
            "effects": List[str],   # 효과
            "evidence_level": str,  # 근거 수준
            "timestamp": str,       # 생성/수정 시간
            "source": str          # 데이터 출처
        }
    },
    
    "interactions": {
        # 상호작용 정보 컬렉션
        "embeddings": List[float],
        "metadata": {
            "supplements": List[str],  # 관련 영양제들
            "effect_type": str,       # 상호작용 유형
            "severity": str,          # 심각도
            "mechanism": str,         # 작용 기전
            "evidence": List[str]     # 근거 출처
        }
    },
    
    "health_patterns": {
        # 건강 패턴 컬렉션
        "embeddings": List[float],
        "metadata": {
            "pattern_type": str,      # 패턴 유형
            "conditions": List[str],  # 관련 건강상태
            "frequency": int,         # 발생 빈도
            "confidence": float,      # 신뢰도
            "last_updated": str       # 최종 업데이트
        }
    }
}
```

### 검색 프로세스
```mermaid
graph TD
    A[검색 요청] --> B[벡터 변환]
    B --> C[1차 검색]
    C --> D[메타데이터 필터링]
    D --> E[관련성 순위화]
    E --> F[결과 반환]

    subgraph "벡터 검색 프로세스"
        B --> G[임베딩 생성]
        G --> H[차원 축소]
        H --> I[정규화]
    end

    subgraph "필터링 프로세스"
        D --> J[카테고리 필터]
        D --> K[신뢰도 필터]
        D --> L[시간 필터]
    end
```

## 주요 기능

### 1. 벡터 DB 시스템
- 임베딩 생성 및 저장
- 벡터 검색
- 데이터베이스 관리

### 2. RAG 서비스
- 컨텍스트 기반 검색
- 관련 문서 검색
- 응답 생성

### 3. 패턴 서비스
- 상호작용 패턴 학습
- 패턴 기반 강화
- 신뢰도 평가

### 4. 건강 서비스
- 영양제 추천
- 건강 상담
- 상호작용 분석

## 업데이트 내역

### 2025년
- 2025-01-02
  - 시스템 구조 최적화
    - 벡터 DB 계층 구조 재설계
    - 컬렉션 구조 최적화
    - 검색 프로세스 개선
  - 서비스 통합
    - 추천 + QA 서비스 통합
    - 패턴 서비스 이동
  - 유틸리티 통합
    - common, security, logger 통합

### 2024년
- 2024-12-31: 기본 시스템 구축
- 2024-12-30: 의학 데이터 시스템
- 2024-12-29: 벡터 DB 시스템