# JERRY 프로젝트 임베딩 시스템 기술 문서

## 1. 개요

JERRY 프로젝트의 임베딩 시스템은 영양제 관련 논문 데이터를 벡터화하여 ChromaDB에 저장하고 검색하는 기능을 제공합니다.

## 2. 시스템 구조

### 2.1 파일 구조
```
1_SRC/
├── core/
│   ├── vector_db/
│   │   ├── vector_store_manager.py   # ChromaDB 관리
│   │   ├── embedding_creator.py      # 임베딩 생성
│   │   └── __init__.py
│   └── data_source/
│       ├── data_source_manager.py    # PubMed 데이터 수집
│       └── __init__.py
├── utils/
│   ├── openai_client.py             # OpenAI API 클라이언트
│   ├── logger_config.py             # 로깅 설정
│   └── __init__.py
├── config/
│   ├── config.yaml                  # 기본 설정
│   └── health_mapping.yaml          # 건강 데이터 매핑
└── models/
    ├── health_data.py              # 건강 데이터 모델
    ├── interaction.py              # 상호작용 모델
    └── supplement.py               # 영양제 모델
```

### 2.2 임베딩 프로세스 순서도
```mermaid
graph TD
    A[config.yaml 로드] --> B[ChromaDB 초기화]
    B --> C[5개 컬렉션 생성]
    
    C --> D[영양제 데이터 수집]
    C --> E[건강 데이터 수집]
    C --> F[상호작용 데이터 수집]
    C --> G[의학 용어 데이터 수집]
    
    D --> D1[영양제 목록 로드]
    D1 --> D2[각 영양제별 PubMed 검색]
    D2 --> D3{카테고리별 검색}
    D3 --> D31[효능/효과]
    D3 --> D32[안전성]
    D3 --> D33[상호작용]
    D3 --> D34[임상연구]
    D31 & D32 & D33 & D34 --> D4[LLM 분석]
    D4 --> D5[임베딩 생성]
    D5 --> D6[supplements 컬렉션 저장]
    
    E --> E1[건강 키워드 로드]
    E1 --> E2[키워드별 PubMed 검색]
    E2 --> E3[LLM 분석]
    E3 --> E4[임베딩 생성]
    E4 --> E5[health_data 컬렉션 저장]
    
    F --> F1[상호작용 쌍 로드]
    F1 --> F2[상호작용별 PubMed 검색]
    F2 --> F3[LLM 분석]
    F3 --> F4[임베딩 생성]
    F4 --> F5[interactions 컬렉션 저장]
    
    G --> G1[의학 용어 로드]
    G1 --> G2[용어별 PubMed 검색]
    G2 --> G3[LLM 분석]
    G3 --> G4[임베딩 생성]
    G4 --> G5[medical_terms 컬렉션 저장]
```

### 2.3 데이터 수집 프로세스

1. **초기화 단계**
   ```python
   ChromaManager.__init__()
   ├── _initialize_chroma_client()
   │   └── ConfigLoader.get_service_settings()
   └── [ChromaDB 연결 설정]
   ```

2. **컬렉션 설정**
   ```python
   ChromaManager.reinitialize_database()
   └── _initialize_collections()
       ├── supplements: 영양제 기본 정보
       ├── interactions: 영양제 간 상호작용
       ├── health_data: 건강 관련 데이터
       ├── health_metrics: 건강 지표
       └── medical_terms: 의학 용어 사전
   ```

3. **데이터 수집 단계**
   ```python
   ChromaManager.initialize_data()
   ├── [영양제 데이터 수집]
   │   ├── ConfigLoader.get_supplements()
   │   └── [각 영양제별 반복]
   │       ├── PubMed 검색 (4개 카테고리)
   │       ├── LLM 분석
   │       └── ChromaDB 저장
   │
   ├── [건강 데이터 수집]
   │   ├── ConfigLoader.get_health_keywords()
   │   └── [각 키워드별 반복]
   │
   ├── [상호작용 데이터 수집]
   │   ├── ConfigLoader.get_interaction_pairs()
   │   └── [각 상호작용 쌍별 반복]
   │
   └── [의학 용어 데이터 수집]
       ├── ConfigLoader.get_medical_terms()
       └── [각 용어별 반복]
   ```

각 수집 프로세스는 독립적으로 실행되며, 한 프로세스의 실패가 다른 프로세스에 영향을 주지 않습니다. 각 데이터는 수집 즉시 개별적으로 저장되어, 중간에 프로세스가 중단되어도 이전 데이터는 안전하게 보관됩니다.

### 2.3 주요 컴포넌트

- **ChromaManager**: ChromaDB 연결 및 컬렉션 관리
- **EmbeddingCreator**: OpenAI API를 사용한 임베딩 생성
- **DataSourceManager**: PubMed API를 통한 논문 데이터 수집
- **OpenAIClient**: OpenAI API 연동 및 임베딩 생성

### 2.4 API 응답 예시

#### PubMed API 응답
```json
{
    "pmid": "39751483",
    "title": "Basic Science and Pathogenesis",
    "abstract": "...",
    "authors": ["Lazarus SS", "..."],
    "journal": "Alzheimers Dement",
    "publication_date": "2024 Dec"
}
```

#### LLM 분석 결과
```json
{
    "key_findings": "...",
    "supplement_effects": "...",
    "safety_considerations": "...",
    "clinical_significance": "...",
    "authors_formatted": "...",
    "categories_formatted": "..."
}
```

#### ChromaDB 저장 형식
```json
{
    "ids": ["paper_39751483"],
    "embeddings": [[0.123, ...]],
    "metadatas": [{
        "pmid": "39751483",
        "title": "...",
        "abstract": "...",
        "authors": "...",
        "journal": "...",
        "publication_date": "...",
        "llm_analysis": "..."
    }]
}
```

### 2.2 컬렉션 구조

ChromaDB에는 다음과 같은 컬렉션들이 존재합니다:

1. **supplements**: 영양제 기본 정보
   - 메타데이터: type, name, category, analysis_method, analysis_time, confidence
2. **interactions**: 영양제 간 상호작용
   - 메타데이터: type, supplements, interaction_type, severity
3. **health_data**: 건강 관련 데이터
   - 메타데이터: type, category, keywords, source
4. **health_metrics**: 건강 지표
   - 메타데이터: type, metric_name, category, related_factors
5. **medical_terms**: 의학 용어 사전
   - 메타데이터: type, term_ko, term_en, category

## 3. 임베딩 프로세스

### 3.1 데이터 수집
1. `config.yaml`에서 영양제 목록 로드
2. PubMed API를 통해 각 영양제 관련 논문 검색
3. 논문 메타데이터 및 초록 수집

### 3.2 데이터 전처리
1. 논문 데이터 유효성 검증
   - 제목 존재 여부
   - 초록 길이 검증
   - 저자 정보 확인
2. 저자, 제목, 초록 등 필요 정보 추출
3. LLM을 통한 논문 내용 분석
   - key_findings
   - supplement_effects
   - safety_considerations
   - clinical_significance

### 3.3 임베딩 생성
1. OpenAI API를 사용하여 텍스트 임베딩 생성
   - 텍스트 정규화
   - API 호출 최적화
   - 에러 처리
2. 임베딩 벡터 정규화 및 검증
3. ChromaDB에 데이터 저장

## 4. 주요 기능

### 4.1 데이터베이스 초기화
```python
async def reinitialize_database(self, force: bool = False)
```
- 기존 컬렉션 삭제
- 새 컬렉션 생성
- 초기 데이터 임베딩

### 4.2 데이터 추가
```python
async def _add_paper_to_collection(self, collection_name: str, paper: Dict) -> bool
```
- 논문 데이터 임베딩 생성
- 메타데이터 구성
- ChromaDB에 저장

### 4.3 임베딩 생성
```python
def __call__(self, input: str | List[str]) -> List[List[float]]
```
- 텍스트 임베딩 생성
- 캐시 관리
- 에러 처리

## 5. 에러 처리

- 초록이 없는 논문 스킵
- 임베딩 생성 실패 시 로깅
- API 연결 오류 처리
- 데이터 유효성 검증 실패 처리

### 5.1 임베딩 생성 에러 처리
```python
try:
    embedding = self.client.create_embedding(text)
except Exception as e:
    logger.error(f"임베딩 생성 실패: {str(e)}")
    return [[0.0] * 1536] * len(texts)
```

## 6. 모니터링

### 6.1 로깅
- 상세한 프로세스 로깅
- 에러 및 경고 메시지
- 성능 메트릭스

### 6.2 상태 확인
```bash
python vector_store_manager.py --action stats --debug
```
- 컬렉션별 문서 수
- 메타데이터 필드 확인
- 임베딩 상태 점검

### 6.3 캐시 모니터링
```python
def get_cache_stats(self) -> dict:
    return {
        "cache_size": len(self._cache),
        "cache_hits": self.cache_hits,
        "cache_misses": self.cache_misses
    }
```

## 7. 설정

### 7.1 환경 변수
- OPENAI_API_KEY: OpenAI API 인증
- PUBMED_API_KEY: PubMed API 인증
- PYTHONPATH: 프로젝트 경로 설정

### 7.2 설정 파일
- config.yaml: 기본 설정
  - ChromaDB 연결 정보
  - OpenAI 모델 설정
  - 영양제 목록
- health_mapping.yaml: 건강 데이터 매핑

## 8. 성능 고려사항

### 8.1 임베딩 생성
- 배치 처리로 API 호출 최적화
- 임베딩 캐싱으로 중복 방지
  - 메모리 기반 캐시
  - 캐시 히트율 모니터링
- 병렬 처리 지원

### 8.2 데이터베이스
- 벡터 인덱싱 최적화
- 메모리 사용량 관리
- 쿼리 성능 최적화

## 9. 향후 개선사항

### 9.1 성능 개선
- 분산 처리 지원
- 영구 캐시 저장소
- 배치 크기 최적화

### 9.2 기능 개선
- 실시간 업데이트 지원
- 증분 업데이트
- 벡터 검색 최적화

# 벡터 스토어 관리

## 1. 초기화 (Initialize)
...

## 2. 업데이트 (Update)

### 2.1 기본 업데이트
벡터 스토어의 업데이트는 기존 데이터를 유지하면서 새로운 데이터만 추가하는 방식으로 동작합니다.

```bash
python vector_store_manager.py --action update --debug
```

### 2.2 컬렉션별 제한 업데이트
각 컬렉션별로 처리할 데이터 수를 제한하여 업데이트할 수 있습니다. 이는 개발/테스트 시에 유용합니다.

```bash
python vector_store_manager.py --action update --debug \
    --supplements-limit 10 \
    --interactions-limit 5 \
    --health-data-limit 5
```

#### 제한 옵션
- `--supplements-limit`: 영양제 데이터 처리 제한 수
- `--interactions-limit`: 상호작용 데이터 처리 제한 수
- `--health-data-limit`: 건강 데이터 처리 제한 수

#### 처리 로직
1. 각 컬렉션별 제한이 0인 경우:
   - 해당 컬렉션은 완전히 건너뜀
   - 로그에 "건너뜀" 상태로 표시

2. 각 컬렉션별 제한이 양수인 경우:
   - 지정된 수만큼만 데이터 처리
   - 제한에 도달하면 해당 컬렉션 처리 중단
   - 로그에 처리된 데이터 수 표시 (예: "5/10")

3. 제한이 설정되지 않은 경우:
   - 모든 데이터 처리
   - 기존 동작과 동일

#### 처리 순서
1. supplements 컬렉션 처리
2. interactions 컬렉션 처리
3. health_data 컬렉션 처리

각 컬렉션은 독립적으로 처리되며, 한 컬렉션의 실패가 다른 컬렉션 처리에 영향을 주지 않습니다.

### 2.3 업데이트 결과 확인
업데이트 완료 후 다음과 같은 통계 정보가 표시됩니다:

```
=== 업데이트 결과 ===
검사한 총 PMID 수: 100
기존 PMID 수: 95
새로 추가된 PMID 수: 4
처리 실패 PMID 수: 1

=== 컬렉션별 처리 현황 ===
supplements 컬렉션: 10/10
interactions 컬렉션: 5/5
health_data 컬렉션: 3/5
```

## 3. 상태 확인 (Stats)
``` 