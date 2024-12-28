#!/usr/bin/env python3
import asyncio
import logging
import argparse
from modules.db.chroma_client import ChromaDBClient
from modules.pubMed import PubMedClient
from config.config_loader import CONFIG
from modules.text_analyzer import TextAnalyzer
from modules.memory_monitor import MemoryMonitor
import gc
from typing import List, Dict
import json
from langchain_core.output_parsers import JsonOutputParser
from modules.chains import AnalysisChain
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

async def manage_chroma_database(action="update", force=False, debug=False, test_mode=False):
    """ChromaDB 관리 작업 수행"""
    try:
        logger.info(f"ChromaDB 관리 작업 시작: {action} (Debug: {debug}, Test: {test_mode})")
        
        chroma_client = ChromaDBClient()
        
        if action == "reinit":
            if force:
                logger.warning("=== 데이터베이스 초기화 모드 ===")
                if not test_mode:
                    confirm = input("모든 데이터가 삭제됩니다. 계속하시겠습니까? (y/N): ")
                    if confirm.lower() != 'y':
                        logger.info("작업이 취소되었습니다.")
                        return
                
                # 데이터 완전 삭제 및 재생성
                chroma_client.reset_collection(force=True)
                
                if not test_mode:
                    # PubMed 데이터 새로 로드
                    await add_pubmed_to_chroma(chroma_client, test_mode=test_mode)
                
                logger.info("데이터베이스 초기화 및 데이터 로드 완료")
            else:
                logger.error("데이터베이스 초기화는 --force 옵션이 필요합니다")
                return
            
        elif action == "update":
            logger.info("=== 데이터베이스 업데이트 모드 ===")
            await add_pubmed_to_chroma(chroma_client, test_mode=test_mode)
            logger.info("데이터베이스 업데이트 완료")
            
        elif action == "stats":
            logger.info("=== 데이터베이스 상태 확인 ===")
            show_database_stats(chroma_client)
        
    except Exception as e:
        logger.error(f"관리 작업 중 오류 발생: {str(e)}", exc_info=True)
        raise

async def add_pubmed_to_chroma(chroma_client: ChromaDBClient, test_mode: bool = False):
    """PubMed 데이터를 ChromaDB에 추가"""
    memory_monitor = MemoryMonitor()
    collection_stats = {}  # 수집 통계
    try:
        # CONFIG 로드 확인
        logger.info("\n=== 설정된 수집 대상 ===")
        logger.info(f"성분 목록: {CONFIG['pubmed']['supplements']}")
        logger.info(f"키워드 목록: {CONFIG['pubmed']['health_keywords']}")
        logger.info("="*50)
        
        pubmed_client = PubMedClient()
        text_analyzer = TextAnalyzer()
        total_processed = 0
        total_skipped = 0
        
        for supplement in CONFIG["pubmed"]["supplements"]:
            logger.info(f"\n>>> 처리 시작: {supplement}")
            if supplement not in collection_stats:
                collection_stats[supplement] = {
                    "total": 0,
                    "processed": 0,
                    "skipped": 0,
                    "combinations": set()
                }
            
            for keyword in CONFIG["pubmed"]["health_keywords"]:
                query = f"{supplement} {keyword}"
                logger.info(f"\n검색 시도: {query}")
                pmids = await pubmed_client.search_pmids(query)
                total_pmids = len(pmids)
                if total_pmids == 0:
                    logger.info(f"검색 결과 없음: {supplement} + {keyword}")
                    continue
                
                logger.info(f"\n{'='*50}")
                logger.info(f"검색: {supplement} + {keyword}")
                logger.info(f"결과: {total_pmids}개 문서")
                logger.info(f"진행: [{'='*20}] 0%")
                
                for idx, pmid in enumerate(pmids, 1):
                    abstract_data = await pubmed_client.fetch_structured_abstract(pmid)
                    if not abstract_data or abstract_data.get("abstract") == "No Abstract":
                        logger.warning(f"초록 없음 (PMID: {pmid}) - 건너뜀")
                        total_skipped += 1
                        collection_stats[supplement]["skipped"] += 1
                        continue
                        
                    progress = int((idx/total_pmids) * 20)
                    logger.info(f"진행: [{'='*progress}{' '*(20-progress)}] {int(idx/total_pmids*100)}%")
                    
                    try:
                        interactions = await text_analyzer.extract_interactions(abstract_data["abstract"])
                        if interactions.get("supplements_mentioned", {}).get("combinations"):
                            collection_stats[supplement]["combinations"].update(
                                interactions["supplements_mentioned"]["combinations"]
                            )
                        
                        structured_data = {
                            "pmid": pmid,
                            "title": abstract_data["title"],
                            "abstract": abstract_data["abstract"],
                            "supplement_name": supplement,
                            "interactions": interactions
                        }
                        
                        chroma_client.add_supplement_data(structured_data)
                        total_processed += 1
                        collection_stats[supplement]["processed"] += 1
                        
                        if memory_monitor.check_memory_threshold():
                            logger.warning("메모리 임계값 초과, GC 실행")
                            gc.collect()
                            memory_monitor.log_memory_usage()
                            
                    except Exception as e:
                        logger.error(f"초록 처리 중 오류: {str(e)}")
                        logger.error(f"문제된 데이터: {structured_data}")
                        raise
                    
                if test_mode:
                    return
                
        # 최종 통계 출력
        logger.info("\n\n=== 수집 완료 통계 ===")
        for supp, stats in collection_stats.items():
            logger.info(f"\n{supp}:")
            logger.info(f"- 처리된 문서: {stats['processed']}")
            logger.info(f"- 건너뛴 문서: {stats['skipped']}")
            if stats['combinations']:
                logger.info("- 발견된 조합:")
                for combo in stats['combinations']:
                    logger.info(f"  * {combo}")
        
    except Exception as e:
        logger.error(f"데이터 추가 중 오류 발생: {str(e)}")
        raise
    finally:
        memory_monitor.log_memory_usage()
        await pubmed_client.close()  # 세션 정리

async def process_batch(chroma_client: ChromaDBClient, batch: List[Dict]):
    """배치 데이터 처리"""
    try:
        logger.info(f"배치 처리 시작 (기: {len(batch)})")
        for data in batch:
            await chroma_client.add_supplement_data(data)
        logger.info("배치 처리 완료")
    except Exception as e:
        logger.error(f"배치 처리 중 오류 발생: {str(e)}")
        raise

def show_database_stats(chroma_client):
    """데이터베이스 상태 출력"""
    try:
        stats = chroma_client.get_data_stats()
        
        logger.info("\n=== ChromaDB 컬렉션 상태 ===")
        for name, info in stats["collections"].items():
            status = "🟢" if info["status"] == "active" else "🔴"
            logger.info(f"{status} {name}: {info['count']} 문서")
        
        logger.info(f"\n총 문서 수: {stats['total_documents']}")
        
        logger.info("\n1. 저장된 보충제:")
        for supplement, info in stats.get("supplements", {}).items():
            logger.info(f"- {supplement}: {info['count']}개 문서, PMID: {', '.join(info['pmids'])}")
        
    except Exception as e:
        logger.error(f"통계 조회 중 오류 발생: {str(e)}")
        raise

async def check_database_stats():
    try:
        chroma_client = ChromaDBClient()
        stats = chroma_client.get_data_stats()
        
        # 상세 통계 출력
        print("\n=== 데이터베이스 통계 ===")
        print(f"총 문서 수: {stats['total_documents']}")
        
        print("\n--- 서플리먼트별 통계 ---")
        for supp, count in stats['known_supplements'].items():
            print(f"- {supp}: {count}개 문서")
        
        print("\n--- 새로 발견된 서플리먼트 ---")
        for supp, count in stats['discovered_supplements'].items():
            print(f"- {supp}: {count}개 문서")
        
        if stats.get('interactions'):
            print("\n--- 상호작용 통계 ---")
            print(f"총 상호작용 수: {stats['interactions']['total']}")
            print("\n심각도별:")
            for severity, count in stats['interactions']['by_severity'].items():
                print(f"- {severity}: {count}개")
        
        return stats
        
    except Exception as e:
        logger.error(f"통계 조회 실패: {str(e)}")
        raise

async def search_database(query: str = None, supplement: str = None):
    try:
        chroma_client = ChromaDBClient()
        
        if supplement:
            results = chroma_client.similarity_search(
                f"supplement:{supplement} {query if query else ''}"
            )
        else:
            results = chroma_client.similarity_search(query)
            
        print("\n=== 검색 결과 ===")
        for i, doc in enumerate(results.get('documents', []), 1):
            print(f"\n문서 {i}:")
            print(f"내용: {doc[:200]}...")
            print(f"메타데이터: {results['metadatas'][i-1]}")
    except Exception as e:
        logger.error(f"검색 실패: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="ChromaDB 관리 도구")
    parser.add_argument("--action", 
                     choices=["reinit", "update", "stats"],
                     required=True,
                     help="수행할 작업")
    parser.add_argument("--force",
                     action="store_true",
                     help="데이터베이스 초기화 시 강제 실행")
    parser.add_argument("--debug",
                     action="store_true",
                     help="디버그 모드 활성화")
    
    args = parser.parse_args()
    
    # 로그 설정
    if args.debug:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        # 필요한 모듈만 INFO 레벨로 설정
        logging.getLogger('modules.db.chroma_client').setLevel(logging.INFO)
        logging.getLogger('modules.pubMed').setLevel(logging.INFO)
        # OpenAI 관련 로그는 ERROR 레벨로 설정
        logging.getLogger('openai').setLevel(logging.ERROR)
        logging.getLogger('httpx').setLevel(logging.ERROR)
        logging.getLogger('httpcore').setLevel(logging.ERROR)
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # 모든 외부 라이브러리 로그 끄기
        for logger_name in logging.root.manager.loggerDict:
            if not logger_name.startswith('modules'):
                logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    
    try:
        if args.action == "stats":
            show_database_stats(ChromaDBClient())
        else:
            asyncio.run(manage_chroma_database(
                action=args.action,
                force=args.force,
                debug=args.debug
            ))
    except Exception as e:
        print(f"예상치 못한 오류 발생: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 