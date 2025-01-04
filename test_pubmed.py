import asyncio
from core.data_source.data_source_manager import DataSourceManager

async def test_pubmed():
    manager = DataSourceManager()
    
    # 1. 기본 검색 테스트
    print("\n=== 기본 검색 테스트 ===")
    results = await manager.collect_data('pubmed', 'Vitamin C benefits', 2)
    for i, result in enumerate(results, 1):
        print(f'\n[결과 {i}]')
        print(f'제목: {result.get("title", "")}')
        print(f'초록 일부: {result.get("abstract", "")[:200]}...')
        print(f'저널: {result.get("journal", "")}')
        print(f'출판일: {result.get("publication_date", "")}')
    
    # API 호출 제한을 피하기 위한 지연
    await asyncio.sleep(2)
    
    # 2. 영양제 검색 테스트
    print("\n=== 영양제 검색 테스트 ===")
    pubmed_source = manager.sources['pubmed']
    supp_results = await pubmed_source.search_supplement('Vitamin D', 2)
    for i, result in enumerate(supp_results, 1):
        print(f'\n[결과 {i}]')
        print(f'제목: {result.get("title", "")}')
        print(f'검색 카테고리: {result.get("search_categories", [])}')
        print(f'초록 일부: {result.get("abstract", "")[:200]}...')
    
    # API 호출 제한을 피하기 위한 지연
    await asyncio.sleep(2)
    
    # 3. 상호작용 검색 테스트
    print("\n=== 상호작용 검색 테스트 ===")
    int_results = await pubmed_source.search_interaction('Vitamin D', 'Calcium', 2)
    for i, result in enumerate(int_results, 1):
        print(f'\n[결과 {i}]')
        print(f'제목: {result.get("title", "")}')
        print(f'검색 카테고리: {result.get("search_categories", [])}')
        print(f'초록 일부: {result.get("abstract", "")[:200]}...')

if __name__ == "__main__":
    asyncio.run(test_pubmed()) 