async def load_structured_data():
    """구조화된 데이터 적재"""
    chroma_client = ChromaDBClient()
    pubmed_client = PubMedClient()

    for supplement in CONFIG["pubmed"]["supplements"]:
        # 기본 데이터 수집
        papers = await pubmed_client.search_papers(supplement)
        
        for paper in papers:
            # 데이터 구조화
            structured_data = await analyze_paper_content(paper)
            
            # 다층 구조로 저장
            await chroma_client.add_supplement_data(structured_data)

async def analyze_paper_content(paper: Dict) -> Dict:
    """논문 내용 분석 및 구조화"""
    analyzer = TextAnalyzer()
    
    return {
        "supplement_name": paper["supplement"],
        "pmid": paper["pmid"],
        "title": paper["title"],
        "abstract": paper["abstract"],
        "interactions": await analyzer.extract_interactions(paper["abstract"]),
        "health_effects": await analyzer.extract_health_effects(paper["abstract"]),
        "contraindications": await analyzer.extract_contraindications(paper["abstract"])
    } 