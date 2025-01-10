import chromadb; client = chromadb.HttpClient(host='10.0.1.10', port=8001); print('컬렉션 목록:', [c.name for c in client.list_collections()])
