from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import openai

class MedicalTranslator:
    def __init__(self, config):
        self.embeddings = OpenAIEmbeddings(openai_api_key=config["openai"]["api_key"])
        
        # 의학 용어 벡터 DB
        self.medical_terms_vectordb = Chroma(
            collection_name="medical_terms_ko_en",
            embedding_function=self.embeddings,
            persist_directory=config["chroma"]["medical_terms_persist_directory"]
        )
        
        # 기본 사전 데이터는 유지 (폴백용)
        self.medical_terms_db = {
            "고혈압": "hypertension",
            "당뇨": "diabetes",
            # ... 기본 용어들
        }
    
    async def translate_medical_term(self, korean_term):
        """
        RAG를 사용한 의학 용어 번역
        1. 벡터 DB에서 유사한 용어 검색
        2. 없으면 기본 사전 검색
        3. 둘 다 없으면 OpenAI로 번역 후 벡터 DB에 저장
        """
        # 1. 벡터 DB 검색
        results = self.medical_terms_vectordb.similarity_search(
            korean_term,
            k=1
        )
        
        if results:
            return results[0].metadata.get("english_term")
            
        # 2. 기본 사전 검색
        if korean_term in self.medical_terms_db:
            return self.medical_terms_db[korean_term]
            
        # 3. OpenAI로 번역
        translated_term = await self.translate_with_openai(korean_term)
        
        # 4. 새로운 번역 결과를 벡터 DB에 저장
        self.medical_terms_vectordb.add_texts(
            texts=[korean_term],
            metadatas=[{
                "korean_term": korean_term,
                "english_term": translated_term,
                "source": "openai_translation",
                "verified": False  # 검증 필요 표시
            }]
        )
        
        return translated_term
            
    def translate_with_openai(self, text):
        """
        OpenAI API를 사용하여 의학 전문 용어 번역
        """
        prompt = f"""
        Translate the following Korean medical term to English.
        Please provide the most accurate medical terminology:
        Korean: {text}
        English:"""
        
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.3,  # 정확성을 위해 낮은 temperature
            max_tokens=100
        )
        return response.choices[0].text.strip() 