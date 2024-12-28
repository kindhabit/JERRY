import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI

logger = logging.getLogger(__name__)

class MedicalDataManager:
    def __init__(self, config):
        self.embeddings = OpenAIEmbeddings(openai_api_key=config["openai"]["api_key"])
        self.llm = OpenAI(openai_api_key=config["openai"]["api_key"])
        
        # 의학 용어 벡터 DB 초기화
        self.terms_vectordb = Chroma(
            collection_name="medical_terms",
            embedding_function=self.embeddings,
            persist_directory=config["chroma"]["medical_terms_dir"]
        )
        
        # 의학 지식 벡터 DB 초기화
        self.knowledge_vectordb = Chroma(
            collection_name="medical_knowledge",
            embedding_function=self.embeddings,
            persist_directory=config["chroma"]["medical_knowledge_dir"]
        )

    async def translate_and_validate(self, korean_term):
        """
        RAG를 활용한 의학 용어 번역 및 검증
        1. 유사 용어 검색
        2. 컨텍스트 기반 번역
        3. 의학 지식 기반 검증
        """
        # 1. 유사 용어 검색
        similar_terms = self.terms_vectordb.similarity_search(
            korean_term,
            k=3,
            filter={"language": "korean"}
        )
        
        # 2. 컨텍스트 구성
        context = self._build_translation_context(korean_term, similar_terms)
        
        # 3. LLM을 통한 번역
        translation_prompt = f"""
        Based on the following context and similar medical terms, 
        translate the Korean medical term to English.
        
        Term to translate: {korean_term}
        
        Context:
        {context}
        
        Provide the translation in JSON format with confidence score and reasoning.
        """
        
        translation_result = await self.llm.agenerate([translation_prompt])
        translated = self._parse_translation_result(translation_result)
        
        # 4. 의학 지식 기반 검증
        validation_result = await self._validate_with_knowledge_base(
            korean_term, 
            translated["english"]
        )
        
        # 5. 결과 저장
        if validation_result["is_valid"]:
            await self._store_validated_term(korean_term, translated, validation_result)
            
        return {
            "korean": korean_term,
            "english": translated["english"],
            "confidence": validation_result["confidence"],
            "category": validation_result["category"],
            "references": validation_result["references"]
        }

    async def _validate_with_knowledge_base(self, korean, english):
        """
        의학 지식 벡터 DB를 활용한 번역 검증
        """
        # 관련 의학 지식 검색
        relevant_docs = self.knowledge_vectordb.similarity_search(
            english,
            k=5
        )
        
        validation_prompt = f"""
        Validate the translation of medical term from Korean to English
        using the following medical knowledge context.
        
        Korean term: {korean}
        English translation: {english}
        
        Medical knowledge context:
        {self._format_documents(relevant_docs)}
        
        Provide validation result in JSON format including:
        - is_valid (boolean)
        - confidence (float)
        - category (string)
        - reasoning (string)
        - references (list)
        """
        
        validation_result = await self.llm.agenerate([validation_prompt])
        return self._parse_validation_result(validation_result)

    async def _store_validated_term(self, korean, translation, validation):
        """
        검증된 용어를 벡터 DB에 저장
        """
        self.terms_vectordb.add_texts(
            texts=[korean],
            metadatas=[{
                "korean": korean,
                "english": translation["english"],
                "category": validation["category"],
                "confidence": validation["confidence"],
                "references": validation["references"],
                "language": "korean"
            }]
        )
        
        # 영문 버전도 저장
        self.terms_vectordb.add_texts(
            texts=[translation["english"]],
            metadatas=[{
                "korean": korean,
                "english": translation["english"],
                "category": validation["category"],
                "confidence": validation["confidence"],
                "references": validation["references"],
                "language": "english"
            }]
        ) 