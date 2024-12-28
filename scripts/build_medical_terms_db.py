import pandas as pd
from modules.medical_translator import MedicalTranslator
from config.config_loader import CONFIG

def load_medical_terms():
    """
    여러 소스에서 의학 용어 데이터 수집
    1. UMLS 데이터
    2. 공개 의학 용어집
    3. 검증된 번역 데이터
    """
    terms = []
    
    # CSV 파일에서 데이터 로드
    df = pd.read_csv("data/medical_terms.csv")
    terms.extend(df.to_dict("records"))
    
    return terms

def main():
    translator = MedicalTranslator(CONFIG)
    terms = load_medical_terms()
    
    # 벡터 DB에 데이터 추가
    for term in terms:
        translator.medical_terms_vectordb.add_texts(
            texts=[term["korean"]],
            metadatas=[{
                "korean_term": term["korean"],
                "english_term": term["english"],
                "source": term["source"],
                "verified": True,
                "category": term["category"]
            }]
        )

if __name__ == "__main__":
    main() 