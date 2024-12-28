from typing import Dict, Optional, List
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class HealthData(BaseModel):
    # 필수 필드
    query: str
    
    # 기본 정보 (Optional)
    uuid: Optional[str] = None
    phoneno: Optional[str] = None
    name: Optional[str] = None
    hosnm: Optional[str] = None
    regdate: Optional[str] = None
    
    # 신체 정보 (Optional)
    height: Optional[float] = None
    weight: Optional[float] = None
    waist_circumference: Optional[float] = None
    bmi: Optional[float] = None
    
    # 혈압 (Optional)
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None
    
    # 간 기능 (Optional)
    sgotast: Optional[int] = None
    sgptalt: Optional[int] = None
    gammagtp: Optional[int] = None
    
    # 콜레스테롤 (Optional)
    total_cholesterol: Optional[int] = None
    hdl_cholesterol: Optional[int] = None
    ldl_cholesterol: Optional[int] = None
    triglyceride: Optional[int] = None
    
    # 혈당 (Optional)
    fasting_blood_sugar: Optional[int] = None
    
    # 신장 기능 (Optional)
    creatinine: Optional[float] = None
    gfr: Optional[int] = None
    
    # 추가 정보 (Optional)
    hmg: Optional[str] = None
    cancerdata: Optional[Dict] = None
    testItem: Optional[List[str]] = None
    diseaseNames: Optional[List[str]] = None
    Findings: Optional[str] = None
    analysisData: Optional[Dict] = None
    
    # ChromaDB 검색용 필드들
    supplements: List[str] = []
    health_metrics: List[str] = []
    conditions: List[str] = []

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        return {k: v for k, v in data.items() if v is not None}

class DataParser:
    @staticmethod
    def parse_health_data(data: Dict) -> HealthData:
        try:
            return HealthData(**data)
        except Exception as e:
            logger.error(f"Health data parsing failed: {str(e)}")
            raise ValueError(f"Invalid health data format: {str(e)}") 