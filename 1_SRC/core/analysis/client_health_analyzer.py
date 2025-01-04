from typing import Dict, Optional, List
from pydantic import BaseModel
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class HealthDataAnalyzer:
    def __init__(self):
        self.logger = logger

    async def analyze_health_data(self, data: Dict) -> Dict:
        """건강 데이터 종합 분석"""
        try:
            # 1. 데이터 파싱 및 검증
            health_data = self.parse_health_data(data)
            
            # 2. 위험 요인 분석
            risk_factors = self.analyze_risk_factors(health_data)
            
            # 3. 건강 상태 컨텍스트 생성
            context = self.build_health_context(health_data, risk_factors)
            
            # 4. 분석 결과 통합
            return {
                "health_data": health_data.model_dump(),
                "risk_factors": risk_factors,
                "context": context,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"건강 데이터 분석 중 오류: {str(e)}")
            raise

    def parse_health_data(self, data: Dict) -> 'HealthData':
        """건강 데이터 파싱"""
        try:
            return HealthData(**data)
        except Exception as e:
            self.logger.error(f"건강 데이터 파싱 실패: {str(e)}")
            raise ValueError(f"잘못된 건강 데이터 형식: {str(e)}")

    def analyze_risk_factors(self, health_data: 'HealthData') -> List[Dict]:
        """건강 위험 요인 분석"""
        risk_factors = []
        health_keywords = CONFIG.get_health_keywords()
        
        # 키워드별 임계값 매핑 생성
        thresholds = {}
        for keyword in health_keywords:
            for condition in keyword.get('conditions', []):
                if 'thresholds' in condition:
                    thresholds[condition['name']] = {
                        'values': condition['thresholds'],
                        'category': keyword['category'],
                        'description': condition.get('description', ''),
                        'lifestyle_factors': keyword.get('lifestyle_factors', [])
                    }
        
        # BMI 분석
        if health_data.bmi:
            obesity_threshold = float(thresholds.get('obesity', {}).get('values', {}).get('bmi', '30').strip('>'))
            underweight_threshold = float(thresholds.get('underweight', {}).get('values', {}).get('bmi', '18.5').strip('<'))
            
            if health_data.bmi >= obesity_threshold:
                risk_factors.append({
                    "type": "obesity",
                    "severity": "high" if health_data.bmi >= obesity_threshold + 5 else "medium",
                    "value": health_data.bmi,
                    "threshold": obesity_threshold,
                    "lifestyle_factors": thresholds.get('obesity', {}).get('lifestyle_factors', [])
                })
            elif health_data.bmi < underweight_threshold:
                risk_factors.append({
                    "type": "underweight",
                    "severity": "medium",
                    "value": health_data.bmi,
                    "threshold": underweight_threshold,
                    "lifestyle_factors": thresholds.get('underweight', {}).get('lifestyle_factors', [])
                })
        
        # 혈압 분석
        if health_data.systolic_bp and health_data.diastolic_bp:
            hypertension = thresholds.get('hypertension', {}).get('values', {})
            systolic_threshold = float(hypertension.get('systolic', '140').strip('>'))
            diastolic_threshold = float(hypertension.get('diastolic', '90').strip('>'))
            
            if health_data.systolic_bp >= systolic_threshold or health_data.diastolic_bp >= diastolic_threshold:
                risk_factors.append({
                    "type": "hypertension",
                    "severity": "high" if health_data.systolic_bp >= systolic_threshold + 20 else "medium",
                    "value": f"{health_data.systolic_bp}/{health_data.diastolic_bp}",
                    "threshold": f"{systolic_threshold}/{diastolic_threshold}",
                    "lifestyle_factors": thresholds.get('hypertension', {}).get('lifestyle_factors', [])
                })
        
        # 콜레스테롤 분석
        if health_data.total_cholesterol:
            chol_threshold = float(thresholds.get('hypercholesterolemia', {}).get('values', {}).get('total', '240').strip('>'))
            if health_data.total_cholesterol > chol_threshold:
                risk_factors.append({
                    "type": "high_cholesterol",
                    "severity": "high" if health_data.total_cholesterol > chol_threshold + 60 else "medium",
                    "value": health_data.total_cholesterol,
                    "threshold": chol_threshold,
                    "lifestyle_factors": thresholds.get('hypercholesterolemia', {}).get('lifestyle_factors', [])
                })
        
        # 간 기능 분석
        liver_thresholds = thresholds.get('elevated_enzymes', {}).get('values', {})
        ast_threshold = float(liver_thresholds.get('ast', '40').strip('>'))
        alt_threshold = float(liver_thresholds.get('alt', '40').strip('>'))
        
        if (health_data.sgotast and health_data.sgotast > ast_threshold) or \
           (health_data.sgptalt and health_data.sgptalt > alt_threshold):
            risk_factors.append({
                "type": "liver_function_abnormal",
                "severity": "high" if (health_data.sgotast and health_data.sgotast > ast_threshold * 2) or \
                                   (health_data.sgptalt and health_data.sgptalt > alt_threshold * 2) else "medium",
                "value": f"AST: {health_data.sgotast}, ALT: {health_data.sgptalt}",
                "threshold": f"AST: {ast_threshold}, ALT: {alt_threshold}",
                "lifestyle_factors": thresholds.get('elevated_enzymes', {}).get('lifestyle_factors', [])
            })
        
        # 생활습관 분석
        if health_data.exercise_frequency is not None:
            exercise_threshold = float(thresholds.get('sedentary', {}).get('values', {}).get('exercise_frequency', '3').strip('<'))
            if health_data.exercise_frequency < exercise_threshold:
                risk_factors.append({
                    "type": "sedentary_lifestyle",
                    "severity": "medium",
                    "value": health_data.exercise_frequency,
                    "threshold": exercise_threshold,
                    "lifestyle_factors": thresholds.get('sedentary', {}).get('lifestyle_factors', [])
                })
        
        return risk_factors

    def build_health_context(
        self, 
        health_data: 'HealthData',
        risk_factors: List[Dict]
    ) -> Dict:
        """건강 상태 컨텍스트 구축"""
        context = {
            "basic_info": {
                "age": health_data.age,
                "gender": health_data.gender,
                "bmi": health_data.bmi
            },
            "risk_factors": risk_factors,
            "current_medications": health_data.current_medications,
            "current_supplements": health_data.current_supplements,
            "health_conditions": health_data.health_conditions,
            "lifestyle": {
                "smoking": health_data.smoking,
                "alcohol": health_data.alcohol,
                "exercise_frequency": health_data.exercise_frequency
            }
        }
        
        # 검사 결과 추가
        if health_data.analysisData:
            context["analysis_data"] = health_data.analysisData
            
        # 암 데이터 추가
        if health_data.cancerdata:
            context["cancer_data"] = health_data.cancerdata
            
        return context

class HealthData(BaseModel):
    # 기본 식별 정보
    uuid: Optional[str] = None
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    
    # 개인 정보
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    
    # 신체 계측
    height: Optional[float] = None
    weight: Optional[float] = None
    bmi: Optional[float] = None
    waist_circumference: Optional[float] = None
    
    # 혈압
    systolic_bp: Optional[int] = None    # 수축기 혈압
    diastolic_bp: Optional[int] = None   # 이완기 혈압
    
    # 혈액 검사
    total_cholesterol: Optional[int] = None
    hdl_cholesterol: Optional[int] = None
    ldl_cholesterol: Optional[int] = None
    triglyceride: Optional[int] = None
    
    # 간 기능
    sgotast: Optional[int] = None        # AST
    sgptalt: Optional[int] = None        # ALT
    gammagtp: Optional[int] = None       # γ-GTP
    
    # 혈당
    fasting_blood_sugar: Optional[int] = None
    
    # 신장 기능
    creatinine: Optional[float] = None
    gfr: Optional[int] = None
    
    # 현재 복용 중인 약물/보조제
    current_medications: Optional[List[str]] = None
    current_supplements: Optional[List[str]] = None
    
    # 건강 상태
    health_conditions: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    
    # 생활습관
    smoking: Optional[bool] = None
    alcohol: Optional[bool] = None
    exercise_frequency: Optional[int] = None  # 주당 운동 횟수
    
    # 분석 결과 저장
    analysisData: Optional[Dict] = None
    cancerdata: Optional[Dict] = None
    
    class Config:
        arbitrary_types_allowed = True 