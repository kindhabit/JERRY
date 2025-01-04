from typing import Dict, Optional, List
from pydantic import BaseModel
from datetime import datetime
from config.config_loader import CONFIG

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
    analysis_results: Optional[Dict] = None
    recommendations: Optional[Dict] = None
    
    # 면역기능 지표
    wbc: Optional[float] = None
    neutrophil: Optional[float] = None
    lymphocyte: Optional[float] = None
    crp: Optional[float] = None
    
    # 갑상선기능 지표
    tsh: Optional[float] = None
    t3: Optional[float] = None
    t4: Optional[float] = None
    
    class Config:
        arbitrary_types_allowed = True

    def get_risk_factors(self) -> List[Dict]:
        """건강 지표 검토 필요 항목 식별"""
        review_items = []
        
        # 참조 범위 로드
        ranges = CONFIG.get_reference_ranges()['ranges']
        
        # BMI 체크
        if self.bmi or self.waist_circumference:
            bmi_ranges = ranges['body_composition']
            review_needed = False
            values = []
            
            if self.bmi and self.bmi > float(bmi_ranges['review_needed']['bmi'].strip('> ')):
                review_needed = True
                values.append(f"BMI: {self.bmi}")
            
            if self.waist_circumference and self.gender:
                waist_limit = float(bmi_ranges['review_needed']['waist_m' if self.gender.lower() == 'male' else 'waist_f'].strip('> '))
                if self.waist_circumference > waist_limit:
                    review_needed = True
                    values.append(f"허리둘레: {self.waist_circumference}")
            
            if review_needed:
                review_items.append({
                    "category": "body_composition",
                    "type": "composition",
                    "value": ", ".join(values),
                    "ranges": bmi_ranges,
                    "note": bmi_ranges['note']
                })
        
        # 혈압 체크
        if self.systolic_bp and self.diastolic_bp:
            bp_ranges = ranges['blood_pressure']
            sys_range = bp_ranges['review_needed']['systolic'].split('-')
            dia_range = bp_ranges['review_needed']['diastolic'].split('-')
            
            if (self.systolic_bp >= int(sys_range[0]) or 
                self.diastolic_bp >= int(dia_range[0])):
                review_items.append({
                    "category": "blood_pressure",
                    "type": "bp",
                    "value": f"{self.systolic_bp}/{self.diastolic_bp}",
                    "ranges": bp_ranges,
                    "note": bp_ranges['note']
                })
        
        # 콜레스테롤 체크
        if any([self.total_cholesterol, self.hdl_cholesterol, 
                self.ldl_cholesterol, self.triglyceride]):
            chol_ranges = ranges['cholesterol']
            
            # 각 지표별 검토
            review_needed = False
            values = []
            
            if self.total_cholesterol and self.total_cholesterol > float(chol_ranges['review_needed']['total'].strip('> ')):
                review_needed = True
                values.append(f"총콜레스테롤: {self.total_cholesterol}")
                
            if self.hdl_cholesterol and self.hdl_cholesterol < float(chol_ranges['review_needed']['hdl'].strip('< ')):
                review_needed = True
                values.append(f"HDL: {self.hdl_cholesterol}")
                
            if self.ldl_cholesterol and self.ldl_cholesterol > float(chol_ranges['review_needed']['ldl'].strip('> ')):
                review_needed = True
                values.append(f"LDL: {self.ldl_cholesterol}")
                
            if self.triglyceride and self.triglyceride > float(chol_ranges['review_needed']['triglyceride'].strip('> ')):
                review_needed = True
                values.append(f"중성지방: {self.triglyceride}")
            
            if review_needed:
                review_items.append({
                    "category": "cholesterol",
                    "type": "lipid_profile",
                    "value": ", ".join(values),
                    "ranges": chol_ranges,
                    "note": chol_ranges['note']
                })
        
        # 간 기능 체크
        if any([self.sgotast, self.sgptalt, self.gammagtp]):
            liver_ranges = ranges['liver_function']
            review_needed = False
            values = []
            
            if self.sgotast and self.sgotast > float(liver_ranges['review_needed']['ast'].strip('> ')):
                review_needed = True
                values.append(f"AST: {self.sgotast}")
                
            if self.sgptalt and self.sgptalt > float(liver_ranges['review_needed']['alt'].strip('> ')):
                review_needed = True
                values.append(f"ALT: {self.sgptalt}")
                
            if self.gammagtp and self.gammagtp > float(liver_ranges['review_needed']['ggt'].strip('> ')):
                review_needed = True
                values.append(f"GGT: {self.gammagtp}")
            
            if review_needed:
                review_items.append({
                    "category": "liver_function",
                    "type": "enzymes",
                    "value": ", ".join(values),
                    "ranges": liver_ranges,
                    "note": liver_ranges['note']
                })
        
        # 혈당 체크
        if self.fasting_blood_sugar:
            sugar_ranges = ranges['blood_sugar']
            if self.fasting_blood_sugar > float(sugar_ranges['review_needed']['fasting'].strip('> ')):
                review_items.append({
                    "category": "blood_sugar",
                    "type": "fasting_glucose",
                    "value": self.fasting_blood_sugar,
                    "ranges": sugar_ranges,
                    "note": sugar_ranges['note']
                })
        
        # 신장 기능 체크
        if any([self.creatinine, self.gfr]):
            kidney_ranges = ranges['kidney_function']
            review_needed = False
            values = []
            
            if self.creatinine and self.gender:
                creatinine_limit = float(kidney_ranges['review_needed'][
                    'creatinine_m' if self.gender.lower() == 'male' else 'creatinine_f'
                ].strip('> '))
                if self.creatinine > creatinine_limit:
                    review_needed = True
                    values.append(f"크레아티닌: {self.creatinine}")
            
            if self.gfr and self.gfr < float(kidney_ranges['review_needed']['gfr'].strip('< ')):
                review_needed = True
                values.append(f"GFR: {self.gfr}")
            
            if review_needed:
                review_items.append({
                    "category": "kidney_function",
                    "type": "kidney",
                    "value": ", ".join(values),
                    "ranges": kidney_ranges,
                    "note": kidney_ranges['note']
                })
        
        # 생활습관 체크
        if self.exercise_frequency is not None:
            lifestyle_ranges = ranges['lifestyle']
            if self.exercise_frequency < lifestyle_ranges['exercise']['minimum_weekly']:
                review_items.append({
                    "category": "lifestyle",
                    "type": "exercise",
                    "value": self.exercise_frequency,
                    "recommended": lifestyle_ranges['exercise'],
                    "note": lifestyle_ranges['note']
                })
        
        # 면역기능 체크
        if any([self.wbc, self.neutrophil, self.lymphocyte, self.crp]):
            immune_ranges = ranges['immune_function']
            review_needed = False
            values = []
            
            if self.wbc and self.wbc > float(immune_ranges['review_needed']['wbc'].strip('> ')):
                review_needed = True
                values.append(f"WBC: {self.wbc}")
            
            if self.crp and self.crp > float(immune_ranges['review_needed']['crp'].strip('> ')):
                review_needed = True
                values.append(f"CRP: {self.crp}")
            
            if review_needed:
                review_items.append({
                    "category": "immune",
                    "type": "immune_function",
                    "value": ", ".join(values),
                    "ranges": immune_ranges,
                    "note": immune_ranges['note']
                })

        # 갑상선기능 체크
        if any([self.tsh, self.t3, self.t4]):
            thyroid_ranges = ranges['thyroid_function']
            review_needed = False
            values = []
            
            if self.tsh and self.tsh > float(thyroid_ranges['review_needed']['tsh'].strip('> ')):
                review_needed = True
                values.append(f"TSH: {self.tsh}")
            
            if self.t3 and self.t3 > float(thyroid_ranges['review_needed']['t3'].strip('> ')):
                review_needed = True
                values.append(f"T3: {self.t3}")
                
            if self.t4 and self.t4 > float(thyroid_ranges['review_needed']['t4'].strip('> ')):
                review_needed = True
                values.append(f"T4: {self.t4}")
            
            if review_needed:
                review_items.append({
                    "category": "metabolic_endocrine",
                    "type": "thyroid_function",
                    "value": ", ".join(values),
                    "ranges": thyroid_ranges,
                    "note": thyroid_ranges['note']
                })
        
        return review_items

    def get_analysis_context(self) -> Dict:
        """분석용 컨텍스트 생성"""
        health_keywords = CONFIG.get_health_keywords()
        health_metrics = CONFIG.get_health_metrics()
        reference_ranges = CONFIG.get_reference_ranges()
        
        # 위험 요인 분석
        risk_factors = self.get_risk_factors()
        
        # RAG 검색을 위한 컨텍스트 구성
        search_context = {
            "primary_concerns": [],     # 주요 건강 우려사항
            "related_metrics": [],      # 관련 건강 지표
            "lifestyle_factors": [],    # 생활습관 요인
            "interaction_risks": [],    # 상호작용 위험
            "search_keywords": set(),   # 검색 키워드
            "conditions": set()         # 건강 상태
        }
        
        # 위험 요인 기반 컨텍스트 구성
        for factor in risk_factors:
            category = factor["category"]
            search_context["primary_concerns"].append({
                "category": category,
                "type": factor["type"],
                "value": factor["value"],
                "severity": "review_needed"  # 검토 필요 수준
            })
            
            # 해당 카테고리의 키워드 및 관련 정보 추가
            for keyword in health_keywords:
                if keyword["category"] == category:
                    search_context["search_keywords"].update(keyword["search_terms"])
                    if "conditions" in keyword:
                        search_context["conditions"].update(keyword["conditions"])
                    if "lifestyle_factors" in keyword:
                        search_context["lifestyle_factors"].extend(keyword["lifestyle_factors"])
            
            # 관련 건강 지표 정보 추가
            for metric_name, metric_info in health_metrics.items():
                if metric_info["category"] == category:
                    search_context["related_metrics"].append({
                        "name": metric_name,
                        "display_name": metric_info["display_name"],
                        "warnings": metric_info.get("interaction_warnings", [])
                    })
                    if "interaction_warnings" in metric_info:
                        search_context["interaction_risks"].extend(metric_info["interaction_warnings"])
        
        # 현재 복용 중인 약물/보조제 관련 컨텍스트
        if self.current_medications or self.current_supplements:
            search_context["interaction_risks"].extend([
                f"medication:{med}" for med in (self.current_medications or [])
            ])
            search_context["interaction_risks"].extend([
                f"supplement:{supp}" for supp in (self.current_supplements or [])
            ])
        
        # 기존 건강 상태 관련 컨텍스트
        if self.health_conditions:
            search_context["conditions"].update(self.health_conditions)
        
        # 생활습관 관련 컨텍스트
        lifestyle_context = {}
        if self.smoking is not None:
            lifestyle_context["smoking"] = "current" if self.smoking else "non_smoker"
        if self.alcohol is not None:
            lifestyle_context["alcohol"] = "drinker" if self.alcohol else "non_drinker"
        if self.exercise_frequency is not None:
            lifestyle_context["exercise"] = {
                "frequency": self.exercise_frequency,
                "level": "inactive" if self.exercise_frequency < 3 else "active"
            }
        search_context["lifestyle_factors"].append(lifestyle_context)
        
        return {
            "basic_info": {
                "age": self.age,
                "gender": self.gender,
                "bmi": self.bmi
            },
            "metrics": {
                "blood_pressure": {
                    "systolic": self.systolic_bp,
                    "diastolic": self.diastolic_bp,
                    "reference": reference_ranges["ranges"]["blood_pressure"],
                    "source": reference_ranges["sources"]["blood_pressure"]
                },
                "blood_sugar": {
                    "fasting": self.fasting_blood_sugar,
                    "reference": reference_ranges["ranges"]["blood_sugar"],
                    "source": reference_ranges["sources"]["diabetes"]
                },
                "cholesterol": {
                    "total": self.total_cholesterol,
                    "hdl": self.hdl_cholesterol,
                    "ldl": self.ldl_cholesterol,
                    "triglyceride": self.triglyceride,
                    "reference": reference_ranges["ranges"]["cholesterol"],
                    "source": reference_ranges["sources"]["lipid"]
                },
                "liver_function": {
                    "ast": self.sgotast,
                    "alt": self.sgptalt,
                    "ggt": self.gammagtp,
                    "reference": reference_ranges["ranges"]["liver_function"],
                    "source": reference_ranges["sources"]["liver"]
                },
                "kidney_function": {
                    "creatinine": self.creatinine,
                    "gfr": self.gfr,
                    "reference": reference_ranges["ranges"]["kidney_function"],
                    "source": reference_ranges["sources"]["kidney"]
                },
                "immune_function": {
                    "wbc": self.wbc,
                    "neutrophil": self.neutrophil,
                    "lymphocyte": self.lymphocyte,
                    "crp": self.crp,
                    "reference": reference_ranges["ranges"]["immune_function"],
                    "source": reference_ranges["sources"]["immune"]
                },
                "thyroid_function": {
                    "tsh": self.tsh,
                    "t3": self.t3,
                    "t4": self.t4,
                    "reference": reference_ranges["ranges"]["thyroid_function"],
                    "source": reference_ranges["sources"]["thyroid"]
                }
            },
            "risk_factors": risk_factors,
            "current_medications": self.current_medications,
            "current_supplements": self.current_supplements,
            "health_conditions": self.health_conditions,
            "lifestyle": {
                "smoking": self.smoking,
                "alcohol": self.alcohol,
                "exercise_frequency": self.exercise_frequency,
                "reference": reference_ranges["ranges"]["lifestyle"]
            },
            "rag_context": {
                "search_keywords": list(search_context["search_keywords"]),
                "primary_concerns": search_context["primary_concerns"],
                "related_metrics": search_context["related_metrics"],
                "lifestyle_factors": list(set(search_context["lifestyle_factors"])),
                "interaction_risks": list(set(search_context["interaction_risks"])),
                "conditions": list(search_context["conditions"]),
                "interaction_check_needed": bool(self.current_medications or self.current_supplements)
            }
        }
