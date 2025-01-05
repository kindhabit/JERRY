from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import date

class BasicInfo(BaseModel):
    age: int
    gender: str
    height: float  # cm
    weight: float  # kg
    blood_type: Optional[str] = None

class VitalSigns(BaseModel):
    blood_pressure_systolic: int
    blood_pressure_diastolic: int
    heart_rate: int
    body_temperature: Optional[float] = None

class BloodTestResults(BaseModel):
    glucose_fasting: float
    total_cholesterol: float
    hdl_cholesterol: float
    ldl_cholesterol: float
    triglycerides: float
    hemoglobin: float
    hematocrit: float
    alt: float  # 간 기능
    ast: float  # 간 기능
    creatinine: float  # 신장 기능

class LifestyleFactors(BaseModel):
    smoking: bool = False
    alcohol_consumption: bool = False
    exercise_frequency: int = 0  # 주간 운동 횟수
    sleep_hours: Optional[float] = None
    stress_level: Optional[int] = None  # 1-5 척도

class MedicalHistory(BaseModel):
    chronic_conditions: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    family_history: List[str] = Field(default_factory=list)
    surgeries: List[str] = Field(default_factory=list)

class HealthData(BaseModel):
    basic_info: BasicInfo
    vital_signs: VitalSigns
    blood_test: BloodTestResults
    lifestyle: LifestyleFactors
    medical_history: MedicalHistory
    examination_date: date = Field(default_factory=date.today)
    
    class Config:
        schema_extra = {
            "example": {
                "basic_info": {
                    "age": 35,
                    "gender": "male",
                    "height": 175.0,
                    "weight": 70.0,
                    "blood_type": "A+"
                },
                "vital_signs": {
                    "blood_pressure_systolic": 120,
                    "blood_pressure_diastolic": 80,
                    "heart_rate": 72
                },
                "blood_test": {
                    "glucose_fasting": 95.0,
                    "total_cholesterol": 180.0,
                    "hdl_cholesterol": 50.0,
                    "ldl_cholesterol": 110.0,
                    "triglycerides": 150.0,
                    "hemoglobin": 14.0,
                    "hematocrit": 42.0,
                    "alt": 25.0,
                    "ast": 25.0,
                    "creatinine": 1.0
                },
                "lifestyle": {
                    "smoking": False,
                    "alcohol_consumption": True,
                    "exercise_frequency": 3,
                    "sleep_hours": 7.0,
                    "stress_level": 3
                },
                "medical_history": {
                    "chronic_conditions": ["hypertension"],
                    "medications": ["amlodipine"],
                    "allergies": [],
                    "family_history": ["diabetes"],
                    "surgeries": []
                }
            }
        }
