from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel

class Evidence(BaseModel):
    pubmed_id: str
    title: str
    abstract: str
    url: str
    publication_date: str
    journal: str
    study_type: str
    strength: str  # 근거 수준 (strong, moderate, weak)

class HealthEffect(BaseModel):
    condition: str
    effect_type: str  # positive, negative, neutral
    strength: str
    mechanism: str
    evidence: List[Evidence]

class Interaction(BaseModel):
    target: str  # 상호작용 대상 (supplement, medication, condition)
    effect: str  # increase, decrease, inhibit, enhance
    severity: str  # high, medium, low
    mechanism: str
    evidence: List[Evidence]

class Supplement(BaseModel):
    name: str
    aliases: List[str]
    health_effects: Dict[str, HealthEffect]  # health_field -> effect
    interactions: List[Interaction]
    evidence: List[Evidence]
    created_at: datetime
    updated_at: datetime 