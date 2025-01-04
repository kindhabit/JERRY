from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel
from .supplement import Evidence

class SupplementInteraction(BaseModel):
    supplements: List[str]
    effect_type: str  # synergistic, antagonistic, competitive
    severity: str    # high, medium, low
    mechanism: str
    evidence: List[Evidence]

class DrugInteraction(BaseModel):
    supplement: str
    drug: str
    drug_category: str
    effect_type: str
    severity: str
    mechanism: str
    contraindications: List[str]
    evidence: List[Evidence]

class HealthConditionInteraction(BaseModel):
    supplement: str
    condition: str
    effect_type: str
    risk_level: str
    precautions: List[str]
    evidence: List[Evidence]
