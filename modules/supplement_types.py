from enum import Enum
from typing import Optional
from dataclasses import dataclass

class EvidenceLevel(Enum):
    HIGH = "A"    # RCT, 메타분석
    MEDIUM = "B"  # 관찰연구, 코호트
    LOW = "C"     # 사례연구, 전문가 의견

@dataclass
class StudyMetrics:
    sample_size: int
    duration_weeks: Optional[int]
    p_value: Optional[float]
    confidence_interval: Optional[str]

@dataclass
class SupplementEffect:
    effect_type: str        # positive/negative
    description: str        # 효과 설명
    mechanism: Optional[str] # 작용 메커니즘
    dosage: Optional[str]   # 용량 정보
    confidence: float       # 신뢰도 