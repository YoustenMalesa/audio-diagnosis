from pydantic import BaseModel
from typing import List, Optional

CLASS_LABELS = [
    'normal',
    'asthma',
    'copd',
    'bronchitis',
    'heart failure',
    'lung fibrosis',
    'pleural effusion',
    'pneumonia'
]


class Prediction(BaseModel):
    label: str
    confidence: float
    severity: Optional[str] = None  # Low | Medium | High
    stage: Optional[str] = None     # Early | Progressed | Advanced
    prob_std: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    samples: Optional[int] = None


class PredictResponse(BaseModel):
    top1: Prediction
    topk: List[Prediction]
    all: List[Prediction]
    model_version: str
