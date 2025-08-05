from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class PredictionResult:
    sample_id: int
    context: str
    original_line: str
    predicted_line: str
    pavg: float
    ptot: float
    is_correct: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)