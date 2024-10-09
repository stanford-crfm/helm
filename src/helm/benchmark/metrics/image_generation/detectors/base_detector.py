from abc import abstractmethod, ABC
from typing import Any, Dict


class BaseDetector(ABC):
    @abstractmethod
    def compute_score(self, caption: str, image_location: str, references: Dict[str, Any]) -> float:
        pass
