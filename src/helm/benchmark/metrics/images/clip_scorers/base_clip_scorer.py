from abc import abstractmethod, ABC


class BaseCLIPScorer(ABC):
    @abstractmethod
    def compute_score(self, caption: str, image_location: str) -> float:
        pass
