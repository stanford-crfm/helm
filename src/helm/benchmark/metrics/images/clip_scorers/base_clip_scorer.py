from abc import abstractmethod, ABC
from typing import List


class BaseCLIPScorer(ABC):
    @abstractmethod
    def compute_score(self, caption: str, image_location: str) -> float:
        pass

    def select_best_image(self, caption: str, image_locations: List[str]) -> str:
        """Selects the image from a list of images with the highest CLIPScore given the caption."""
        assert len(image_locations) > 0, "Need at least one image"

        if len(image_locations) == 1:
            return image_locations[0]

        scores: List[float] = [self.compute_score(caption, image_location) for image_location in image_locations]
        return image_locations[scores.index(max(scores))]
