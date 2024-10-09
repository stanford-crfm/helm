from typing import Dict, Optional
from dataclasses import asdict

from helm.common.cache import Cache, CacheConfig
from helm.common.clip_score_request import DEFAULT_CLIP_SCORE_MODEL, CLIPScoreRequest, CLIPScoreResult
from helm.clients.clip_scorers.base_clip_scorer import BaseCLIPScorer


class CLIPScoreClientError(Exception):
    pass


class CLIPScoreClient:
    def __init__(self, cache_config: CacheConfig):
        self.cache = Cache(cache_config)
        self._clip_scorer: Optional[BaseCLIPScorer] = None

    def compute_score(self, request: CLIPScoreRequest) -> CLIPScoreResult:
        """
        Compute a CLIPScore for a given caption and image.
        """
        # TODO: support multilingual CLIPScore and other CLIP models.
        assert request.model == DEFAULT_CLIP_SCORE_MODEL, f"Unsupported model: {request.model}"
        assert not request.multilingual

        try:

            def do_it():
                if self._clip_scorer is None:
                    from helm.clients.clip_scorers.clip_scorer import CLIPScorer

                    self._clip_scorer = CLIPScorer()

                score: float = self._clip_scorer.compute_score(
                    caption=request.caption, image_location=request.image_location
                )
                return {"score": score}

            cache_key: Dict = asdict(request)
            results, cached = self.cache.get(cache_key, do_it)

        except Exception as e:
            raise CLIPScoreClientError(e)

        return CLIPScoreResult(
            success=True,
            cached=cached,
            score=results["score"],
        )
