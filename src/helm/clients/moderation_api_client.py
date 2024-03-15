from typing import Any, Dict

from helm.common.request import wrap_request_time
from helm.common.cache import Cache, CacheConfig
from helm.common.moderations_api_request import (
    ModerationCategoryScores,
    ModerationCategoryFlaggedResults,
    ModerationAPIRequest,
    ModerationAPIRequestResult,
)
from helm.common.optional_dependencies import handle_module_not_found_error


class ModerationAPIClient:
    """
    From https://beta.openai.com/docs/guides/moderation/overview, the moderation endpoint is a tool
    to check whether content complies with OpenAI's content policy. Developers can thus identify content
    that OpenAI's content policy prohibits and take action, for instance by filtering it.
    """

    # For descriptions of the models, see https://beta.openai.com/docs/api-reference/moderations/create
    LATEST_MODEL: str = "text-moderation-latest"
    STABLE_MODEL: str = "text-moderation-stable"

    # List of categories (https://beta.openai.com/docs/guides/moderation/overview)
    HATE: str = "hate"
    HATE_THREATENING: str = "hate/threatening"
    SELF_HARM: str = "self-harm"
    SEXUAL: str = "sexual"
    SEXUAL_MINORS: str = "sexual/minors"
    VIOLENCE: str = "violence"
    VIOLENCE_GRAPHIC: str = "violence/graphic"

    def __init__(self, api_key: str, cache_config: CacheConfig):
        self.cache = Cache(cache_config)
        try:
            from openai import OpenAI
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["openai"])
        # TODO: Add OpenAI organization.
        self.client = OpenAI(api_key=api_key)

    def get_moderation_results(self, request: ModerationAPIRequest) -> ModerationAPIRequestResult:
        """
        Sends a request to OpenAI's moderation endpoint.
        https://beta.openai.com/docs/api-reference/moderations/create
        """
        try:
            import openai
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["openai"])

        raw_request: Dict[str, str] = {
            "input": request.text,
            "model": self.LATEST_MODEL if request.use_latest_model else self.STABLE_MODEL,
        }

        try:

            def do_it() -> Dict[str, Any]:
                result = self.client.moderations.create(input=request.text).model_dump(mode="json")
                assert "results" in result and len(result["results"]) > 0, f"Invalid response: {result}"
                return result

            response, cached = self.cache.get(raw_request, wrap_request_time(do_it))
        except openai.OpenAIError as e:
            error: str = f"Moderation API error: {e}"
            return ModerationAPIRequestResult(
                success=False, cached=False, error=error, flagged=None, flagged_results=None, scores=None
            )

        moderation_result = response["results"][0]
        category_results: Dict[str, bool] = moderation_result["categories"]
        score_results: Dict[str, float] = moderation_result["category_scores"]

        flagged_results = ModerationCategoryFlaggedResults(
            hate_flagged=category_results[self.HATE],
            hate_threatening_flagged=category_results[self.HATE_THREATENING],
            self_harm_flagged=category_results[self.SELF_HARM],
            sexual_flagged=category_results[self.SEXUAL],
            sexual_minors_flagged=category_results[self.SEXUAL_MINORS],
            violence_flagged=category_results[self.VIOLENCE],
            violence_graphic_flagged=category_results[self.VIOLENCE_GRAPHIC],
        )
        scores = ModerationCategoryScores(
            hate_score=score_results[self.HATE],
            hate_threatening_score=score_results[self.HATE_THREATENING],
            self_harm_score=score_results[self.SELF_HARM],
            sexual_score=score_results[self.SEXUAL],
            sexual_minors_score=score_results[self.SEXUAL_MINORS],
            violence_score=score_results[self.VIOLENCE],
            violence_graphic_score=score_results[self.VIOLENCE_GRAPHIC],
        )
        return ModerationAPIRequestResult(
            success=True,
            cached=cached,
            flagged=moderation_result["flagged"],
            flagged_results=flagged_results,
            scores=scores,
        )

    def will_be_flagged(self, text: str) -> bool:
        """Returns True if the text is against OpenAI's content policy and will be flagged, False otherwise."""
        result: ModerationAPIRequestResult = self.get_moderation_results(
            # Use the latest model so the account does not get banned
            ModerationAPIRequest(text=text, use_latest_model=True)
        )
        assert result.flagged is not None
        return result.flagged
