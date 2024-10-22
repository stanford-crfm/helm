from typing import Any, Dict, List, Optional

from helm.common.cache import CacheConfig
from helm.common.file_caches.file_cache import FileCache
from helm.common.general import singleton
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request, RequestResult, GeneratedOutput, wrap_request_time
from helm.clients.moderation_api_client import ModerationAPIClient
from helm.clients.client import CachingClient
from helm.clients.image_generation.dalle2_client import DALLE2Client
from helm.clients.image_generation.image_generation_client_utils import get_single_image_multimedia_object

try:
    import openai
except ModuleNotFoundError as missing_module_exception:
    handle_module_not_found_error(missing_module_exception, ["openai"])


class DALLE3Client(DALLE2Client):
    """
    Client for the OpenAI's DALL-E 3 API.
    DALL-E 3 cookbook with explanations for the different parameters:
    https://cookbook.openai.com/articles/what_is_new_with_dalle_3
    """

    DEFAULT_IMAGE_SIZE_STR: str = "1024x1024"
    VALID_IMAGE_SIZES: List[str] = [DEFAULT_IMAGE_SIZE_STR, "1792x1024", "1024x1792"]

    def __init__(
        self,
        api_key: str,
        cache_config: CacheConfig,
        file_cache: FileCache,
        moderation_api_client: ModerationAPIClient,
        org_id: Optional[str] = None,
    ):
        super().__init__(api_key, cache_config, file_cache, moderation_api_client, org_id)

    def make_request(self, request: Request) -> RequestResult:
        self.fail_if_invalid_request(request)
        if self.moderation_api_client.will_be_flagged(request.prompt):
            return self.get_content_policy_violated_result(request)

        raw_request: Dict[str, Any] = {
            "model": "dall-e-3",
            "prompt": request.prompt,
            "n": 1,  # As of December 2023, the DALL-E 3 API only supports a single generated image per request
            "size": self.get_size_str(request),
            "response_format": "b64_json",  # Always set to b64_json as URLs are only valid for an hour
        }

        if request.model_engine == "dall-e-3":
            raw_request["quality"] = "standard"
            raw_request["style"] = "vivid"
        elif request.model_engine == "dall-e-3-natural":
            raw_request["quality"] = "standard"
            raw_request["style"] = "natural"
        elif request.model_engine == "dall-e-3-hd":
            raw_request["quality"] = "hd"
            raw_request["style"] = "vivid"
        elif request.model_engine == "dall-e-3-hd-natural":
            raw_request["quality"] = "hd"
            raw_request["style"] = "natural"
        else:
            raise ValueError(f"Invalid DALL-E 3 model: {request.model_engine}")

        responses: List[Dict[str, Any]] = []
        all_cached: bool = True

        # Since the DALL-E 3 API only supports a single generated image, make `request.num_completions` requests
        for completion_index in range(request.num_completions):
            try:

                def do_it() -> Dict[str, Any]:
                    return self.generate_with_dalle_api({**raw_request})

                cache_key = CachingClient.make_cache_key({"completion_index": completion_index, **raw_request}, request)
                response, cached = self._cache.get(cache_key, wrap_request_time(do_it))

                responses.append(response)
                all_cached = all_cached and cached
            except openai.OpenAIError as e:
                return self.handle_openai_error(request, e)

        completions: List[GeneratedOutput] = []
        total_request_time: float = 0
        for response in responses:
            image_response: Dict[str, Any] = singleton(response["data"])
            completions.append(
                GeneratedOutput(
                    # From https://cookbook.openai.com/articles/what_is_new_with_dalle_3,
                    # "a new feature in the latest DALL·E-3 API is prompt rewriting, where we use
                    # GPT-4 to optimize all of your prompts before they’re passed to DALL-E."
                    text=image_response["revised_prompt"],
                    multimodal_content=get_single_image_multimedia_object(image_response["file_path"]),
                    logprob=0,
                    tokens=[],
                )
            )
            total_request_time += response["request_time"]

        return RequestResult(
            success=True,
            cached=all_cached,
            request_time=total_request_time,
            completions=completions,
            embedding=[],
        )
