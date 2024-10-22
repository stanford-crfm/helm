from typing import Any, Dict, List, Optional
import base64

from helm.common.cache import CacheConfig, Cache
from helm.common.general import hlog
from helm.common.file_caches.file_cache import FileCache
from helm.common.media_object import MultimediaObject
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request, RequestResult, GeneratedOutput, wrap_request_time
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from helm.clients.moderation_api_client import ModerationAPIClient
from helm.clients.client import Client, CachingClient
from helm.clients.image_generation.image_generation_client_utils import get_single_image_multimedia_object

try:
    import openai
    from openai import OpenAI
except ModuleNotFoundError as missing_module_exception:
    handle_module_not_found_error(missing_module_exception, ["openai"])


class DALLE2Client(Client):
    MAX_PROMPT_LENGTH: int = 1000
    DEFAULT_IMAGE_SIZE_STR: str = "512x512"
    VALID_IMAGE_SIZES: List[str] = ["256x256", DEFAULT_IMAGE_SIZE_STR, "1024x1024"]

    # Set the finish reason to this if the prompt violates OpenAI's content policy
    CONTENT_POLICY_VIOLATED_FINISH_REASON: str = (
        "The prompt violates OpenAI's content policy. "
        "See https://labs.openai.com/policies/content-policy for more information."
    )

    # The DALL-E API will respond with the following error messages (or even a substring of the message)
    # if it has any issues generating images for a particular prompt
    PROMPT_FLAGGED_ERROR: str = (
        "Your request was rejected as a result of our safety system. "
        "Your prompt may contain text that is not allowed by our safety system."
    )
    PROMPT_FLAGGED_ERROR2: str = (
        "Something went wrong with your generation. You may try again or ask for a different prompt"
    )
    PROMPT_FLAGGED_ERROR3: str = (
        "The server had an error while processing your request. Sorry about that! You can retry your request, "
        "or contact us through our help center at help.openai.com if the error persists."
    )

    def __init__(
        self,
        api_key: str,
        cache_config: CacheConfig,
        file_cache: FileCache,
        moderation_api_client: ModerationAPIClient,
        org_id: Optional[str] = None,
    ):
        self.file_cache: FileCache = file_cache
        self._cache = Cache(cache_config)

        self.client = OpenAI(api_key=api_key, organization=org_id)
        self.moderation_api_client: ModerationAPIClient = moderation_api_client

    def get_content_policy_violated_result(self, request: Request) -> RequestResult:
        """
        Return a RequestResult with no images and a finish reason indicating that the prompt / generated images
        violate OpenAI's content policy.
        """
        no_image = GeneratedOutput(
            text="",
            logprob=0,
            tokens=[],
            multimodal_content=MultimediaObject(),
            finish_reason={"reason": self.CONTENT_POLICY_VIOLATED_FINISH_REASON},
        )
        return RequestResult(
            success=True,
            cached=False,
            request_time=0,
            completions=[no_image] * request.num_completions,
            embedding=[],
        )

    def get_size_str(self, request: Request) -> str:
        """
        Return the size string for the image generation request.
        If the request does not specify a size, return the default size.
        """
        assert request.image_generation_parameters is not None
        w: Optional[int] = request.image_generation_parameters.output_image_width
        h: Optional[int] = request.image_generation_parameters.output_image_height
        if w is None or h is None:
            return self.DEFAULT_IMAGE_SIZE_STR

        image_dimensions: str = f"{w}x{h}"
        assert image_dimensions in self.VALID_IMAGE_SIZES, f"Valid image sizes are {self.VALID_IMAGE_SIZES}"
        return image_dimensions

    def fail_if_invalid_request(self, request: Request) -> None:
        """
        Validate the request to ensure it is a valid request for the DALL-E API.
        """
        assert request.image_generation_parameters is not None
        if len(request.prompt) > self.MAX_PROMPT_LENGTH:
            raise ValueError("The maximum length of the prompt is 1000 characters.")
        if request.num_completions < 1 or request.num_completions > 10:
            raise ValueError("`num_completions` must be between 1 and 10.")

    def handle_openai_error(self, request: Request, error: Exception) -> RequestResult:
        """
        Handle a thrown error from the DALL-E API.
        """
        if (
            str(error) in self.PROMPT_FLAGGED_ERROR
            # Sometimes the DALL-E API will add additional information to the error message.
            or self.PROMPT_FLAGGED_ERROR2 in str(error)
            or self.PROMPT_FLAGGED_ERROR3 in str(error)
        ):
            # Some requests fail even if we check the prompt against the moderation API.
            # For example, "black" in Spanish (negro) causes requests to DALL-E to fail even
            # though the prompt does not get flagged by the Moderation API.
            hlog(f"Failed safety check: {request.prompt}")
            return self.get_content_policy_violated_result(request)
        else:
            return RequestResult(
                success=False, cached=False, error=f"DALL-E error: {error}", completions=[], embedding=[]
            )

    def generate_with_dalle_api(self, raw_request: Dict[str, Any]) -> Dict:
        """
        Makes a single request to generate the images with the DALL-E API.
        """
        result = self.client.images.generate(**raw_request).model_dump(mode="json")
        assert "data" in result, f"Invalid response: {result} from prompt: {raw_request['prompt']}"

        for image in result["data"]:
            # Write out the image to a file and save the path
            image["file_path"] = self.file_cache.store(lambda: base64.b64decode(image["b64_json"]))
            # Don't cache contents of `b64_json` as we already have the image stored
            image.pop("b64_json", None)
        return result

    def make_request(self, request: Request) -> RequestResult:
        self.fail_if_invalid_request(request)

        # Use the Moderation API to check if the prompt violates OpenAI's content policy before generating images
        if self.moderation_api_client.will_be_flagged(request.prompt):
            return self.get_content_policy_violated_result(request)

        # https://beta.openai.com/docs/api-reference/images/create#images/create-response_format
        raw_request: Dict[str, Any] = {
            "prompt": request.prompt,
            "n": request.num_completions,
            "size": self.get_size_str(request),
            "response_format": "b64_json",  # Always set to b64_json as URLs are only valid for an hour
        }

        try:

            def do_it() -> Dict[str, Any]:
                # To maintain backwards compatibility, specify the model in the request but not in the cache key
                return self.generate_with_dalle_api({"model": "dall-e-2", **raw_request})

            cache_key = CachingClient.make_cache_key(raw_request, request)
            response, cached = self._cache.get(cache_key, wrap_request_time(do_it))
        except openai.OpenAIError as e:
            return self.handle_openai_error(request, e)

        completions: List[GeneratedOutput] = [
            GeneratedOutput(
                text="",
                logprob=0,
                tokens=[],
                multimodal_content=get_single_image_multimedia_object(generated_image["file_path"]),
            )
            for generated_image in response["data"]
        ]
        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("This client does not support tokenizing.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("This client does not support decoding.")
