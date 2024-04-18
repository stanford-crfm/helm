# mypy: check_untyped_defs = False
from re import search, Match
from typing import Any, Dict, List, Optional, cast, Union

from helm.benchmark.model_metadata_registry import is_vlm
from helm.common.media_object import TEXT_TYPE
from helm.common.request import GeneratedOutput, Token, Request
from helm.common.hierarchical_logger import hlog
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.gpt4v_originality_request import (
    wrap_request_time,
    GPT4VOriginalityRequestResult,
    GPT4VScoreOutput,
)
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
)
from .client import truncate_sequence
from .openai_client import OpenAIClient

try:
    import openai
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["openai"])


class GPT4VOriginalityClient(OpenAIClient):
    # TODO: Design a more structured prompt for evaluation. We need to explain each level of the
    # TODO: originality score in one short sentence.
    EVALUATION_PROMPT_TEMPLATE = (
        "Please rate the generated texts given the image from 1 to 5. Try your best "
        "to give only the rating without other explanations.\n{GENERATED_TEXT}"
    )
    REX_PATTERN = r"[1-5]"

    def _is_gpt4v_model_engine(self, model_engine: str) -> bool:
        if model_engine.startswith("gpt-4-vision"):
            return True
        return False

    def _reformat_text_request(self, original_text_input: str) -> str:
        return self.EVALUATION_PROMPT_TEMPLATE.format(GENERATED_TEXT=original_text_input)

    def _convert_completion_to_originality_score(self, completions: GeneratedOutput) -> GPT4VScoreOutput:
        # TODO: We might consider improving the extraction process of GPT4V generated score here.
        new_text: str = completions.text
        match_seq: Optional[Match[str]] = search(self.REX_PATTERN, new_text)
        if match_seq:
            new_score: float = float(match_seq.group())
        else:
            raise ValueError(f"Could not find a score in the completion text: {new_text}")
        gpt4vscore = GPT4VScoreOutput(score=new_score, logprob=completions.logprob, tokens=completions.tokens)
        return gpt4vscore

    def _make_scoring_request(self, request: Request) -> GPT4VOriginalityRequestResult:
        messages: Optional[List[Dict[str, Union[str, Any]]]] = request.messages
        # Only support multimodal_prompt as the input for now
        # Convert prompt into a single message
        # For now, put the whole prompt in a single user message, and expect the response
        # to be returned in a single assistant message.
        # TODO: Support ChatML for creating multiple messages with different roles.
        # See: https://github.com/openai/openai-python/blob/main/chatml.md

        # Content can either be text or a list of multimodal content made up of text and images:
        # https://platform.openai.com/docs/guides/vision
        content: Union[str, List[Union[str, Any]]]
        if request.multimodal_prompt is not None:
            content = []
            for media_object in request.multimodal_prompt.media_objects:
                if media_object.is_type("image") and media_object.location:
                    from helm.common.images_utils import encode_base64

                    base64_image: str = encode_base64(media_object.location)
                    content.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    )
                elif media_object.is_type(TEXT_TYPE):
                    if media_object.text is None:
                        raise ValueError("MediaObject of text type has missing text field value")
                    text_input: str = self._reformat_text_request(media_object.text)
                    content.append({"type": media_object.type, "text": text_input})
                else:
                    raise ValueError(f"Unrecognized MediaObject type {media_object.type}")

        else:
            raise ValueError("Input request has missing multimodal prompt value")

        messages = [{"role": "user", "content": content}]

        # Fixing the most generation parameters here.
        raw_request: Dict[str, Any] = {
            "model": self._get_model_for_request(request),
            "messages": messages,
            "temperature": 1.0,
            "top_p": 1.0,
            "n": 1,
            "stop": None,
            # Note: Chat models may require adding an extra token to max_tokens
            # for the internal special role token.
            "max_tokens": 15,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }

        # OpenAI's vision API doesn't allow None values for stop.
        # Fails with "body -> stop: none is not an allowed value" if None is passed.
        if is_vlm(request.model) and raw_request["stop"] is None:
            raw_request.pop("stop")

        def do_it() -> Dict[str, Any]:
            return self.client.chat.completions.create(**raw_request).model_dump(mode="json")

        try:
            cache_key = self._get_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except openai.OpenAIError as e:
            if self.INAPPROPRIATE_IMAGE_ERROR in str(e):
                hlog(f"Failed safety check: {str(request)}")
                empty_completion = GeneratedOutput(
                    text="",
                    logprob=0,
                    tokens=[],
                    finish_reason={"reason": self.CONTENT_POLICY_VIOLATED_FINISH_REASON},
                )
                empty_score = GPT4VScoreOutput(
                    score=0.0,
                    logprob=0,
                    tokens=[],
                    finish_reason={"reason": self.CONTENT_POLICY_VIOLATED_FINISH_REASON},
                )
                return GPT4VOriginalityRequestResult(
                    success=True,
                    cached=False,
                    request_time=0,
                    completions=[empty_completion] * request.num_completions,
                    scores=[empty_score] * request.num_completions,
                )

            error: str = f"OpenAI error: {e}"
            return GPT4VOriginalityRequestResult(success=False, cached=False, error=error, completions=[], scores=[])

        scores: List[GPT4VScoreOutput] = []
        completions: List[GeneratedOutput] = []
        for raw_completion in response["choices"]:
            # The OpenAI chat completion API doesn't support echo.
            # If `echo_prompt` is true, combine the prompt and completion.
            raw_completion_content = raw_completion["message"]["content"]
            text: str = request.prompt + raw_completion_content if request.echo_prompt else raw_completion_content
            # The OpenAI chat completion API doesn't return us tokens or logprobs, so we tokenize ourselves.
            tokenization_result: TokenizationRequestResult = self.tokenizer.tokenize(
                TokenizationRequest(text, tokenizer=self.tokenizer_name)
            )
            # Log probs are not currently not supported by the OpenAI chat completion API, so set to 0 for now.
            tokens: List[Token] = [
                Token(text=cast(str, raw_token), logprob=0) for raw_token in tokenization_result.raw_tokens
            ]
            completion = GeneratedOutput(
                text=text,
                logprob=0,  # OpenAI does not provide logprobs
                tokens=tokens,
                finish_reason={"reason": raw_completion["finish_reason"]},
            )
            # Truncate the text by stop sequences
            truncated_completion: GeneratedOutput = truncate_sequence(completion, request)
            completions.append(truncated_completion)
            # Convert the completion to originality score output
            scores.append(self._convert_completion_to_originality_score(truncated_completion))

        return GPT4VOriginalityRequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            scores=scores,
            completions=completions,
        )

    def get_originality_scores(self, request: Request) -> GPT4VOriginalityRequestResult:
        """
        Compute the originality score of a pair of given text and image using
        OpenAI GPT models.
        Returns a value from 1 to 5.
        """
        # We currently only support GPT4V evaluation.
        assert self._is_gpt4v_model_engine(
            request.model_engine
        ), f"Expect the model to be the GPT4V model engine, but got {request.model_engine}."
        return self._make_scoring_request(request)
