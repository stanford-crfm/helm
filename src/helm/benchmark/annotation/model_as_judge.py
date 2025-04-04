import json
import re
from dataclasses import dataclass
from typing import Dict, Optional, TypedDict, Union, Callable, Any, Set

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.hierarchical_logger import hlog
from helm.common.request import Request


class AnnotatorResponseParseFailure(Exception):
    def __init__(self, response_text: str, **kwargs):
        self.response_text = response_text
        super().__init__(kwargs)


@dataclass
class AnnotatorModelInfo:
    model_name: str
    model_deployment: str


def score_with_reasoning_with_gpt_and_llama(
    auto_client: AutoClient,
    annotator_prompt: str,
) -> Dict[str, Optional[Union[str, float]]]:
    """EXPERIMENTAL: DO NOT USE IN PRODUCTION

    Score using GPT-4o and Llama 3.1 for safety scenarios in HELM Safety."""
    # TODO: Make this configurable
    SHORT_NAME_TO_MODEL_INFO: Dict[str, AnnotatorModelInfo] = {
        "gpt": AnnotatorModelInfo(model_name="openai/gpt-4o-2024-05-13", model_deployment="openai/gpt-4o-2024-05-13"),
        "llama": AnnotatorModelInfo(
            model_name="meta/llama-3.1-405b-instruct-turbo", model_deployment="together/llama-3.1-405b-instruct-turbo"
        ),
    }
    result: Dict[str, Optional[Union[str, float]]] = {"prompt_text": annotator_prompt}
    for short_name, model_info in SHORT_NAME_TO_MODEL_INFO.items():
        try:
            score_and_reasoning = score_with_reasoning(
                auto_client,
                annotator_prompt,
                annotator_model=model_info.model_name,
                annotator_model_deployment=model_info.model_deployment,
            )
            result[f"{short_name}_score"] = score_and_reasoning["score"]
            result[f"{short_name}_reasoning"] = score_and_reasoning["reasoning"]
        except AnnotatorResponseParseFailure as e:
            result[f"{short_name}_score"] = None
            result[f"{short_name}_reasoning"] = None
            result[f"{short_name}_response_text"] = e.response_text
    return result


class ScoreAndReasoning(TypedDict):
    score: float
    reasoning: str


def score_with_reasoning(
    auto_client: AutoClient,
    annotator_prompt: str,
    annotator_model: str,
    annotator_model_deployment: str,
) -> ScoreAndReasoning:
    annotator_request = Request(
        model=annotator_model,
        model_deployment=annotator_model_deployment,
        prompt=annotator_prompt,
        temperature=0.0,
        max_tokens=256,
    )
    annotator_response = auto_client.make_request(annotator_request)
    if not annotator_response.success:
        raise Exception(f"Annotation request failed: {annotator_response.error}")
    assert len(annotator_response.completions) == 1
    annotator_response_text = annotator_response.completions[0].text

    # fuzzy match regex check, allows for different casing, or forgetting / in end tag
    reasoning_match = re.search(
        r"<\s*reasoning\s*>(.*?)<\/?\s*reasoning\s*>", annotator_response_text, re.DOTALL | re.IGNORECASE
    )
    score_match = re.search(r"<\s*score\s*>(.*?)<\/?\s*score\s*>", annotator_response_text, re.DOTALL | re.IGNORECASE)
    if not reasoning_match or not score_match:
        raise AnnotatorResponseParseFailure(
            message=f"Could not parse markup in raw response: '{annotator_response_text}'",
            response_text=annotator_response_text,
        )
    reasoning = reasoning_match.group(1).strip()
    try:
        score = float(score_match.group(1).strip())
    except ValueError:
        raise AnnotatorResponseParseFailure(
            message=f"Could not parse score as float from raw request: '{annotator_response_text}'",
            response_text=annotator_response_text,
        )

    return {"reasoning": reasoning, "score": score}


class LLMAsJuryAnnotator(Annotator):
    """
    A flexible LLM-based annotator that can be configured for different annotation scenarios.

    This annotator supports:
    - Custom prompt templates
    - Multiple evaluation models
    - Configurable evaluation criteria
    - Robust error handling
    """

    def __init__(
        self,
        auto_client: AutoClient,
        prompt_template: str,
        annotation_criteria: Dict[str, Set[str]],
        annotator_models: Dict[str, AnnotatorModelInfo],
        preprocessor: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the LLMAsJuryAnnotator.

        :param auto_client: Client for making API requests
        :param prompt_template: Template for generating prompts
        :param annotation_criteria: Dictionary defining expected annotation structure
        :param annotator_models: Dictionary of models to use for annotation
        :param preprocessor: Optional function to preprocess model responses
        """
        self._auto_client = auto_client
        self._prompt_template = prompt_template
        self._annotation_criteria = annotation_criteria
        self._annotator_models = annotator_models
        self._preprocessor = preprocessor or self._sanitize_model_response

    def _sanitize_model_response(self, model_response: str) -> str:
        """
        Sanitize the model response to extract JSON.

        :param model_response: Raw model response
        :return: Extracted JSON string
        """
        json_match = re.search(r"\{.*\}", model_response, re.DOTALL)
        return json_match.group(0) if json_match else model_response

    def _interpolate_prompt(
        self, request_state: RequestState, custom_replacements: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Interpolate prompt template with request state information.

        :param request_state: The current request state
        :param custom_replacements: Optional dictionary of additional replacements
        :return: Interpolated prompt
        """
        base_replacements = {
            "{{QUESTION}}": request_state.instance.input.text,
            "{{RESPONSE}}": (
                request_state.result.completions[0].text
                if request_state.result and request_state.result.completions
                else ""
            ),
            "{{GOLD_RESPONSE}}": request_state.instance.references[0].output.text,
        }

        # Allow custom replacements to override base replacements
        if custom_replacements:
            base_replacements.update(custom_replacements)

        prompt = self._prompt_template
        for key, value in base_replacements.items():
            prompt = prompt.replace(key, str(value))

        return prompt

    def _validate_annotation(self, annotator_criteria: Dict[str, Any], annotator_name: str) -> bool:
        """
        Validate the annotation meets expected criteria.

        :param annotator_criteria: Annotation dictionary to validate
        :param annotator_name: Name of the annotator model
        :return: Whether the annotation is valid
        """
        for key, value in self._annotation_criteria.items():
            if key not in annotator_criteria:
                hlog(
                    f"WARNING: Annotator did not find the expected key "
                    f"'{key}' in the response from {annotator_name}."
                )
                return False

            for subkey in value:
                if subkey not in annotator_criteria[key]:
                    hlog(
                        f"WARNING: Annotator did not find the expected subkey "
                        f"'{subkey}' in the response from {annotator_name}."
                    )
                    return False
        return True

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        """
        Annotate the request state using multiple LLM models.

        :param request_state: The request state to annotate
        :return: Dictionary of annotations from different models
        """
        assert request_state.result
        assert len(request_state.result.completions) == 1

        # Check for empty model output
        model_output_text = request_state.result.completions[0].text
        if not model_output_text.strip():
            hlog("WARNING: Annotator skipped sending requests because the model response was empty")
            return {
                "prompt_text": None,
                "empty_output_equivalence_judgement": False,
            }

        # Prepare prompt
        annotator_prompt = self._interpolate_prompt(request_state)
        annotations: Dict[str, Union[Optional[str], Optional[bool], Dict[str, Any]]] = {"prompt_text": annotator_prompt}

        # Track failed annotations for each model
        failed_counts: Dict[str, int] = {name: 0 for name in self._annotator_models}

        # Annotate using multiple models
        for annotator_name, annotator_model_info in self._annotator_models.items():
            try:
                annotator_criteria = self._annotate_with_model(annotator_prompt, annotator_model_info, annotator_name)

                if annotator_criteria is not None:
                    annotations[annotator_name] = annotator_criteria
                else:
                    failed_counts[annotator_name] += 1

            except Exception as e:
                hlog(f"ERROR annotating with {annotator_name}: {e}")
                failed_counts[annotator_name] += 1

        hlog(f"Failed model annotations: {failed_counts}")
        return annotations

    def _annotate_with_model(
        self, prompt: str, model_info: AnnotatorModelInfo, annotator_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Annotate using a specific model with enhanced JSON parsing.

        :param prompt: Interpolated prompt
        :param model_info: Model information
        :param annotator_name: Name of the annotator
        :return: Annotation criteria or None if failed
        """
        annotator_request = Request(
            model=model_info.model_name,
            model_deployment=model_info.model_deployment,
            prompt=prompt,
            temperature=0.0,
            max_tokens=4096,
        )

        annotator_response = self._auto_client.make_request(annotator_request)

        if not annotator_response.success:
            hlog(f"WARNING: Got an error response from {model_info.model_name}: " f"{annotator_response.error}")
            return None

        try:
            annotator_output = annotator_response.completions[0].text
            annotator_output = self._preprocessor(annotator_output)

            try:
                annotator_criteria = json.loads(annotator_output)
            except json.JSONDecodeError as e:
                if e.msg == "Expecting ',' delimiter":
                    # Attempt to fix incomplete JSON by adding a closing brace
                    annotator_output = annotator_output + "}"
                    try:
                        annotator_criteria = json.loads(annotator_output)
                    except Exception as ex:
                        hlog(
                            f"WARNING: Error parsing response from {model_info.model_name} "
                            f"after adding closing brace: {ex}. "
                            f"Model output: {annotator_output}"
                        )
                        return None
                else:
                    # For other JSON decoding errors
                    hlog(
                        f"WARNING: JSON decoding error from {model_info.model_name}: {e}. "
                        f"Model output: {annotator_output}"
                    )
                    return None

            # Validate annotation structure
            if not self._validate_annotation(annotator_criteria, annotator_name):
                return None

            return annotator_criteria

        except Exception as e:
            hlog(
                f"WARNING: Unexpected error processing response from {model_info.model_name}: {e}. "
                f"Model output: {annotator_output}"
            )
            return None
