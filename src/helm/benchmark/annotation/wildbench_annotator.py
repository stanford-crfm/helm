import re
from typing import Any, Optional, Union
from importlib.resources import files
from typing import Dict

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo
from helm.clients.auto_client import AutoClient
from helm.common.hierarchical_logger import hlog
from helm.common.request import Request


class WildBenchAnnotator(Annotator):
    """The WildBench autograder."""

    name = "wildbench"

    def __init__(self, auto_client: AutoClient):
        self._auto_client = auto_client
        template_path = files("helm.benchmark.annotation.wildbench").joinpath("eval_template.score.v2.md")
        with template_path.open("r") as f:
            self._score_template = f.read()
        self._pattern = re.compile(
            r'"strengths"\s*:\s*"(.*?)"\s*,\s*"weaknesses"\s*:\s*"(.*?)"\s*,\s*"score"\s*:\s*(".*?"|\d+)', re.DOTALL
        )

    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        assert request_state.instance.extra_data
        model_output_text = request_state.result.completions[0].text
        if not model_output_text.strip():
            # Following https://github.com/allenai/WildBench/blob/d6b8dcaf377d173d031980f97c16e1a82618c03d/src/eval.py
            hlog(
                "WARNING: WildBenchAnnotator skipped sending requests to annotator models "
                "because the model response was empty"
            )
            return {
                "prompt_text": None,
                "empty_output_score": 1.0,
            }

        input_messages = request_state.instance.input.messages
        assert input_messages is not None
        history = []
        for round in input_messages[:-1]:
            noun = "USER: " if round["role"] == "user" else "ASSISTANT: "
            history.append(noun + round["content"])
        history_text = "\n\n".join(history)
        user_query_text = input_messages[-1]["content"]
        checklist_text = "\n".join(
            [f"- {checklist_item}" for checklist_item in request_state.instance.extra_data["checklist"]]
        )
        annotator_prompt = (
            self._score_template.replace("{$history}", history_text)
            .replace("{$user_query}", user_query_text)
            .replace("{$model_output}", model_output_text)
            .replace("{$checklist}", checklist_text)
        )

        SHORT_NAME_TO_MODEL_INFO: Dict[str, AnnotatorModelInfo] = {
            "gpt": AnnotatorModelInfo(
                model_name="openai/gpt-4o-2024-05-13",
                model_deployment="openai/gpt-4o-2024-05-13",
            ),
            "llama": AnnotatorModelInfo(
                model_name="meta/llama-3.1-405b-instruct-turbo",
                model_deployment="together/llama-3.1-405b-instruct-turbo",
            ),
            "claude": AnnotatorModelInfo(
                model_name="anthropic/claude-3-5-sonnet-20241022",
                model_deployment="anthropic/claude-3-5-sonnet-20241022",
            ),
        }
        annotations: Dict[str, Union[Optional[str], Optional[float]]] = {"prompt_text": annotator_prompt}
        for annotator_name, annotator_model_info in SHORT_NAME_TO_MODEL_INFO.items():
            annotator_request = Request(
                model=annotator_model_info.model_name,
                model_deployment=annotator_model_info.model_deployment,
                prompt=annotator_prompt,
                temperature=0.0,
                max_tokens=2000,
            )
            strengths: Optional[str] = None
            weaknesses: Optional[str] = None
            score: Optional[float] = None
            annotator_response = self._auto_client.make_request(annotator_request)
            if not annotator_response.success:
                hlog(
                    "WARNING: WildBenchAnnotator got an error response from "
                    f"{annotator_model_info.model_name}: : {annotator_response.error}"
                )
            else:
                assert len(annotator_response.completions) == 1
                annotator_response_text = annotator_response.completions[0].text
                annotator_response_parts = self._pattern.search(annotator_response_text)
                if not annotator_response_parts:
                    hlog(
                        "WARNING: WildBenchAnnotator got a malformed annotation from "
                        f"{annotator_model_info.model_name}: {annotator_response_text}"
                    )
                else:
                    strengths = annotator_response_parts[1].strip()
                    weaknesses = annotator_response_parts[2].strip()
                    score_text = annotator_response_parts[3].strip().strip('"')
                    try:
                        score = float(score_text)
                    except ValueError:
                        hlog(
                            "WARNING: WildBenchAnnotator could not parse the score from the annotation from "
                            f"{annotator_model_info.model_name}: {annotator_response_text}"
                        )

            annotations[f"{annotator_name}_strengths"] = strengths
            annotations[f"{annotator_name}_weaknesses"] = weaknesses
            annotations[f"{annotator_name}_score"] = score

        return annotations
