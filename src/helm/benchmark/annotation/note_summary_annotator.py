import evaluation_instruments.instruments.pdsqi_9.pdsqi_prompt as pdsqi

from typing import Dict, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo, LLMAsJuryAnnotator
from helm.clients.auto_client import AutoClient

from evaluation_instruments import prep


class NoteSummaryAnnotator(LLMAsJuryAnnotator):
    """The NoteSummary autograder."""

    name = "note_summary"

    def __init__(
        self,
        auto_client: AutoClient,
        annotator_models: Dict[str, AnnotatorModelInfo],
        template_name: Optional[str] = None,
    ):
        super().__init__(
            auto_client=auto_client,
            prompt_template="",
            annotation_criteria={},
            annotator_models=annotator_models,
        )

    def _interpolate_prompt(
        self, request_state: RequestState, custom_replacements: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Interpolate prompt template with request state information.

        :param request_state: The current request state
        :param custom_replacements: Optional dictionary of additional replacements
        :return: Interpolated prompt
        """
        notes = (request_state.instance.extra_data or {}).get("notes", [])
        prompt = pdsqi.resolve_prompt(
            summary_to_evaluate=(
                request_state.result.completions[0].text
                if request_state.result and request_state.result.completions
                else ""
            ),
            notes=notes,
            target_specialty="emergency medicine",
            output_mode=prep.OutputMode.EXPLAINED_SCORE,
        )
        return prompt[1]["content"]
