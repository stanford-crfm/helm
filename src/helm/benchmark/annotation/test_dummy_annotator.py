from typing import Any
from dataclasses import replace
import pytest

from helm.benchmark.annotation.annotator import Annotator, DummyAnnotator
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Instance, Input
from helm.common.request import Request, RequestResult, GeneratedOutput


class TestDummyAnnotator:
    def setup_method(self):
        self.annotator: Annotator = DummyAnnotator()
        self.request_state = RequestState(
            instance=Instance(input=Input(text="hello world"), references=[]),
            request=Request(),
            request_mode="original",
            output_mapping=None,
            result=None,
            train_trial_index=0,
            num_train_instances=0,
            prompt_truncated=False,
            reference_index=None,
        )

    def test_annotate(self):
        request_state: RequestState = replace(
            self.request_state,
            result=RequestResult(
                success=True,
                embedding=[],
                completions=[GeneratedOutput(text="How are you?", logprob=0, tokens=[])],
                cached=True,
            ),
        )
        annotations: Any = self.annotator.annotate(request_state)
        assert annotations is not None
        assert "all_caps" in annotations[0]
        assert request_state.result is not None  # To make mypy happy
        assert annotations[0]["all_caps"] == request_state.result.completions[0].text.upper()

    def test_annotate_no_result(self):
        with pytest.raises(ValueError):
            self.annotator.annotate(self.request_state)
