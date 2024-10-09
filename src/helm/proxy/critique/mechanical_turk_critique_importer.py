from collections import defaultdict
import csv
import os
from threading import Lock
from typing import Dict, List, Optional, Tuple, Union
import re
import sys

from helm.common.critique_request import (
    CritiqueRequest,
    CritiqueResponse,
    CritiqueTaskTemplate,
    QuestionType,
    CritiqueRequestResult,
)
from helm.common.hierarchical_logger import hlog
from helm.proxy.critique.mechanical_turk_utils import replace_emoji_characters

csv.field_size_limit(sys.maxsize)

# A representation of fields that can be used as a dict key.
_CritiqueRequestKey = Tuple[Tuple[str, str], ...]


class _MechanicalTurkRequestImporter:
    """Exports critique request results.

    The request results will be imported from all files matching
    mturk/{template.name}/Batch_{batch_number}_batch_results.csv"""

    def __init__(self, template: CritiqueTaskTemplate):
        self._template: CritiqueTaskTemplate = template
        self._request_key_to_results: Dict[_CritiqueRequestKey, CritiqueRequestResult] = {}

    def _get_directory_path(self):
        return os.path.join("mturk", self._template.name)

    def _make_request_key(self, fields: Dict[str, str]) -> _CritiqueRequestKey:
        """Make a request key from request fields."""
        return tuple((k, v) for k, v in sorted(fields.items()))

    def _import_from_file_path(self, file_path: str) -> None:
        """Import all rows from the CSV and store them in `self._request_key_to_results`."""
        request_key_to_responses: Dict[_CritiqueRequestKey, List[CritiqueResponse]] = defaultdict(list)
        with open(file_path) as f:
            dict_reader = csv.DictReader(f)
            for row in dict_reader:
                request_key = self._make_request_key(self._get_fields_from_row(row))
                response = self._get_response_from_row(row)
                request_key_to_responses[request_key].append(response)
            for request_key, responses in request_key_to_responses.items():
                self._request_key_to_results[request_key] = CritiqueRequestResult(responses)

    def _get_fields_from_row(self, row: Dict[str, str]) -> Dict[str, str]:
        fields = {}
        for key, value in row.items():
            if key.startswith("Input."):
                field_key = key[len("Input.") :]
                fields[field_key] = value
        return fields

    def _get_response_from_row(self, row: Dict[str, str]) -> CritiqueResponse:
        answers: Dict[str, Union[str, List[str]]] = {}
        for question in self._template.questions:
            if question.question_type == QuestionType.MULTIPLE_CHOICE:
                for option_index, option in enumerate(question.options):
                    raw_answer = row[f"Answer.{question.name}.{option_index}.on"]
                    if raw_answer == "true":
                        answers[question.name] = option
                        break
            elif question.question_type == QuestionType.CHECKBOX:
                checkbox_options: List[str] = []
                for option_index, option in enumerate(question.options):
                    raw_answer = row[f"Answer.{question.name}.{option_index}.on"]
                    if raw_answer == "true":
                        checkbox_options.append(option)
                answers[question.name] = checkbox_options
            elif question.question_type == QuestionType.FREE_RESPONSE:
                answers[question.name] = row[f"Answer.{question.name}"]
            else:
                raise ValueError(f"Unknown question_type: {question.question_type}")
        return CritiqueResponse(
            id=row["AssignmentId"],
            respondent_id=row["WorkerId"],
            answers=answers,
        )

    def initialize(self) -> None:
        """Initialize the instance.

        Thread-hostile.
        Must be called exactly once per instance.
        Must be called before `import_request_result()`."""
        if not os.path.exists(self._get_directory_path()) or not os.path.isdir(self._get_directory_path()):
            return

        for file_name in os.listdir(self._get_directory_path()):
            if re.match("Batch_\\d+_batch_results.csv", file_name):
                file_path = os.path.join(self._get_directory_path(), file_name)
                hlog(f"Importing Mechanical Turk results from {file_path}")
                self._import_from_file_path(file_path)

    def import_request_result(self, fields: Dict[str, str]) -> Optional[CritiqueRequestResult]:
        """Import the request result.

        `initialize()` must be called before calling this."""
        return self._request_key_to_results.get(self._make_request_key(fields))


_importers_lock: Lock = Lock()
_importer: Dict[str, _MechanicalTurkRequestImporter] = {}


def import_request_result(request: CritiqueRequest) -> Optional[CritiqueRequestResult]:
    """Imports a request result from CSV files.

    Before calling this, the user should manually download the response CSV files from the
    Mechanical Turk web UI and place them at
    turk/{template.name}/Batch_{batch_number}_batch_results.csv"""
    template: CritiqueTaskTemplate = request.template
    with _importers_lock:
        if template.name not in _importer:
            _importer[template.name] = _MechanicalTurkRequestImporter(template)
            _importer[template.name].initialize()
    encoded_fields = {
        field_name: replace_emoji_characters(field_value) for field_name, field_value in request.fields.items()
    }
    return _importer[template.name].import_request_result(encoded_fields)
