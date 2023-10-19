import csv
from datetime import datetime
import os
from threading import Lock
from typing import Dict, List, Sequence
import textwrap
import re

from helm.common.critique_request import CritiqueQuestionTemplate, CritiqueRequest, CritiqueTaskTemplate, QuestionType
from helm.common.general import ensure_directory_exists
from helm.common.hierarchical_logger import hlog
from helm.proxy.critique.mechanical_turk_utils import replace_emoji_characters


def _indent_to_level(text: str, level: int) -> str:
    """Helper for indenting XML to the same level as the external template."""
    return textwrap.indent(text, " " * 4 * level).lstrip()


def _format_template_tags(raw_text: str) -> str:
    """Convert from Surge AI template tag format to Mechanical Turk template tag format.

    {{field}} -> ${field}"""
    return re.sub(r"{{([^{}]+)}}", "${\\1}", raw_text)


def _render_template_crowd_html(task_template: CritiqueTaskTemplate) -> str:
    """Render the Crowd HTML for the template."""
    validation_crowd_html = textwrap.dedent(
        """\
        <script>
            // Validates that an option is selected for each radio group
            // because Mechanical Turk Crowd HTML does not do so automatically.
            // Source: https://stackoverflow.com/a/71064873
            function validateForm() {
                var valid = true;
                var radioGroups = document.querySelectorAll("crowd-radio-group");
                for (var i = 0; i < radioGroups.length; i++) {
                    var validGroup = false;
                    var radioButtons = radioGroups[i].children;
                    for (var j = 0; j < radioButtons.length; j++) {
                        validGroup = validGroup || radioButtons[j].checked;
                    }
                    valid = valid && validGroup;
                }
                return valid;
            }

            document.addEventListener("DOMContentLoaded", function(event) {
                document.querySelector('crowd-form').onsubmit = function(e) {
                    if (!validateForm()) {
                        alert("Please answer all the questions in order to submit.");
                        e.preventDefault();
                    }
                }
            });
        </script>"""
    )

    instructions_crowd_html = (
        f'<p style="white-space: pre-wrap;">{_format_template_tags(task_template.instructions)}</p>'
    )
    divider_html = "\n<hr>"
    questions_crowd_html = "\n<hr>\n".join(
        [_render_question_crowd_html(question) for question in task_template.questions]
    )
    return textwrap.dedent(
        f"""\
        <script src="https://assets.crowd.aws/crowd-html-elements.js"></script>
        {_indent_to_level(validation_crowd_html, 2)}
        <crowd-form answer-format="flatten-objects">
            {_indent_to_level(instructions_crowd_html, 3)}
            {_indent_to_level(divider_html, 3)}
            {_indent_to_level(questions_crowd_html, 3)}
            {_indent_to_level(divider_html, 3)}
        </crowd-form>"""
    )


def _render_question_crowd_html(question_template: CritiqueQuestionTemplate) -> str:
    """Render the Crowd HTML for a question."""
    question_input_crowd_html: str
    if question_template.question_type == QuestionType.FREE_RESPONSE:
        question_input_crowd_html = textwrap.dedent(
            f"""\
            <crowd-text-area name="{question_template.name}" required></crowd-text-area>"""
        )
    elif question_template.question_type == QuestionType.MULTIPLE_CHOICE:
        question_input_crowd_html = _render_multiple_choice_options_crowd_html(
            question_template.name, question_template.options
        )
    elif question_template.question_type == QuestionType.CHECKBOX:
        question_input_crowd_html = _render_checkbox_options_crowd_html(
            question_template.name, question_template.options
        )
    return textwrap.dedent(
        f"""\
        <p style=\"white-space: pre-wrap;\">
            {_format_template_tags(question_template.text)}
        </p>
        {_indent_to_level(question_input_crowd_html, 2)}"""
    )


def _render_multiple_choice_options_crowd_html(name: str, options: List[str]) -> str:
    """Render the Crowd HTML for the options of a multiple-choice question."""
    buttons_crowd_html = "\n<br>\n".join(
        [
            f"""<crowd-radio-button name="{name}.{index}">{_format_template_tags(option)}</crowd-radio-button>"""
            for index, option in enumerate(options)
        ]
    )
    return textwrap.dedent(
        f"""\
        <crowd-radio-group>
            {_indent_to_level(buttons_crowd_html, 3)}
        </crowd-radio-group>"""
    )


def _render_checkbox_options_crowd_html(name: str, options: List[str]) -> str:
    """Render the Crowd HTML for the options of a checkbox question."""
    return "\n<br>\n".join(
        [
            f"""<crowd-checkbox name="{name}.{index}">{_format_template_tags(option)}</crowd-checkbox>"""
            for index, option in enumerate(options)
        ]
    )


class _MechanicalTurkCritiqueRequestExporter:
    """Exports critique requests.

    - The requests will be exported to mturk/{template.name}/requests_{timestamp}.csv
    - The template Crowd HTML will be exported to mturk/{template.name}/layout_{timestamp}.html"""

    def __init__(self, template: CritiqueTaskTemplate):
        self._template: CritiqueTaskTemplate = template
        self._lock: Lock = Lock()

        self._directory_path = os.path.join("mturk", self._template.name)
        timestamp = datetime.now().isoformat()
        self._template_filename = os.path.join(self._directory_path, f"layout_{timestamp}.html")
        self._requests_filename = os.path.join(self._get_directory_path(), f"requests_{timestamp}.csv")

        # Protected by `_lock`.
        # Populated by `_initialize()`.
        self._field_names: Sequence[str] = []

    def _get_directory_path(self):
        # TODO: Make this configurable.
        return os.path.join("mturk", self._template.name)

    def _initialize(self, field_names: Sequence[str]) -> None:
        # self._lock must be held when calling this.
        ensure_directory_exists(self._get_directory_path())

        hlog(f"Exporting Mechanical Turk layout to {self._template_filename}")
        with open(self._template_filename, "w") as f:
            f.write(_render_template_crowd_html(self._template))

        hlog(f"Exporting Mechanical Turk requests to {self._requests_filename}")
        with open(self._requests_filename, "w") as f:
            self._field_names = field_names
            dict_writer: csv.DictWriter = csv.DictWriter(f, fieldnames=field_names)
            dict_writer.writeheader()

    def export(self, fields: Dict[str, str]):
        """Export a single critique request.

        - The request will be written as a row to mturk/{template.name}/requests_{timestamp}.csv
        - The template Crowd HTML will be written to mturk/{template.name}/layout_{timestamp}.html
          when this is called for the first time"""
        with self._lock:
            if not self._field_names:
                self._initialize(list(fields.keys()))
            assert self._field_names
            # Unfortunately, we have to re-open and close the file every time.
            # TODO: Support exporting batches of requests.
            with open(self._requests_filename, "a") as f:
                dict_writer = csv.DictWriter(f, fieldnames=self._field_names)
                dict_writer.writerow(fields)


_exporters_lock: Lock = Lock()
_exporters: Dict[str, _MechanicalTurkCritiqueRequestExporter] = {}


def export_request(request: CritiqueRequest):
    """Exports critique requests.

    After the calling this, the user should manually upload the generated CSV
    and Crowd HTML files to the Mechanical Turk web UI.

    - The requests will be exported to mturk/{template.name}/requests_{timestamp}.csv
    - The template Crowd HTML will be exported to mturk/{template.name}/layout_{timestamp}.html"""

    template: CritiqueTaskTemplate = request.template
    with _exporters_lock:
        if template.name not in _exporters:
            _exporters[template.name] = _MechanicalTurkCritiqueRequestExporter(template)
    encoded_fields = {
        field_name: replace_emoji_characters(field_value) for field_name, field_value in request.fields.items()
    }
    _exporters[template.name].export(encoded_fields)
