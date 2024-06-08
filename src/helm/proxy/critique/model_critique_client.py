from typing import Dict, List, Union, Optional
import string
import dataclasses

from helm.benchmark.run_spec_factory import get_default_model_deployment_for_model
from helm.common.critique_request import (
    CritiqueRequest,
    CritiqueRequestResult,
    CritiqueResponse,
    CritiqueQuestionTemplate,
    CritiqueTaskTemplate,
)
from helm.common.hierarchical_logger import hlog
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request, RequestResult, GeneratedOutput
from helm.clients.client import Client
from helm.proxy.critique.critique_client import CritiqueClient
from helm.common.media_object import MultimediaObject, MediaObject


class CritiqueParseError(Exception):
    pass


class ModelCritiqueClient(CritiqueClient):
    """A CritiqueClient that queries a Model to answer CritiqueRequests."""

    VISION_LANGUAGE_MODELS = ["openai/gpt-4-vision", "reka/reka", "huggingface/prometheus-vision"]

    def __init__(self, client: Client, model_name):
        self._client = client
        self._model_name = model_name
        self._model_deployment_name = (
            get_default_model_deployment_for_model(model_name, warn_arg_deprecated=False, ignore_deprecated=True)
            or self._model_name
        )
        self.vision_language = False
        for vision_language_model_name in self.VISION_LANGUAGE_MODELS:
            if model_name.startswith(vision_language_model_name):
                self.vision_language = True
                break

    def _interpolate_fields(self, text: str, fields: Dict[str, str]) -> str:
        for key, value in fields.items():
            text = text.replace("{{" + key + "}}", value)
        return text

    def _question_to_prompt(self, question: CritiqueQuestionTemplate, fields: Dict[str, str]) -> str:
        prompt: str = self._interpolate_fields(question.text, fields)
        if question.question_type == "free_response":
            prompt += "\nAnswer: "
        else:
            prompt += "\nOptions: "
            if len(question.options) > 26:
                raise CritiqueParseError("Currently, only up to 26 options are supported")
            for i, letter in enumerate(string.ascii_uppercase[: len(question.options)]):
                prompt += f"\n{letter}. {question.options[i]}"
            if question.question_type == "multiple_choice":
                prompt += "\nAnswer with a capital letter.\nAnswer: "
            elif question.question_type == "checkbox":
                prompt += "\nAnswer with capital letters separated by commas. You may select several options.\nAnswer: "
        return prompt

    def _task_to_requests(self, task: CritiqueTaskTemplate, fields: Dict[str, str]) -> List[Request]:
        base_prompt: str = self._interpolate_fields(task.instructions, fields)

        requests: List[Request] = []
        for question in task.questions:
            prompt: str
            if len(question.text) > 0:
                prompt = base_prompt + "\n\n" + self._question_to_prompt(question, fields)
            else:
                # We may don't want to add extra newlines and prompts
                # if the question text is empty (e.g., the Vibe-Eval evaluator).
                prompt = base_prompt
            if question.question_type == "free_response":
                max_tokens = 100 if task.max_tokens is None else task.max_tokens
            elif question.question_type == "checkbox":
                # We multiply by 2 because the model will generate a comma after each option.
                max_tokens = len(question.options) * 2
            else:
                max_tokens = 1

            # Special case for Anthropic to handle prefix and suffix.
            # TODO(josselin): Fix this once refactor of HELM allows for automatic model prefix and suffix.
            if self._model_name.startswith("anthropic"):
                try:
                    import anthropic
                except ModuleNotFoundError as e:
                    handle_module_not_found_error(e, ["anthropic"])

                prompt = anthropic.HUMAN_PROMPT + prompt + anthropic.AI_PROMPT

            multimodal_prompt: Optional[MultimediaObject] = None
            if self.vision_language:
                assert question.media_object is not None, "Expect media_object for vision-language models"
                image_media: MediaObject = question.media_object
                text_media: MediaObject = MediaObject(text=prompt, content_type="text/plain")
                multimodal_prompt = MultimediaObject(media_objects=[image_media, text_media])
                prompt = ""  # set to empty string to avoid conflicts with multimodal_prompt

            request = Request(
                model=self._model_name,
                model_deployment=self._model_deployment_name,
                prompt=prompt,
                max_tokens=max_tokens,
                echo_prompt=False,
                multimodal_prompt=multimodal_prompt,
            )
            requests.append(request)
        return requests

    def _execute_requests(self, requests: List[Request], num_respondents: int) -> List[List[RequestResult]]:
        """Execute a list of requests and return the responses."""
        responses: List[List[RequestResult]] = []
        for request in requests:
            responses.append([])
            for i in range(num_respondents):
                request_with_random = dataclasses.replace(request, random=str(i))
                response: RequestResult = self._client.make_request(request_with_random)
                responses[-1].append(response)
        return responses

    def _parse_completion_to_question_choice(self, completion: str) -> List[str]:
        """Convert a completion to a list of answer represented by a list of capital letters."""
        completion_parsed = completion.replace(" ", "").replace("\n", "").replace(".", "").upper()
        answers = completion_parsed.split(",")
        if len(answers) < 1:
            raise CritiqueParseError(f"Invalid answer: {completion}. There are no answers once parsed: {answers}.")
        for i, answer in enumerate(answers):
            if answer not in string.ascii_uppercase:
                raise CritiqueParseError(
                    f"Invalid answer: {completion}. Some answers are not capital letters, once parsed: {answers}. "
                    f"Error happened at answer {i}, which is {answer}."
                )
        return answers

    def _multiple_choice_completion_to_answer(
        self, question: CritiqueQuestionTemplate, completion: GeneratedOutput
    ) -> Optional[str]:
        """Convert a multiple choice completion to an answer."""
        assert question.question_type == "multiple_choice"
        try:
            answers: List[str] = self._parse_completion_to_question_choice(completion.text)
            if len(answers) != 1:
                raise CritiqueParseError(
                    f"Invalid answer: {completion}. Multiple choice questions should have one answer."
                )
            letter_answer = answers[0]
            choice_rank = string.ascii_uppercase.index(letter_answer)
            if choice_rank >= len(question.options):
                raise CritiqueParseError(
                    f"Invalid answer: {completion}. The answer is out of range of the options: {question.options}"
                )
            return letter_answer
        except CritiqueParseError as e:
            # If there was an error parsing the answer, we assume the user did not answer the question.
            hlog(f"Error parsing answer: {e}. Skipping question (and so the respondent entirely)")
            return None

    def _checkbox_completion_to_answer(
        self, question: CritiqueQuestionTemplate, completion: GeneratedOutput
    ) -> Optional[List[str]]:
        """Convert a checkbox completion to an answer."""
        assert question.question_type == "checkbox"
        try:
            answers: List[str] = self._parse_completion_to_question_choice(completion.text)
            if len(answers) > len(question.options):
                raise CritiqueParseError(
                    f"Invalid answer: {completion}. Checkbox questions should have at most one answer per option."
                )
            return answers
        except CritiqueParseError as e:
            # If there was an error parsing the answer, we assume the user did not answer the question.
            hlog(f"Error parsing answer: {e}. Skipping question (and so the respondent entirely)")
            return None

    def _free_response_completion_to_answer(
        self, question: CritiqueQuestionTemplate, completion: GeneratedOutput
    ) -> str:
        """Convert a free response completion to an answer."""
        assert question.question_type == "free_response"
        return completion.text

    def _letter_answer_to_mapped_answer(
        self, letter_answer: Union[str, List[str]], question: CritiqueQuestionTemplate, fields: Dict[str, str]
    ) -> Union[str, List[str]]:
        """Convert a letter answer to a mapped answer."""
        if question.question_type == "multiple_choice":
            assert isinstance(letter_answer, str)
            return self._interpolate_fields(question.options[string.ascii_uppercase.index(letter_answer)], fields)
        elif question.question_type == "checkbox":
            assert isinstance(letter_answer, str)
            return [
                self._interpolate_fields(question.options[string.ascii_uppercase.index(letter)], fields)
                for letter in letter_answer
            ]
        else:
            # Free response should be returned as is.
            assert isinstance(letter_answer, str)
            return letter_answer

    def _get_responses(
        self, questions: List[CritiqueQuestionTemplate], results: List[List[RequestResult]], fields: Dict[str, str]
    ) -> List[CritiqueResponse]:
        """Convert a list of request results to a list of CritiqueResponses."""
        assert len(questions) == len(results)

        responses: List[CritiqueResponse] = []
        for respondent_id in range(len(results[0])):
            answers: Dict[str, Union[str, List[str]]] = {}
            valid_response: bool = True
            for question_index, result in enumerate(results):
                question = questions[question_index]
                answer: Optional[Union[str, List[str]]] = None
                if not result[respondent_id].success:
                    raise RuntimeError(f"Request failed: {result[respondent_id]}.")
                if question.question_type == "multiple_choice":
                    answer = self._multiple_choice_completion_to_answer(question, result[respondent_id].completions[0])
                elif question.question_type == "checkbox":
                    answer = self._checkbox_completion_to_answer(question, result[respondent_id].completions[0])
                elif question.question_type == "free_response":
                    answer = self._free_response_completion_to_answer(question, result[respondent_id].completions[0])
                else:
                    raise ValueError(f"Unknown question type: {question.question_type}")

                # If the answer is None, it means the user did not answer the question.
                # Not only we will not add the answer to the response, but we will completely
                # skip the respondent.
                if answer is None:
                    valid_response = False
                    break

                mapped_answer: Union[str, List[str]] = self._letter_answer_to_mapped_answer(answer, question, fields)
                answers[question.name] = mapped_answer
            if valid_response:
                responses.append(
                    CritiqueResponse(id=str(respondent_id), respondent_id=str(respondent_id), answers=answers)
                )
        return responses

    def make_critique_request(self, request: CritiqueRequest) -> CritiqueRequestResult:
        """Queries the model specified in the constructor to answer a CritiqueRequest."""

        # This returns one request per question. We still need to duplicate each request
        # for the number of respondents.
        requests: List[Request] = self._task_to_requests(request.template, request.fields)

        # This returns a list (for each question) of lists (for each respondent) of RequestResults.
        results: List[List[RequestResult]] = self._execute_requests(requests, request.template.num_respondents)

        # Parse the completions into CritiqueResponses.
        responses: List[CritiqueResponse] = self._get_responses(request.template.questions, results, request.fields)

        return CritiqueRequestResult(responses)
