from typing import Dict, List, Union
import string
import dataclasses

from helm.common.critique_request import (
    CritiqueRequest,
    CritiqueRequestResult,
    CritiqueResponse,
    CritiqueQuestionTemplate,
    CritiqueTaskTemplate,
)
from helm.common.request import Request, RequestResult, Sequence
from helm.proxy.clients.client import Client
from helm.proxy.clients.critique_client import CritiqueClient


class CritiqueParseError(Exception):
    pass


class ModelCritiqueClient(CritiqueClient):
    """A CritiqueClient that queries a Model to answer CritiqueRequests."""

    def __init__(self, client: Client, model_name):
        self._client = client
        self._model_name = model_name

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
            for i, letter in enumerate(string.ascii_uppercase[: len(question.options)]):
                prompt += f"\n{letter}. {question.options[i]}"
            if question.question_type == "multiple_choice":
                prompt += "\nAnswer with a single letter corresponding to the option.\nAnswer: "
            elif question.question_type == "checkbox":
                prompt += (
                    "\nAnswer with all letters seperated by commas, corresponding to the selected options.\nAnswer: "
                )
        return prompt

    def _task_to_requests(self, task: CritiqueTaskTemplate, fields: Dict[str, str]) -> List[Request]:
        base_prompt: str = self._interpolate_fields(task.instructions, fields)

        requests: List[Request] = []
        for question in task.questions:
            prompt: str = base_prompt + "\n\n" + self._question_to_prompt(question, fields)
            if question.question_type == "free_response":
                # TODO: Make max_tokens configurable
                max_tokens = 100
            elif question.question_type == "checkbox":
                # We multiply by 2 because the model will generate a comma after each option.
                max_tokens = len(question.options) * 2
            else:
                max_tokens = 1
            request = Request(
                model=self._model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                echo_prompt=False,
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
        assert len(answers) >= 1, f"Invalid answer: {completion}. There are no answers once parsed: {answers}."
        assert all(
            [answer in string.ascii_uppercase for answer in answers]
        ), f"Invalid answer: {completion}. Some answers are not capital letters, once parsed: {answers}."
        return answers

    def _multiple_choice_completion_to_answer(self, question: CritiqueQuestionTemplate, completion: Sequence) -> str:
        """Convert a multiple choice completion to an answer."""
        assert question.question_type == "multiple_choice"
        answers: List[str] = self._parse_completion_to_question_choice(completion.text)
        assert len(answers) == 1, f"Invalid answer: {completion}. Multiple choice questions should have one answer."
        return answers[0]

    def _checkbox_completion_to_answer(self, question: CritiqueQuestionTemplate, completion: Sequence) -> List[str]:
        """Convert a checkbox completion to an answer."""
        assert question.question_type == "checkbox"
        answers: List[str] = self._parse_completion_to_question_choice(completion.text)
        assert len(answers) <= len(
            question.options
        ), f"Invalid answer: {completion}. Checkbox questions should have at most one answer per option."
        return answers

    def _free_response_completion_to_answer(self, question: CritiqueQuestionTemplate, completion: Sequence) -> str:
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
            for question_index, result in enumerate(results):
                question = questions[question_index]
                answer: Union[str, List[str]] = ""
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
                mapped_answer: Union[str, List[str]] = self._letter_answer_to_mapped_answer(answer, question, fields)
                answers[question.name] = mapped_answer
            responses.append(CritiqueResponse(id=str(respondent_id), respondent_id=str(respondent_id), answers=answers))
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
