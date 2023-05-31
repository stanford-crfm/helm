from abc import ABC, abstractmethod
from hashlib import sha512
import json
import random
import threading
from typing import Dict, List, Union, Set

from cattrs import unstructure
import scaleapi
from scaleapi.tasks import TaskType, TaskStatus
from scaleapi.exceptions import ScaleDuplicateResource

from helm.common.hierarchical_logger import hlog
from helm.common.cache import Cache, CacheConfig
from helm.common.critique_request import (
    CritiqueQuestionTemplate,
    CritiqueRequest,
    CritiqueRequestResult,
    CritiqueTaskTemplate,
    CritiqueResponse,
    QuestionType,
)


class CritiqueClient(ABC):
    """A client that allows making critique requests."""

    @abstractmethod
    def make_critique_request(self, request: CritiqueRequest) -> CritiqueRequestResult:
        """Get responses to a critique request."""
        pass


class RandomCritiqueClient(CritiqueClient):
    """A CritiqueClient that returns random choices for debugging."""

    def make_critique_request(self, request: CritiqueRequest) -> CritiqueRequestResult:
        responses: List[CritiqueResponse] = []
        random.seed(0)
        for respondent_index in range(request.template.num_respondents):
            answers: Dict[str, Union[str, List[str]]] = {}
            for question in request.template.questions:
                if question.question_type == QuestionType.MULTIPLE_CHOICE:
                    answers[question.name] = random.choice(question.options)
                elif question.question_type == QuestionType.CHECKBOX:
                    answers[question.name] = random.sample(question.options, random.randint(0, len(question.options)))
                elif question.question_type == QuestionType.FREE_RESPONSE:
                    answers[question.name] = random.choice(["foo", "bar", "bax", "qux"])
                else:
                    raise ValueError(f"Unknown question type: {question.question_type}")
            responses.append(
                CritiqueResponse(id=str(respondent_index), respondent_id=str(respondent_index), answers=answers)
            )
        return CritiqueRequestResult(responses)


class ScaleCritiqueClientError(Exception):
    pass


# Set of existing projects in Scale.
_scale_projects_lock: threading.Lock = threading.Lock()
_scale_projects: Set[str] = set()


def _ensure_project_exists(client: scaleapi.ScaleClient, project_name: str):
    """Ensure that the Scale project exists, creating it if necessary."""
    with _scale_projects_lock:
        if project_name not in _scale_projects:
            try:
                client.create_project(
                    project_name=project_name,
                    task_type=TaskType.TextCollection,
                    rapid=True,
                    params={},
                )
                hlog(f"Created new Scale project: {project_name}")
                hlog(
                    "IMPORTANT: Run scripts/scale/create_and_setup_project.py to set up a "
                    "calibration batch in your project."
                )
            except ScaleDuplicateResource:
                existing_project = client.get_project(project_name=project_name)
                if existing_project.type != TaskType.TextCollection.value:
                    raise ScaleCritiqueClientError(
                        f"The existing project with name '{project_name}' has a task type of "
                        f"'{existing_project.type}' instead of '{TaskType.TextCollection.value}'. "
                        "Rename the existing batch to a different name to allow HELM to create a new project "
                        "with the correct task type."
                    )
                hlog(f"Reusing existing Scale project: {project_name}")
            _scale_projects.add(project_name)


# Set of existing batches in Scale.
_scale_batches: Set[str] = set()
_scale_batches_lock: threading.Lock = threading.Lock()


def _ensure_batch_exists(client: scaleapi.ScaleClient, project_name: str, batch_name: str) -> None:
    """Ensure that the Scale batch exists, creating it if necessary."""
    _ensure_project_exists(client, project_name)
    with _scale_batches_lock:
        if batch_name not in _scale_batches:
            try:
                client.create_batch(
                    project=project_name,
                    batch_name=batch_name,
                    calibration_batch=False,
                    self_label_batch=False,
                )
                hlog(f"Created new Scale batch: {batch_name}")
            except ScaleDuplicateResource:
                existing_batch = client.get_batch(batch_name=batch_name)
                if existing_batch.project != project_name:
                    raise Exception(
                        f"A batch named '{batch_name}' already exists in a project '{existing_batch.project}' "
                        f"but credentials.conf has scaleProject set to a different project '{project_name}'. "
                        "Either rename the existing batch to a different name to allow HELM to create a new batch, or "
                        f"change scaleProject in credentials.conf to '{existing_batch.project}'. {existing_batch}"
                    )
                if existing_batch.status != "staging":
                    raise Exception(
                        f"New tasks cannot be added to the existing batch named '{batch_name}' because "
                        f"its status is '{existing_batch.status}' instead of 'staging'. "
                        "Rename the existing batch to a different name to allow HELM to create a new batch."
                    )
                hlog(f"Reusing existing Scale batch: {batch_name}")
            _scale_batches.add(batch_name)


_scale_cache_lock: threading.Lock = threading.Lock()
_SCALE_IMAGE_ATTACHMENT_KEY = "image_attachment"


class ScaleCritiqueClient(CritiqueClient):
    # TODO #1614: Move this to its own file
    """A CritiqueClient that creates tasks for workers on Scale.

    Scale AI concepts:

    - A **project** contains **tasks** which can be in **batches** (not used here)
    - A **task** is created in a project. It represents an individual unit of work to be done by a Tasker
      It contains **attachments** which is the data to be annotated, and **fields** which are the
        instructions and questions to be displayed to the Tasker. A task has also a general **instruction**
        which is displayed before the fields.
    - A **task response**: represents for each question a list of answers from different workers.

    Mapping of HELM concepts to Scale AI concepts:

    - A `CritiqueRequest` maps to a **task**
        - `CritiqueRequest.template` indicates which **project** the task should be created in.
    - A `CritiqueTaskTemplate` maps to a **task**
    - A `CritiqueQuestionTemplate` maps to a **field** in a task.
    - A `CritiqueResponse` maps to a **task response**.
    """

    def __init__(
        self,
        api_key: str,
        cache_config: CacheConfig,
        project_name: str,
    ):
        self._cache = Cache(cache_config)
        self._client = scaleapi.ScaleClient(api_key)
        self._project_name = project_name

    def _interpolate_fields(self, text: str, fields: Dict[str, str]) -> str:
        for field_name, field_value in fields.items():
            text = text.replace("{{" + field_name + "}}", field_value)
        return text

    def _critique_question_to_scale_field(self, question: CritiqueQuestionTemplate, fields: Dict[str, str]):
        if question.question_type == "multiple_choice" or question.question_type == "checkbox":
            return {
                "type": "category",
                "field_id": question.name,  # This must be unique, so we use the question name
                "title": question.name,
                "description": self._interpolate_fields(question.text, fields),
                "choices": [{"label": option, "value": option} for option in question.options],
                "min_choices": 0 if question.question_type == "checkbox" else 1,
                "max_choices": len(question.options) if question.question_type == "checkbox" else 1,
            }
        else:
            raise ValueError(f"Unsupported question type {question.question_type}")

    def _get_or_create_scale_task(self, template: CritiqueTaskTemplate, fields: Dict[str, str]) -> str:
        """Get or create a task on Scale and return the Scale task ID."""
        batch_name = template.name
        _ensure_batch_exists(client=self._client, project_name=self._project_name, batch_name=batch_name)

        # Used both for the cache key and the task unique_id
        cache_key = {
            "batch": batch_name,
            "task": unstructure(template),
            "fields": fields,
        }

        def create_scale_task() -> Dict[str, str]:
            """
            Creates a Scale Task (which is one single question from a CritiqueQuestionTemplate)
            Returns the Scale Task ID.
            """

            # We create a unique_id for the task so that we can reuse it if it already exists
            # It contains the same information as the task itself (like the cache key)
            # This is redundant with the cache but it's a good safety net
            # NOTE: Technically, sha512 could have collisions but it's unlikely.
            unique_id: str = sha512(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()
            instructions: str = self._interpolate_fields(template.instructions, fields)
            attachments: List[Dict[str, str]] = [
                {
                    "type": "text",
                    "content": instructions,
                }
            ]
            if _SCALE_IMAGE_ATTACHMENT_KEY in fields:
                # We add the image as the first attachment so that it is displayed first
                # (Usually the instructions are not as important when an image is present)
                attachments = [
                    {
                        "type": "image",
                        "content": fields[_SCALE_IMAGE_ATTACHMENT_KEY],
                    }
                ] + attachments
            payload = dict(
                batch=batch_name,
                unique_id=unique_id,
                instruction="Evaluate the AI model generated output following the instructions below",
                attachments=attachments,
                response_required=template.num_respondents,
                fields=[self._critique_question_to_scale_field(question, fields) for question in template.questions],
            )

            try:
                task = self._client.create_task(TaskType.TextCollection, **payload)
                return {"id": task.id}
            except ScaleDuplicateResource as err:
                hlog(f"ScaleDuplicateResource when creating task: {unique_id}. Error: {err.message}")
                task = self._client.get_task(unique_id)
                return {"id": task.id}

        with _scale_cache_lock:
            task_response, is_cached = self._cache.get(
                cache_key,
                create_scale_task,
            )
        task_id: str = task_response["id"]
        if is_cached:
            hlog(f"Reusing existing Scale task: {task_id}")
        else:
            hlog(f"Creating new Scale task: {task_id}")
        return task_id

    def finalize_batch(self, batch_name: str):
        self._client.finalize_batch(batch_name=batch_name)

    def _get_worker_responses(self, task_id: str) -> List[CritiqueResponse]:
        task: scaleapi.tasks.Task = self._client.get_task(task_id)
        if task.status != TaskStatus.Completed.value:
            return []
        else:
            annotations: Dict[str, List[str]] = task.response["annotations"]

            # The format of annotations is:
            # {
            #   "category_field_1": [
            #      answer_1_respondent_1,
            #      answer_1_respondent_2,
            #      ...
            #   ],
            #   "category_field_2": [
            #      answer_2_respondent_1,
            #      answer_2_respondent_2,
            #      ...
            #   ],
            #   ...
            # }
            # We want to convert it to:
            # [
            #   {
            #     "id": "respondent_1",
            #     "answers": {
            #       "category_field_1": answer_1_respondent_1
            #       "category_field_2": answer_2_respondent_1
            #       ...
            #     }
            #   },
            #   {
            #     "id": "respondent_2",
            #     "answers": {
            #       "category_field_1": answer_1_respondent_2
            #       "category_field_2": answer_2_respondent_2
            #       ...
            #     }
            #   },
            #   ...
            # ]

            # First, we get the list of respondents
            num_respondents: int = len(annotations[list(annotations.keys())[0]])

            # Then, we create the list of responses
            responses: List[CritiqueResponse] = []
            for respondent_index in range(num_respondents):
                answers: Dict[str, Union[str, List[str]]] = {}
                for field_name, field_answers in annotations.items():
                    answers[field_name] = field_answers[respondent_index]
                responses.append(
                    CritiqueResponse(id=str(respondent_index), respondent_id=str(respondent_index), answers=answers)
                )
            return responses

    def make_critique_request(self, request: CritiqueRequest) -> CritiqueRequestResult:
        """
        Create a task on Scale AI and fetch responses from Scale AI if available.

        Returns CritiqueRequestResult if worker answers are complete, or None otherwise.
        The intended use is to call it once to create the task, wait a while, and then call it
        later to fetch answers.

        First, attempt to find a Scale AI project for the template. If one exists, reuse that project.
        Otherwise, create a new project using the template.

        Second, attempt to find a Scale AI task inside this project for the fields. If one exists,
        reuse that task. Otherwise, create a new task inside the project using the fields.

        Finally, check if responses are available by checking if the number of workers who have responded
        is equal to the requested number of workers. If so, return those responses.

        This method is idempotent, because projects and tasks are not created if they already exist.

        The cache will store the mappings from template to Scale AI Project ID and from questions to Scale AI
        task ID. If the cache is deleted, the mappings should be conserved on Scale AI side and the API calls
        should return a ScaleDuplicateResource error which is handled by the method. We still prefer to use
        the cache to avoid unnecessary API calls and to not depend on Scale AI side.
        Note that worker responses are currently not cached.
        """
        # TODO: Remove/fix _get_or_create_scale_project_and_batch().
        # For now we are forcing the user to provide a batch_name in the credentials.

        # _, batch_name = self._get_or_create_scale_project_and_batch(request.template)
        # self._batch_name = batch_name
        task_id: str = self._get_or_create_scale_task(request.template, request.fields)
        worker_responses: List[CritiqueResponse] = self._get_worker_responses(task_id)
        return CritiqueRequestResult(worker_responses)
