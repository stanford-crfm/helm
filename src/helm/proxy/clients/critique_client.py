from abc import ABC, abstractmethod
from hashlib import sha512
import json
import random
import threading
from typing import Dict, List, Union, Tuple, Optional

# Surge AI
from cattrs import unstructure
import surge
from surge import questions as surge_questions

# Scale
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


_surge_cache_lock = threading.Lock()
_scale_cache_lock = threading.Lock()


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


class SurgeAICritiqueClient(CritiqueClient):
    # TODO #1614: Move this to its own file
    """A CritiqueClient that creates tasks for workers on Surge AI.

    Surge AI concepts:

    - A **project** contains **instructions** and **questions**, which are templates that can contain
      placeholders.
    - A **task** is created in a project and contains **fields** that are interpolated into the
      placeholders in the project's instructions and questions templates to instantiate the actual instructions
      and questions in the task.
    - A **task response** is a response to a task by a single worker and contains answers to all the questions
      in the task.

    Mapping of HELM concepts to Surge AI concepts:

    - A `CritiqueTaskTemplate` maps to a **project**.
    - A `CritiqueQuestionTemplate` maps to a **question** template in a project.
    - A `CritiqueRequest` maps to a **task**
        - `CritiqueRequest.template` indicates which **project** the task should be created in.
        - `CritiqueRequest.fields` provides the fields that are interpolated into the placeholders in the
          projects' instructions and questions templates.
    - A `CritiqueResponse` maps to a **task response**.
    - A `CritiqueRequestResult` maps to a list of **task responses** across multiple workers for a task.
    """

    def __init__(self, api_key: str, cache_config: CacheConfig):
        surge.api_key = api_key
        self._cache = Cache(cache_config)

    def _to_surge_question(self, question: CritiqueQuestionTemplate) -> surge_questions.MultipleChoiceQuestion:
        if question.question_type != "multiple_choice":
            raise ValueError("Currently, only multiple_choice questions are supported")
        return surge_questions.MultipleChoiceQuestion(
            text=question.text,
            options=question.options,
        )

    def _get_or_create_surge_project(self, template: CritiqueTaskTemplate) -> str:
        """Get or create a project on Surge AI and return the Surge AI project ID.

        Attempt to find a Surge AI project for the template from the cache. If one exists, reuse that project.
        Otherwise, create a new project using the template and save it to the cache. Return the Surge AI project ID."""

        def create_surge_project():
            project = surge.Project.create(
                name=template.name,
                instructions=template.instructions,
                questions=[self._to_surge_question(question) for question in template.questions],
                num_workers_per_task=template.num_respondents,
            )
            return {"id": project.id}

        with _surge_cache_lock:
            # Example cache key:
            # {
            #   "template": {
            #     # See CritiqueQuestionTemplate for complete schema
            #     "name": "some_name",
            #     "instructions": "some_instructions",
            #     "num_respondents": 1,
            #     "questions": []
            #   }
            # }
            #
            # Example cache value:
            # {"id": "17e323f1-f7e4-427c-a2d5-456743aba8"}
            #
            # Note:
            # We do not cache the additional fields returned by surge.Project.create()
            # because they are mutable server-side, and server-side mutations may cause
            # stale cache issues.
            project_response, is_cached = self._cache.get({"template": unstructure(template)}, create_surge_project)
        project_id = project_response["id"]
        if is_cached:
            hlog(f"Reusing existing Surge AI project: {project_id}")
        else:
            hlog(f"Creating new Surge AI project: {project_id}")
        return project_id

    def _get_or_create_task(self, project_id: str, fields: Dict[str, str]) -> str:
        """Get or create a task on Surge AI and return the Surge AI project ID.

        Attempt to find a Surge AI task inside this project for the fields from the cache.
        If one exists, reuse that task. Otherwise, create a new task inside the project using the fields.
        and save it to the cache. Return the Surge AI task ID."""
        project = surge.Project.retrieve(project_id)

        def create_surge_task():
            tasks = project.create_tasks([fields], launch=False)  # TODO: Make launch parameter configurable
            if len(tasks) != 1:
                return RuntimeError(f"Expected one task in Surge response, but got {len(tasks)} tasks")
            task = tasks[0]
            return {"id": task.id}

        with _surge_cache_lock:
            # Example cache key:
            # {
            #   "project_id": "17e323f1-f7e4-427c-a2d5-456743aba8",
            #   "fields": {
            #     "some_field": "some_value"
            #   }
            # }
            #
            # Example cache value:
            # {"id": "17e323f1-f7e4-427c-a2d5-456743aba8"}
            #
            # Note:
            # We do not cache the additional fields returned by surge.Project.create()
            # because they are mutable server-side, and server-side mutations may cause
            # stale cache issues.
            task_response, is_cached = self._cache.get({"project_id": project_id, "fields": fields}, create_surge_task)
        task_id = task_response["id"]
        if is_cached:
            hlog(f"Reusing existing Surge AI task: {task_id}")
        else:
            hlog(f"Creating new Surge AI task: {task_id}")
        return task_id

    def _get_worker_responses(self, task_id: str, questions: List[CritiqueQuestionTemplate]) -> List[CritiqueResponse]:
        task = surge.Task.retrieve(task_id)
        return [
            CritiqueResponse(
                id=task_response.id,
                respondent_id=task_response.worker_id,
                answers={question.name: task_response.data[question.name] for question in questions},
            )
            for task_response in task.responses
        ]

    def make_critique_request(self, request: CritiqueRequest) -> CritiqueRequestResult:
        """Create a task on Surge AI and fetch responses from Surge AI if available.

        Returns CritiqueRequestResult if worker answers are complete, or None otherwise.
        The intended use is to call it once to create the task, wait a while, and then call it
        later to fetch answers.

        First, attempt to find a Surge AI project for the template. If one exists, reuse that project.
        Otherwise, create a new project using the template.

        Second, attempt to find a Surge AI task inside this project for the fields. If one exists,
        reuse that task. Otherwise, create a new task inside the project using the fields.

        Finally, check if responses are available by checking if the number of workers who have responded
        is equal to the requested number of workers. If so, return those responses.

        This method is idempotent, because projects and tasks are not created if they already exist.

        The cache will store the mappings from template to Surge AI Project ID and from fields to Surge AI
        question ID. If the cache is deleted, the mappings will be lost, and this method will not be able
        to fetch results from the previous projects and tasks, and will have to create new projects and tasks.
        Note that worker responses are currently not cached."""
        project_id = self._get_or_create_surge_project(request.template)
        task_id = self._get_or_create_task(project_id, request.fields)
        worker_responses = self._get_worker_responses(task_id, request.template.questions)
        return CritiqueRequestResult(worker_responses)


class ScaleCritiqueClient(CritiqueClient):
    # TODO #1614: Move this to its own file
    """A CritiqueClient that creates tasks for workers on Scale.

    Scale AI concepts:

    - A **project** contains **tasks** which can be in **batches** (not used here)
    - A **task** is created in a project. It represents an individual unit of work to be done by a Tasker
      It contains **attachments** which is the data to be annotated, and **fields** which are the
        instructions and questions to be displayed to the Tasker. A task has also a general **instruction**
        which is displayed before the fields.
    - A **task response**: TODO

    Mapping of HELM concepts to Scale AI concepts:

    - A `CritiqueRequest` maps to a **task**
        - `CritiqueRequest.template` indicates which **project** the task should be created in.
    - A `CritiqueTaskTemplate` maps to a **task**
    - A `CritiqueQuestionTemplate` maps to a **field** in a task.
    - A `CritiqueResponse` maps to TODO
    """

    def __init__(
        self,
        api_key: str,
        cache_config: CacheConfig,
        project_name: Optional[str] = None,
        batch_name: Optional[str] = None,
    ):
        self._cache = Cache(cache_config)
        self._client = scaleapi.ScaleClient(api_key)
        self._project_name = project_name
        self._batch_name = batch_name

    def _get_or_create_scale_project_and_batch(self, template: CritiqueTaskTemplate) -> Tuple[str, str]:
        """Get or create a project and a batch associated to it on Scale and return the
        Scale project name and batch name.

        If the project_name was specified in the credentials, use it (should already exist).
        In order to send tasks, the project needs to already have a calibration batch.
        Otherwise, create a new project using the template name.
        If the project is created, there will be no calibration batch so the batch will be set
        as self-labeling.

        If the batch_name was specified in the credentials, use it (does not necessarily need to exist).
        Otherwise, create a new batch using the project name and the template name.

        This is using Scale Rapid.

        Attempt to find a Scale project for the template. If one exists, reuse that project.
        Otherwise, create a new project using the template. Return the Scale project name.
        Same for the batch."""

        project_was_created: bool = self._project_name is None
        project_name: Optional[str] = None
        batch_name: Optional[str] = None

        def create_scale_project():
            try:
                project = self._client.create_project(
                    project_name=template.name,
                    task_type=TaskType.TextCollection,
                    rapid=True,
                    params={},
                )
            except ScaleDuplicateResource as err:
                hlog(f"ScaleDuplicateResource when creating project: {template.name}. Error: {err.message}")
                # Get the existing project and checks that it has the same params
                # NOTE: This should not happen with the cache but in case the cache is deleted
                # we want to make sure we don't create a new project with the same name
                project = self._client.get_project(template.name)
                if project.type != TaskType.TextCollection.value:
                    raise RuntimeError(
                        f"Project {template.name} already exists with different task_type: "
                        f"'{project.type}' instead of '{TaskType.TextCollection.value}'"
                    ) from err
            return {"project_name": project.name}

        def create_scale_batch():
            if project_name is None:
                raise RuntimeError("Trying to create a batch without a project name.")
            try:
                batch_name: str = f"{project_name}-HELMBatch"
                batch = self._client.create_batch(
                    project=project_name,
                    batch_name=batch_name,
                    calibration_batch=False,
                    # If the project was created, there is no calibration batch so we set it as self-labeling
                    # otherwise the API will return an error.
                    self_label_batch=True,  # project_was_created,
                )
            except ScaleDuplicateResource as err:
                hlog(
                    f"ScaleDuplicateResource when creating batch: '{batch_name}' in project: "
                    f"{project_name}. Error: {err.message}"
                )
                # Get the existing batch and checks that it has the same project
                # NOTE: This should not happen with the cache but in case the cache is deleted
                # we want to make sure we don't create a new batch with the same name
                batch = self._client.get_batch(batch_name)
                if batch.project != project.name:
                    raise RuntimeError(
                        f"Batch {batch_name} already exists with different project: " f"{batch.project}"
                    ) from err
            return {"batch_name": batch.name}

        # Check if a project_name was specified in the credentials
        if self._project_name is not None:
            project_name = self._project_name
            hlog(f"Using existing Scale project: {project_name} (defined in credentials)")
            # Checks that the project exists
            try:
                project = self._client.get_project(project_name)
                project_name = project.name
            except Exception as err:
                raise RuntimeError(f"Project {project_name} does not exist") from err

            # Check if a batch_name was specified in the credentials
            if self._batch_name is not None:
                batch_name = self._batch_name
                hlog(f"Using existing Scale batch: {batch_name} (defined in credentials)")
                # Checks that the batch exists
                try:
                    batch = self._client.get_batch(batch_name)
                except Exception as err:
                    raise RuntimeError(f"Batch {batch_name} does not exist") from err
                # Checks that the batch is in the project
                if batch.project != project_name:
                    raise RuntimeError(
                        f"Batch {batch_name} is not in project {project_name}. "
                        f"It is in project {batch.project} instead."
                    )
                return project_name, batch_name

            # If we reach here it means that a project_name was specified but not a batch_name
            # Create a new batch using the template name
            with _scale_cache_lock:
                batch_response, is_cached = self._cache.get(
                    {"project": project_name, "template": template.name}, create_scale_batch
                )
            batch_name = batch_response["batch_name"]
        else:
            # If we reach here it means that no project_name was specified
            # Create a new project using the template name
            # Create a new batch using the project name and the template name

            with _scale_cache_lock:
                project_response, is_cached = self._cache.get({"template": template.name}, create_scale_project)
            project_name = project_response["project_name"]
            if is_cached:
                hlog(f"Reusing existing Scale project: {project_name}")
            else:
                hlog(f"Creating new Scale project: {project_name}")

            with _scale_cache_lock:
                batch_response, is_cached = self._cache.get(
                    {"project": project_name, "template": template.name}, create_scale_batch
                )
            batch_name = batch_response["batch_name"]

        if is_cached:
            hlog(f"Reusing existing Scale batch: {batch_name}")
        else:
            hlog(f"Creating new Scale batch: {batch_name}")

        return project_name, batch_name

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
                "min_choices": 1,
                "max_choices": len(question.options) if question.question_type == "checkbox" else 1,
            }
        else:
            raise ValueError(f"Unsupported question type {question.question_type}")

    def _get_or_create_scale_task(
        self, project_name: str, batch_name: str, template: CritiqueTaskTemplate, fields: Dict[str, str]
    ) -> str:
        """Get or create a task on Scale and return the Scale task ID."""

        # Used both for the cache key and the task unique_id
        cache_key = {"project": project_name, "batch": batch_name, "task": unstructure(template), "fields": fields}

        def create_scale_task() -> Dict[str, str]:
            """
            Creates a Scale Task (which is one single question from a CritiqueQuestionTemplate)
            Returns the Scale Task ID.
            """

            # We create a unique_id for the task so that we can reuse it if it already exists
            # It contains the same information as the task itself (like the cache key)
            # This is redundant with the cache but it's a good safety net
            # NOTE: Technically, sha512 could have collisions but it's unlikely.
            unique_id: str = sha512(str(json.dumps(cache_key, sort_keys=True)).encode()).hexdigest()
            instructions: str = self._interpolate_fields(template.instructions, fields)
            payload = dict(
                project=project_name,
                batch=batch_name,
                unique_id=unique_id,
                instruction="Evaluate the AI model generated output following the instructions below",
                attachment_type="text",
                attachments=[
                    {
                        "type": "text",
                        "content": instructions,
                    }
                ],
                response_required=template.num_respondents,
                fields=[self._critique_question_to_scale_field(question, fields) for question in template.questions],
            )

            try:
                task = self._client.create_task(TaskType.TextCollection, **payload)
                return {"id": task.id}
            except ScaleDuplicateResource as err:
                hlog(f"ScaleDuplicateResource when creating task: {unique_id}. Error: {err.message}")
                # Get the existing task and checks that it has the same content (attachments and fields)
                # NOTE: This should not happen with the cache but in case the cache is deleted
                # we want to make sure we don't create a new task with the same content
                task = self._client.get_task(unique_id)
                if task.params["attachments"] != payload["attachments"]:
                    raise RuntimeError(
                        f"Task {unique_id} already exists with different attachments: " f"{task.params['attachments']}"
                    ) from err
                # No need to check for fields, project_name and instructions because they are part of the unique_id
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

    def _finalize_batch(self, batch_name: str):
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
        project_name, batch_name = self._get_or_create_scale_project_and_batch(request.template)
        task_id: str = self._get_or_create_scale_task(project_name, batch_name, request.template, request.fields)
        worker_responses: List[CritiqueResponse] = self._get_worker_responses(task_id)
        # self._finalize_batch(batch_name)
        return CritiqueRequestResult(worker_responses)
