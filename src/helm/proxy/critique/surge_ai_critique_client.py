from cattrs import unstructure
import threading
from typing import Dict, List

from helm.common.cache import Cache, CacheConfig
from helm.common.critique_request import (
    CritiqueQuestionTemplate,
    CritiqueRequest,
    CritiqueRequestResult,
    CritiqueResponse,
    CritiqueTaskTemplate,
)
from helm.common.hierarchical_logger import hlog
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.proxy.critique.critique_client import CritiqueClient

try:
    import surge
    from surge import questions as surge_questions
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["human-evaluation"])


_surge_cache_lock = threading.Lock()


class SurgeAICritiqueClient(CritiqueClient):
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
