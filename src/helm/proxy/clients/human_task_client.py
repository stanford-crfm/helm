from abc import ABC, abstractmethod
import random
import threading
from typing import Dict, List, Optional

from helm.common.hierarchical_logger import hlog
from helm.common.cache import Cache, CacheConfig
from helm.common.human_task_request import (
    HumanQuestionTemplate,
    HumanTaskRequest,
    HumanTaskRequestResult,
    HumanTaskTemplate,
    HumanTaskWorkerResponse,
)

from cattrs import unstructure

import surge
from surge import questions as surge_questions


_surge_cache_lock = threading.Lock()


class HumanTaskClient(ABC):
    @abstractmethod
    def get_human_task(self, request: HumanTaskRequest) -> Optional[HumanTaskRequestResult]:
        """Get responses to a human task.

        Creates the human task on a labeller platform if it does not already exist.
        Returns HumanTaskRequestResult if worker answers are complete, or None otherwise.
        The intended use is to call it once to create the task, wait a while, and then call it
        later to fetch answers."""
        pass


class RandomHumanTaskClient(HumanTaskClient):
    """A HumanTaskClient that returns random choices for debugging."""

    def get_human_task(self, request: HumanTaskRequest) -> Optional[HumanTaskRequestResult]:
        workers: List[HumanTaskWorkerResponse] = []
        for _ in range(request.template.num_workers):
            answers: List[str] = []
            for question in request.template.questions:
                if question.question_type != "multiple_choice":
                    raise ValueError("Currently, only multiple_choice questions are supported")
                answers.append(random.choice(question.options))
            workers.append(HumanTaskWorkerResponse(answers=answers))
        return HumanTaskRequestResult(workers=workers)


class SurgeAIHumanTaskClient(HumanTaskClient):
    """A HumanTaskClient that creates tasks for workers on Surge AI."""

    def __init__(self, api_key: str, cache_config: CacheConfig):
        surge.api_key = api_key
        self._cache = Cache(cache_config)

    def _to_surge_question(self, question: HumanQuestionTemplate) -> surge_questions.MultipleChoiceQuestion:
        if question.question_type != "multiple_choice":
            raise ValueError("Currently, only multiple_choice questions are supported")
        return surge_questions.MultipleChoiceQuestion(
            text=question.text,
            options=question.options,
        )

    def _get_or_create_surge_project(self, template: HumanTaskTemplate) -> str:
        """Get or create a project on Surge AI and return the Surge AI project ID.

        Attempt to find a Surge AI project for the template. If one exists, reuse that project.
        Otherwise, create a new project using the template. Return the Surge AI project ID."""

        def create_surge_project():
            project = surge.Project.create(
                name=template.name,
                instructions=template.instructions,
                questions=[self._to_surge_question(question) for question in template.questions],
                num_workers_per_task=template.num_workers,
            )
            return {"id": project.id}

        with _surge_cache_lock:
            project_response, is_cached = self._cache.get({"template": unstructure(template)}, create_surge_project)
        project_id = project_response["id"]
        if is_cached:
            hlog(f"Reusing existing Surge AI project: {project_id}")
        else:
            hlog(f"Creating new Surge AI project: {project_id}")
        return project_id

    def _get_or_create_task(self, project_id: str, fields: Dict[str, str]) -> str:
        """Get or create a task on Surge AI and return the Surge AI project ID.

        Attempt to find a Surge AI task inside this project for the fields.
        If one exists, reuse that task. Otherwise, create a new task inside the project using the fields.
        Return the Surge AI task ID."""
        project = surge.Project.retrieve(project_id)

        def create_surge_task():
            tasks = project.create_tasks([fields], launch=False)  # TODO: Make launch parameter configurable
            if len(tasks) != 1:
                return RuntimeError(f"Expected one task in Surge response, but got {len(tasks)} tasks")
            task = tasks[0]
            return {"id": task.id}

        with _surge_cache_lock:
            task_response, is_cached = self._cache.get({"project_id": project_id, "fields": fields}, create_surge_task)
        task_id = task_response["id"]
        if is_cached:
            hlog(f"Reusing existing Surge AI task: {task_id}")
        else:
            hlog(f"Creating new Surge AI task: {task_id}")
        return task_id

    def _get_worker_responses(
        self, task_id: str, questions: List[HumanQuestionTemplate]
    ) -> List[HumanTaskWorkerResponse]:
        task = surge.Task.retrieve(task_id)
        return [
            HumanTaskWorkerResponse(answers=[task_response.data[question.text] for question in questions])
            for task_response in task.responses
        ]

    def get_human_task(self, request: HumanTaskRequest) -> Optional[HumanTaskRequestResult]:
        """Create a human task on Surge AI and fetch responses from Surge AI if available.

        Returns HumanTaskRequestResult if worker answers are complete, or None otherwise.
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
        return HumanTaskRequestResult(workers=self._get_worker_responses(task_id, request.template.questions))
