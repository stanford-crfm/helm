from abc import ABC, abstractmethod

import random
from helm.common.hierarchical_logger import hlog
from helm.common.cache import Cache, CacheConfig
from helm.common.human_task_request import (
    HumanQuestionTemplate,
    HumanTaskRequest,
    HumanTaskRequestResult,
    HumanTaskTemplate,
)
from cattrs import unstructure
import threading

from typing import Dict, List
import surge
from surge import questions as surge_questions


class HumanTaskClient(ABC):
    @abstractmethod
    def get_human_task(self, request: HumanTaskRequest) -> HumanTaskRequestResult:
        pass


class RandomHumanTaskClient(HumanTaskClient):
    """A HumanTaskClient that returns random choices for debugging."""

    def get_human_task(self, request: HumanTaskRequest) -> HumanTaskRequestResult:
        answers: List[List[str]] = []
        for question in request.template.questions:
            if question.question_type != "multiple_choice":
                raise ValueError("Currently, only multiple_choice questions are supported")
            answers.append([random.choice(question.options) for _ in range(request.template.num_workers)])
        return HumanTaskRequestResult(answers=answers)


_surge_cache_lock = threading.Lock()


class SurgeAIHumanTaskClient(HumanTaskClient):
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

    def _get_task_responses(self, task_id: str, questions: List[HumanQuestionTemplate]) -> List[List[str]]:
        task = surge.Task.retrieve(task_id)
        return [[task_response.data[question.text] for task_response in task.responses] for question in questions]

    def get_human_task(self, request: HumanTaskRequest) -> HumanTaskRequestResult:
        project_id = self._get_or_create_surge_project(request.template)
        task_id = self._get_or_create_task(project_id, request.fields)
        return HumanTaskRequestResult(answers=self._get_task_responses(task_id, request.template.questions))
