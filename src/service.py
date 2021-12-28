import os
import mako.template  # type: ignore
import pyhocon  # type: ignore
from general import ensure_directory_exists
from typing import Dict, List, Tuple, Any
from schemas import Query, QueryResult, Request, RequestResult, GeneralInfo
from models import all_models, get_model_group
from users import Authentication, Users, User

from auto_client import AutoClient
from example_queries import example_queries

VERSION = "1.0"
CREDENTIALS_FILE = "credentials.conf"
USERS_FILE = "users.jsonl"
CACHE_DIR = "cache"
MAX_EXPANSION = 1000


def parse_hocon(text: str):
    return pyhocon.ConfigFactory.parse_string(text)


def expand_environments(environments: Dict[str, List[str]]):
    """
    `environments` is a map from variable names to a list of strings.
    Return: a list of environments, where for each variable, we choose one of its string.
    """
    output_environments: List[Dict[str, str]] = []

    def recurse(old_items: List[Tuple[str, List[str]]], new_items: List[Tuple[str, str]]):
        if len(output_environments) >= MAX_EXPANSION:
            return
        if len(old_items) == 0:
            output_environments.append(dict(new_items))
        else:
            item, rest_old_items = old_items[0], old_items[1:]
            key, list_value = item
            for elem_value in list_value:
                recurse(rest_old_items, new_items + [(key, elem_value)])

    recurse(list(environments.items()), [])
    return output_environments


def substitute_text(text: str, environment: Dict[str, str]) -> str:
    """
    Example:
        text = "Hello {name}"
        environment = {"name": "Sue"}
        Return "Hello Sue"
    """
    return mako.template.Template(text).render(**environment)


def synthesize_request(prompt: str, settings: str, environment: Dict[str, str]) -> Request:
    """Substitute `environment` into `prompt` and `settings`."""
    request: Dict[str, Any] = {}
    request["prompt"] = substitute_text(prompt, environment)
    request.update(parse_hocon(substitute_text(settings, environment)))
    return Request(**request)


class Service(object):
    """
    Main class that supports various functionality for the server.
    """

    def __init__(self, base_path: str = ".", read_only: bool = False):
        credentials_path = os.path.join(base_path, CREDENTIALS_FILE)
        cache_path = os.path.join(base_path, CACHE_DIR)
        ensure_directory_exists(cache_path)
        users_path = os.path.join(base_path, USERS_FILE)

        if os.path.exists(credentials_path):
            with open(credentials_path) as f:
                credentials = parse_hocon(f.read())
        else:
            credentials = {}
        self.client = AutoClient(credentials, cache_path)
        self.users = Users(users_path, read_only=read_only)

    def finish(self):
        self.users.finish()

    def get_general_info(self):
        return GeneralInfo(version=VERSION, exampleQueries=example_queries, allModels=all_models)

    def expand_query(self, query: Query) -> QueryResult:
        """Turn the `query` into requests."""
        prompt = query.prompt
        settings = query.settings
        environments = parse_hocon(query.environments)
        requests = []
        for environment in expand_environments(environments):
            request = synthesize_request(prompt, settings, environment)
            requests.append(request)
        return QueryResult(requests=requests)

    def make_request(self, auth: Authentication, request: Request) -> RequestResult:
        """Actually make a request to an API."""
        self.users.authenticate(auth)
        model_group = get_model_group(request.model)
        # Make sure we can use
        self.users.check_can_use(auth.username, model_group)

        # Use!
        request_result = self.client.make_request(request)

        # Only deduct if not cached
        if not request_result.cached:
            # Estimate number of tokens (TODO: fix this)
            count = sum(len(completion.text.split(" ")) for completion in request_result.completions)
            self.users.use(auth.username, model_group, count)

        return request_result

    def get_user(self, auth: Authentication) -> User:
        """Get information about a user."""
        self.users.authenticate(auth)
        return self.users.username_to_users[auth.username]
