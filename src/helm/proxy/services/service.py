import mako.template
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from helm.common.general import parse_hocon
from helm.common.critique_request import CritiqueRequest, CritiqueRequestResult
from helm.common.perspective_api_request import PerspectiveAPIRequestResult, PerspectiveAPIRequest
from helm.common.tokenization_request import (
    WindowServiceInfo,
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from helm.common.request import Request, RequestResult
from helm.proxy.models import Model
from helm.proxy.query import Query, QueryResult
from helm.proxy.accounts import Authentication, Account

VERSION = "1.0"
CREDENTIALS_FILE = "credentials.conf"
ACCOUNTS_FILE = "accounts.sqlite"
CACHE_DIR = "cache"
MONGO_URI = "mongo_uri"
MAX_EXPANSION = 1000


@dataclass(frozen=True)
class GeneralInfo:
    version: str
    example_queries: List[Query]
    all_models: List[Model]


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


class Service(ABC):
    @abstractmethod
    def get_general_info(self) -> GeneralInfo:
        """Get general info."""
        pass

    @abstractmethod
    def get_window_service_info(self, model_name: str) -> WindowServiceInfo:
        """Get window service info."""
        pass

    @abstractmethod
    def expand_query(self, query: Query) -> QueryResult:
        """Turn the `query` into requests."""
        pass

    @abstractmethod
    def make_request(self, auth: Authentication, request: Request) -> RequestResult:
        """Actually make a request to an API."""
        pass

    @abstractmethod
    def tokenize(self, auth: Authentication, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenize via an API."""
        pass

    @abstractmethod
    def decode(self, auth: Authentication, request: DecodeRequest) -> DecodeRequestResult:
        """Decodes to text."""
        pass

    @abstractmethod
    def get_toxicity_scores(self, auth: Authentication, request: PerspectiveAPIRequest) -> PerspectiveAPIRequestResult:
        """Get toxicity scores for a batch of text."""
        pass

    def make_critique_request(self, auth: Authentication, request: CritiqueRequest) -> CritiqueRequestResult:
        """Get responses to a critique request."""
        pass

    @abstractmethod
    def create_account(self, auth: Authentication) -> Account:
        """Creates a new account."""
        pass

    @abstractmethod
    def delete_account(self, auth: Authentication, api_key: str) -> Account:
        """Deletes an account."""
        pass

    @abstractmethod
    def get_accounts(self, auth: Authentication) -> List[Account]:
        """Get list of accounts."""
        pass

    @abstractmethod
    def get_account(self, auth: Authentication) -> Account:
        """Get information about an account."""
        pass

    @abstractmethod
    def update_account(self, auth: Authentication, account: Account) -> Account:
        """Update account."""
        pass

    @abstractmethod
    def rotate_api_key(self, auth: Authentication, account: Account) -> Account:
        """Generate a new API key for a given account."""
        pass

    @abstractmethod
    def shutdown(self, auth: Authentication):
        """Shutdown server."""
        pass
