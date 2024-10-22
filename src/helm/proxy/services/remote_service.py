import argparse
import json
import requests
import urllib.parse
from dataclasses import asdict
from typing import Any, List, Optional

from helm.common.cache import CacheConfig
from helm.common.cache_backend_config import BlackHoleCacheBackendConfig
from helm.common.authentication import Authentication
from helm.common.moderations_api_request import ModerationAPIRequest, ModerationAPIRequestResult
from helm.common.critique_request import CritiqueRequest, CritiqueRequestResult
from helm.common.nudity_check_request import NudityCheckRequest, NudityCheckResult
from helm.common.file_upload_request import FileUploadRequest, FileUploadResult
from helm.common.perspective_api_request import PerspectiveAPIRequest, PerspectiveAPIRequestResult
from helm.common.clip_score_request import CLIPScoreRequest, CLIPScoreResult
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequestResult,
    DecodeRequest,
)
from helm.common.request import Request, RequestResult
from dacite import from_dict
from helm.proxy.accounts import Account
from helm.proxy.query import Query, QueryResult
from helm.proxy.services.service import Service, GeneralInfo


class RemoteServiceError(Exception):
    pass


class RemoteService(Service):
    NOT_SUPPORTED_ERROR: str = "Not supported through the remote service."

    def __init__(self, base_url):
        self.base_url: str = base_url

    @staticmethod
    def _check_response(response: Any, request: Optional[str] = None):
        if type(response) is dict and "error" in response and response["error"]:
            error_message: str = response["error"]
            if request:
                error_message += f" Request: {request}"

            raise RemoteServiceError(error_message)

    def get_general_info(self) -> GeneralInfo:
        response = requests.get(f"{self.base_url}/api/general_info").json()
        return from_dict(GeneralInfo, response)

    def expand_query(self, query: Query) -> QueryResult:
        params = asdict(query)
        response = requests.get(f"{self.base_url}/api/query?{urllib.parse.urlencode(params)}").json()
        RemoteService._check_response(response)
        return from_dict(QueryResult, response)

    def make_request(self, auth: Authentication, request: Request) -> RequestResult:
        request_json: str = json.dumps(asdict(request))
        params = {
            "auth": json.dumps(asdict(auth)),
            "request": request_json,
        }
        response = requests.get(f"{self.base_url}/api/request?{urllib.parse.urlencode(params)}").json()
        RemoteService._check_response(response, request_json)
        return from_dict(RequestResult, response)

    def tokenize(self, auth: Authentication, request: TokenizationRequest) -> TokenizationRequestResult:
        request_json: str = json.dumps(asdict(request))
        params = {
            "auth": json.dumps(asdict(auth)),
            "request": request_json,
        }
        response = requests.get(f"{self.base_url}/api/tokenize?{urllib.parse.urlencode(params)}").json()
        RemoteService._check_response(response, request_json)
        return from_dict(TokenizationRequestResult, response)

    def decode(self, auth: Authentication, request: DecodeRequest) -> DecodeRequestResult:
        request_json: str = json.dumps(asdict(request))
        params = {
            "auth": json.dumps(asdict(auth)),
            "request": request_json,
        }
        response = requests.get(f"{self.base_url}/api/decode?{urllib.parse.urlencode(params)}").json()
        RemoteService._check_response(response, request_json)
        return from_dict(DecodeRequestResult, response)

    def upload(self, auth: Authentication, request: FileUploadRequest) -> FileUploadResult:
        raise NotImplementedError(self.NOT_SUPPORTED_ERROR)

    def check_nudity(self, auth: Authentication, request: NudityCheckRequest) -> NudityCheckResult:
        raise NotImplementedError(self.NOT_SUPPORTED_ERROR)

    def compute_clip_score(self, auth: Authentication, request: CLIPScoreRequest) -> CLIPScoreResult:
        raise NotImplementedError(self.NOT_SUPPORTED_ERROR)

    def get_toxicity_scores(self, auth: Authentication, request: PerspectiveAPIRequest) -> PerspectiveAPIRequestResult:
        request_json: str = json.dumps(asdict(request))
        params = {
            "auth": json.dumps(asdict(auth)),
            "request": request_json,
        }
        response = requests.get(f"{self.base_url}/api/toxicity?{urllib.parse.urlencode(params)}").json()
        RemoteService._check_response(response, request_json)
        return from_dict(PerspectiveAPIRequestResult, response)

    def get_moderation_results(self, auth: Authentication, request: ModerationAPIRequest) -> ModerationAPIRequestResult:
        request_json: str = json.dumps(asdict(request))
        params = {
            "auth": json.dumps(asdict(auth)),
            "request": request_json,
        }
        response = requests.get(f"{self.base_url}/api/moderation?{urllib.parse.urlencode(params)}").json()
        RemoteService._check_response(response, request_json)
        return from_dict(ModerationAPIRequestResult, response)

    def make_critique_request(self, auth: Authentication, request: CritiqueRequest) -> CritiqueRequestResult:
        raise NotImplementedError("make_critique_request is not supported by RemoteServer")

    def create_account(self, auth: Authentication) -> Account:
        data = {"auth": json.dumps(asdict(auth))}
        response = requests.post(f"{self.base_url}/api/account", data=data).json()
        RemoteService._check_response(response)
        return from_dict(Account, response)

    def delete_account(self, auth: Authentication, api_key: str) -> Account:
        data = {
            "auth": json.dumps(asdict(auth)),
            "api_key": api_key,
        }
        response = requests.delete(f"{self.base_url}/api/account", data=data).json()
        RemoteService._check_response(response)
        return from_dict(Account, response)

    def get_accounts(self, auth: Authentication) -> List[Account]:
        params = {"auth": json.dumps(asdict(auth)), "all": "true"}
        response = requests.get(f"{self.base_url}/api/account?{urllib.parse.urlencode(params)}").json()
        RemoteService._check_response(response)
        return [from_dict(Account, account_response) for account_response in response]

    def get_account(self, auth: Authentication) -> Account:
        params = {"auth": json.dumps(asdict(auth))}
        response = requests.get(f"{self.base_url}/api/account?{urllib.parse.urlencode(params)}").json()
        RemoteService._check_response(response)
        return from_dict(Account, response[0])

    def update_account(self, auth: Authentication, account: Account) -> Account:
        data = {
            "auth": json.dumps(asdict(auth)),
            "account": json.dumps(asdict(account)),
        }
        response = requests.put(f"{self.base_url}/api/account", data=data).json()
        RemoteService._check_response(response)
        return from_dict(Account, response)

    def rotate_api_key(self, auth: Authentication, account: Account) -> Account:
        """Generate a new API key for this account."""
        data = {
            "auth": json.dumps(asdict(auth)),
            "account": json.dumps(asdict(account)),
        }
        response = requests.put(f"{self.base_url}/api/account/api_key", data=data).json()
        RemoteService._check_response(response)
        return from_dict(Account, response)

    def shutdown(self, auth: Authentication):
        """Shutdown server (admin-only)."""
        params = {"auth": json.dumps(asdict(auth))}
        try:
            response = requests.get(f"{self.base_url}/api/shutdown?{urllib.parse.urlencode(params)}").json()
            RemoteService._check_response(response)
        except requests.exceptions.ConnectionError:
            # A ConnectionError is expected when shutting down the server.
            pass

    def get_cache_config(self, shard_name: str) -> CacheConfig:
        """Returns a CacheConfig"""
        return BlackHoleCacheBackendConfig().get_cache_config(shard_name)


def add_service_args(parser: argparse.ArgumentParser):
    """Add command-line arguments to enable command-line utilities to specify how to connect to a remote server."""
    parser.add_argument("--server-url", type=str, default=None, help="URL of proxy server to connect to")
    parser.add_argument(
        "--api-key-path", type=str, default="proxy_api_key.txt", help="Path to a file containing the API key"
    )


def create_authentication(args) -> Authentication:
    with open(args.api_key_path) as f:
        api_key = f.read().strip()
    return Authentication(api_key=api_key)
