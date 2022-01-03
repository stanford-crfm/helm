import json
import requests
import urllib.parse
from dataclasses import asdict
from typing import Any, List

from common.authentication import Authentication
from common.request import Request, RequestResult
from dacite import from_dict
from proxy.accounts import Account
from proxy.query import Query, QueryResult
from proxy.service import Service, GeneralInfo


class RemoteServiceError(Exception):
    pass


class RemoteService(Service):
    def __init__(self, base_url="http://crfm-models.stanford.edu"):
        self.base_url = base_url

    @staticmethod
    def _check_response(response: Any):
        if type(response) is dict and "error" in response and response["error"]:
            raise RemoteServiceError(response["error"])

    def get_general_info(self) -> GeneralInfo:
        response = requests.get(f"{self.base_url}/api/general_info").json()
        return from_dict(GeneralInfo, response)

    def expand_query(self, query: Query) -> QueryResult:
        params = asdict(query)
        response = requests.get(f"{self.base_url}/api/query?{urllib.parse.urlencode(params)}").json()
        RemoteService._check_response(response)
        return from_dict(QueryResult, response)

    def make_request(self, auth: Authentication, request: Request) -> RequestResult:
        params = {
            "auth": json.dumps(asdict(auth)),
            "request": json.dumps(asdict(request)),
        }
        response = requests.get(f"{self.base_url}/api/request?{urllib.parse.urlencode(params)}").json()
        RemoteService._check_response(response)
        return from_dict(RequestResult, response)

    def create_account(self, auth: Authentication) -> Account:
        data = {"auth": json.dumps(asdict(auth))}
        response = requests.post(f"{self.base_url}/api/account", data=data).json()
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
        return response

    def rotate_api_key(self, auth: Authentication, account: Account) -> Account:
        """Generate a new API key for this account."""
        data = {
            "auth": json.dumps(asdict(auth)),
            "account": json.dumps(asdict(account)),
        }
        response = requests.put(f"{self.base_url}/api/account/api_key", data=data).json()
        RemoteService._check_response(response)
        return response
