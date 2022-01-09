import os
import signal
from typing import List

from common.authentication import Authentication
from common.general import ensure_directory_exists, parse_hocon
from common.request import Request, RequestResult
from common.hierarchical_logger import hlog
from proxy.accounts import Accounts, Account
from proxy.auto_client import AutoClient
from proxy.example_queries import example_queries
from proxy.models import all_models, get_model_group
from proxy.query import Query, QueryResult
from proxy.service import (
    Service,
    CREDENTIALS_FILE,
    CACHE_DIR,
    ACCOUNTS_FILE,
    GeneralInfo,
    VERSION,
    expand_environments,
    synthesize_request,
)
from proxy.tokenizer.auto_token_counter import AutoTokenCounter


class ServerService(Service):
    """
    Main class that supports various functionality for the server.
    """

    def __init__(self, base_path: str = ".", read_only: bool = False):
        credentials_path = os.path.join(base_path, CREDENTIALS_FILE)
        cache_path = os.path.join(base_path, CACHE_DIR)
        ensure_directory_exists(cache_path)
        accounts_path = os.path.join(base_path, ACCOUNTS_FILE)

        if os.path.exists(credentials_path):
            with open(credentials_path) as f:
                credentials = parse_hocon(f.read())
        else:
            credentials = {}
        self.client = AutoClient(credentials, cache_path)
        self.token_counter = AutoTokenCounter()
        self.accounts = Accounts(accounts_path, read_only=read_only)

    def finish(self):
        self.accounts.finish()

    def get_general_info(self) -> GeneralInfo:
        return GeneralInfo(version=VERSION, example_queries=example_queries, all_models=all_models)

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
        # TODO: try to invoke the API even if we're not authenticated, and if
        #       it turns out the results are cached, then we can just hand back the results.
        #       https://github.com/stanford-crfm/benchmarking/issues/56

        self.accounts.authenticate(auth)
        model_group = get_model_group(request.model)
        # Make sure we can use
        self.accounts.check_can_use(auth.api_key, model_group)

        # Use!
        request_result = self.client.make_request(request)

        # Only deduct if not cached
        if not request_result.cached:
            # Count the number of tokens used
            count: int = self.token_counter.count_tokens(request, request_result.completions)
            hlog(f"{request.model_organization}: {count} tokens")
            self.accounts.use(auth.api_key, model_group, count)

        return request_result

    def create_account(self, auth: Authentication) -> Account:
        """Creates a new account."""
        return self.accounts.create_account(auth)

    def delete_account(self, auth: Authentication, api_key: str) -> Account:
        return self.accounts.delete_account(auth, api_key)

    def get_accounts(self, auth: Authentication) -> List[Account]:
        """Get list of accounts."""
        return self.accounts.get_all_accounts(auth)

    def get_account(self, auth: Authentication) -> Account:
        """Get information about an account."""
        return self.accounts.get_account(auth)

    def update_account(self, auth: Authentication, account: Account) -> Account:
        """Update account."""
        return self.accounts.update_account(auth, account)

    def rotate_api_key(self, auth: Authentication, account: Account) -> Account:
        """Generate a new API key for this account."""
        return self.accounts.rotate_api_key(auth, account)

    def shutdown(self, auth: Authentication):
        """Shutdown server (admin-only)."""
        self.accounts.check_admin(auth)

        pid = os.getpid()
        hlog(f"Shutting down server by killing its own process {pid}...")
        os.kill(pid, signal.SIGTERM)
        hlog("Done.")
