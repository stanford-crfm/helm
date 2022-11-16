import os
import signal
from typing import List

from helm.common.authentication import Authentication
from helm.common.general import ensure_directory_exists, parse_hocon
from helm.common.perspective_api_request import PerspectiveAPIRequest, PerspectiveAPIRequestResult
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from helm.common.request import Request, RequestResult
from helm.common.hierarchical_logger import hlog
from helm.proxy.accounts import Accounts, Account
from helm.proxy.clients.auto_client import AutoClient
from helm.proxy.example_queries import example_queries
from helm.proxy.models import ALL_MODELS, get_model_group
from helm.proxy.query import Query, QueryResult
from helm.proxy.retry import retry_request
from helm.proxy.token_counters.auto_token_counter import AutoTokenCounter
from .service import (
    Service,
    CREDENTIALS_FILE,
    CACHE_DIR,
    ACCOUNTS_FILE,
    GeneralInfo,
    VERSION,
    expand_environments,
    synthesize_request,
)


class ServerService(Service):
    """
    Main class that supports various functionality for the server.
    """

    def __init__(self, base_path: str = ".", root_mode=False, mongo_uri: str = ""):
        credentials_path = os.path.join(base_path, CREDENTIALS_FILE)
        cache_path = os.path.join(base_path, CACHE_DIR)
        ensure_directory_exists(cache_path)
        accounts_path = os.path.join(base_path, ACCOUNTS_FILE)

        if os.path.exists(credentials_path):
            with open(credentials_path) as f:
                credentials = parse_hocon(f.read())
        else:
            credentials = {}

        self.client = AutoClient(credentials, cache_path, mongo_uri)
        self.token_counter = AutoTokenCounter(self.client.huggingface_client)
        self.accounts = Accounts(accounts_path, root_mode=root_mode)
        self.perspective_api_client = self.client.get_toxicity_classifier_client()

    def get_general_info(self) -> GeneralInfo:
        return GeneralInfo(version=VERSION, example_queries=example_queries, all_models=ALL_MODELS)

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
        model_group: str = get_model_group(request.model)
        # Make sure we can use
        self.accounts.check_can_use(auth.api_key, model_group)

        # Use!
        request_result: RequestResult = self.client.make_request(request)

        # Only deduct if not cached
        if not request_result.cached:
            # Count the number of tokens used
            count: int = self.token_counter.count_tokens(request, request_result.completions)
            self.accounts.use(auth.api_key, model_group, count)

        return request_result

    def tokenize(self, auth: Authentication, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenize via an API."""
        self.accounts.authenticate(auth)
        return self.client.tokenize(request)

    def decode(self, auth: Authentication, request: DecodeRequest) -> DecodeRequestResult:
        """Decodes to text."""
        self.accounts.authenticate(auth)
        return self.client.decode(request)

    def get_toxicity_scores(self, auth: Authentication, request: PerspectiveAPIRequest) -> PerspectiveAPIRequestResult:
        @retry_request
        def get_toxicity_scores_with_retry(request: PerspectiveAPIRequest) -> PerspectiveAPIRequestResult:
            return self.perspective_api_client.get_toxicity_scores(request)

        self.accounts.authenticate(auth)
        return get_toxicity_scores_with_retry(request)

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
