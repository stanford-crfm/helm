import os
import signal
from typing import List

from helm.common.local_context import LocalContext
from helm.common.cache import CacheConfig
from helm.common.cache_backend_config import CacheBackendConfig, BlackHoleCacheBackendConfig
from helm.common.critique_request import CritiqueRequest, CritiqueRequestResult
from helm.common.authentication import Authentication
from helm.common.moderations_api_request import ModerationAPIRequest, ModerationAPIRequestResult
from helm.common.clip_score_request import CLIPScoreRequest, CLIPScoreResult
from helm.common.nudity_check_request import NudityCheckRequest, NudityCheckResult
from helm.common.file_upload_request import FileUploadRequest, FileUploadResult
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
from helm.benchmark.model_deployment_registry import get_model_deployment_host_organization
from helm.proxy.query import Query, QueryResult
from helm.proxy.token_counters.auto_token_counter import AutoTokenCounter
from helm.proxy.services.service import (
    Service,
    ACCOUNTS_FILE,
    GeneralInfo,
)


class ServerService(Service):
    """
    Main class that supports various functionality for the server.
    """

    def __init__(
        self,
        base_path: str = "prod_env",
        root_mode: bool = False,
        cache_backend_config: CacheBackendConfig = BlackHoleCacheBackendConfig(),
    ):
        accounts_path = os.path.join(base_path, ACCOUNTS_FILE)

        self.context = LocalContext(base_path, cache_backend_config)
        self.token_counter = AutoTokenCounter(self.context.tokenizer)
        self.accounts = Accounts(accounts_path, root_mode=root_mode)

    def get_general_info(self) -> GeneralInfo:
        return self.context.get_general_info()

    def expand_query(self, query: Query) -> QueryResult:
        return self.context.expand_query(query)

    def _get_model_group_for_model_deployment(self, model_deployment: str) -> str:
        if model_deployment.startswith("openai/"):
            if model_deployment.startswith("openai/code-"):
                return "codex"
            elif model_deployment.startswith("openai/dall-e-"):
                return "dall_e"
            elif model_deployment.startswith("openai/gpt-4"):
                return "gpt4"
            elif model_deployment.startswith("openai/gpt-3"):
                return "gpt3"
            elif (
                model_deployment.startswith("openai/o1")
                or model_deployment.startswith("openai/o3")
                or model_deployment.startswith("openai/o4")
            ):
                return "o1"
            else:
                return "openai"
        elif model_deployment.startswith("ai21/"):
            return "jurassic"
        else:
            return get_model_deployment_host_organization(model_deployment)

    def make_request(self, auth: Authentication, request: Request) -> RequestResult:
        """Actually make a request to an API."""
        # TODO: try to invoke the API even if we're not authenticated, and if
        #       it turns out the results are cached, then we can just hand back the results.
        #       https://github.com/stanford-crfm/benchmarking/issues/56

        self.accounts.authenticate(auth)
        model_group: str = self._get_model_group_for_model_deployment(request.model_deployment)
        # Make sure we can use
        self.accounts.check_can_use(auth.api_key, model_group)

        # Use!
        request_result: RequestResult = self.context.make_request(request)

        # Only deduct if not cached
        if not request_result.cached:
            # Count the number of tokens used
            count: int = self.token_counter.count_tokens(request, request_result.completions)
            self.accounts.use(auth.api_key, model_group, count)

        return request_result

    def tokenize(self, auth: Authentication, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenize via an API."""
        self.accounts.authenticate(auth)
        return self.context.tokenize(request)

    def decode(self, auth: Authentication, request: DecodeRequest) -> DecodeRequestResult:
        """Decodes to text."""
        self.accounts.authenticate(auth)
        return self.context.decode(request)

    def upload(self, auth: Authentication, request: FileUploadRequest) -> FileUploadResult:
        """Uploads a file to external storage."""
        self.accounts.authenticate(auth)
        return self.context.upload(request)

    def check_nudity(self, auth: Authentication, request: NudityCheckRequest) -> NudityCheckResult:
        """Check for nudity."""
        self.accounts.authenticate(auth)
        return self.context.check_nudity(request)

    def compute_clip_score(self, auth: Authentication, request: CLIPScoreRequest) -> CLIPScoreResult:
        """Computes CLIPScore for a given caption and image."""
        self.accounts.authenticate(auth)
        return self.context.compute_clip_score(request)

    def get_toxicity_scores(self, auth: Authentication, request: PerspectiveAPIRequest) -> PerspectiveAPIRequestResult:
        self.accounts.authenticate(auth)
        return self.context.get_toxicity_scores(request)

    def get_moderation_results(self, auth: Authentication, request: ModerationAPIRequest) -> ModerationAPIRequestResult:
        self.accounts.authenticate(auth)
        return self.context.get_moderation_results(request)

    def make_critique_request(self, auth: Authentication, request: CritiqueRequest) -> CritiqueRequestResult:
        self.accounts.authenticate(auth)
        return self.context.make_critique_request(request)

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

    def get_cache_config(self, shard_name: str) -> CacheConfig:
        return self.context.get_cache_config(shard_name)
