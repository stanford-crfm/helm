import copy
import pytest

from common.authentication import Authentication
from common.request import Request
from proxy.accounts import AuthenticationError
from .query import Query
from proxy.server_service import ServerService


def get_authentication():
    return Authentication(api_key="crfm")


class TestServerService:
    def setup_method(self, method):
        base_path = "test_env"
        self.service = ServerService(base_path=base_path, read_only=True)
        self.auth = get_authentication()

    def teardown_method(self, method):
        self.service.finish()

    def test_expand_query(self):
        query = Query(prompt="8 5 4 ${x} ${y}", settings="", environments="x: [1, 2, 3]\ny: [4, 5]",)
        assert len(self.service.expand_query(query).requests) == 3 * 2

    def test_make_request(self):
        num_completions = 2
        request = Request(prompt="1 2 3", model="simple/model1", num_completions=num_completions)
        result = self.service.make_request(self.auth, request)
        assert len(result.completions) == num_completions

    def test_example_queries(self):
        """Make sure example queries parse."""
        general_info = self.service.get_general_info()
        for query in general_info.example_queries:
            response = self.service.expand_query(query)
            assert len(response.requests) > 0

    def test_create_account(self):
        account = self.service.create_account(self.auth)
        assert account.api_key
        non_admin_auth = Authentication(account.api_key)

        # Only admins can create accounts
        raised = False
        try:
            self.service.create_account(non_admin_auth)
        except Exception as e:
            raised = True
            assert type(e) == AuthenticationError
        assert raised

    def test_get_accounts(self):
        account = self.service.create_account(self.auth)
        non_admin_auth = Authentication(account.api_key)

        accounts = self.service.get_accounts(self.auth)
        assert len(accounts) == 2

        # Only admins can get all accounts
        raised = False
        try:
            self.service.get_accounts(non_admin_auth)
        except Exception as e:
            raised = True
            assert type(e) == AuthenticationError
        assert raised

    def test_get_account(self):
        account = self.service.get_account(self.auth)
        assert account.api_key == "crfm"

        # Any user can access their own account
        account = self.service.create_account(self.auth)
        non_admin_auth = Authentication(account.api_key)
        self.service.get_account(non_admin_auth)

    def test_update_account(self):
        # Users can update their own account
        account = self.service.get_account(self.auth)
        account_copy = copy.deepcopy(account)
        account_copy.description = "new description"
        account = self.service.update_account(self.auth, account_copy)
        assert account.description == "new description"

        # Admin cannot update usage.used
        account_copy = copy.deepcopy(account)
        current_usage: int = account.usages["gpt3"]["daily"].used
        account_copy.usages["gpt3"]["daily"].used = -1
        account = self.service.update_account(self.auth, account_copy)
        assert account.usages["gpt3"]["daily"].used == current_usage

        # Non-admin users cannot promote themselves to admins
        account = self.service.create_account(self.auth)
        account_copy = copy.deepcopy(account)
        account_copy.is_admin = True
        non_admin_auth = Authentication(account.api_key)
        account = self.service.update_account(non_admin_auth, account_copy)
        assert not account.is_admin

        # Admins can make new admins though
        account = self.service.update_account(self.auth, account_copy)
        assert account.is_admin

    def test_rotate_api_key(self):
        # Admin can change other's API key
        account = self.service.create_account(self.auth)
        old_api_key = account.api_key
        non_admin_auth = Authentication(account.api_key)
        account = self.service.rotate_api_key(self.auth, account)
        assert account.api_key != old_api_key

        # Only admins can change API key for a user
        raised = False
        try:
            self.service.rotate_api_key(non_admin_auth, account)
        except Exception as e:
            raised = True
            assert type(e) == AuthenticationError
        assert raised


def get_prod_service():
    # Note that this is not checked in / available by default
    return ServerService(base_path="prod_env", read_only=True)


def helper_prod_test_service(request: Request, expected_text: str):
    """Make a `request` to the production server."""
    service = get_prod_service()
    auth = get_authentication()
    result = service.make_request(auth, request)
    print(result)
    assert result.success
    assert len(result.completions) == request.num_completions

    for completion in result.completions:
        # Make sure the token text builds the same as the top-level text
        assert "".join(token.text for token in completion.tokens) == completion.text

        # Check echo is working
        if request.echo_prompt:
            assert completion.text.startswith(request.prompt)

        # Don't generate too many tokens
        if not request.echo_prompt:
            assert len(completion.tokens) <= request.max_tokens

        # Consistency of log probs
        assert completion.logprob == sum(token.logprob for token in completion.tokens)

        # TODO: OpenAI's API returns null for the first token, so skip checking it; investigate this
        for token in completion.tokens[1:]:
            assert len(token.top_logprobs) == request.top_k_per_token

            # If generated token was one of the top, make sure has the right probability
            if token.text in token.top_logprobs:
                assert token.logprob == token.top_logprobs[token.text]

            # If temperature = 0, then make sure we're getting the top probability token
            if request.temperature == 0:
                assert token.text in token.top_logprobs
                assert token.logprob == max(token.top_logprobs.values())

    # Make sure we get the expected_text in one of the completions
    assert any(completion.text == expected_text for completion in result.completions)


# Models that we want to test
prod_models = ["openai/davinci", "ai21/j1-jumbo"]


# TODO: put a flag on this so that it's easy to use pytest to still run these slow tests
# https://www.py4u.net/discuss/204728
@pytest.mark.skip(reason="Requires production")
def test_prod_continue():
    # Test that we're continuing
    prompt = "Paris is the capital of"
    for model in prod_models:
        request = Request(prompt=prompt, model=model, max_tokens=1, num_completions=1, temperature=0)
        helper_prod_test_service(request, " France")


@pytest.mark.skip(reason="Requires production")
def test_prod_echo():
    # If we're echoing the prompt, make sure we're getting the same thing back
    prompt = "I like pickles."
    for model in prod_models:
        request = Request(prompt=prompt, model=model, max_tokens=0, num_completions=1, echo_prompt=True)
        helper_prod_test_service(request, prompt)
