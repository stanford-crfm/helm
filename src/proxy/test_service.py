import pytest
from common.authentication import Authentication
from common.request import Request
from .query import Query
from .service import Service


def get_test_service():
    return Service(base_path="test_env", read_only=True)


def get_prod_service():
    # Note that this is not checked in / available by default
    return Service(base_path="prod_env", read_only=True)


def get_authentication():
    return Authentication(username="crfm", password="crfm")


def test_expand_query():
    service = get_test_service()
    query = Query(prompt="8 5 4 ${x} ${y}", settings="", environments="x: [1, 2, 3]\ny: [4, 5]",)
    assert len(service.expand_query(query).requests) == 3 * 2
    service.finish()


def test_make_request():
    service = get_test_service()
    auth = get_authentication()
    num_completions = 2
    request = Request(prompt="1 2 3", model="simple/model1", num_completions=num_completions)
    result = service.make_request(auth, request)
    assert len(result.completions) == num_completions
    service.finish()


def test_example_queries():
    """Make sure example queries parse."""
    service = get_test_service()
    general_info = service.get_general_info()
    for query in general_info.example_queries:
        response = service.expand_query(query)
        assert len(response.requests) > 0
    service.finish()


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
