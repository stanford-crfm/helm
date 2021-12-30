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

# TODO: put a flag on this so that it's easy to use pytest to still run these slow tests
# https://www.py4u.net/discuss/204728
@pytest.mark.skip(reason="Requires production")
def test_openai_client1():
    service = get_prod_service()
    auth = get_authentication()
    request = Request(prompt="Paris is the capital of", model="openai/davinci", max_tokens=1, num_completions=1, temperature=0)
    result = service.make_request(auth, request)
    assert result.success
    assert len(result.completions) == 1
    assert result.completions[0].text == " France"

@pytest.mark.skip(reason="Requires production")
def test_openai_client2():
    service = get_prod_service()
    auth = get_authentication()
    request = Request(prompt="I like cheese.", model="openai/davinci", max_tokens=0, num_completions=1, temperature=0, echo_prompt=True)
    result = service.make_request(auth, request)
    assert result.success
    assert len(result.completions) == 1
    assert result.completions[0].text == "I like cheese."
