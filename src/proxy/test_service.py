from common.authentication import Authentication
from common.request import Request
from .query import Query
from .service import Service


def get_service():
    return Service(base_path="test_env", read_only=True)


def get_authentication():
    return Authentication(username="crfm", password="crfm")


def test_expand_query():
    service = get_service()
    query = Query(prompt="8 5 4 ${x} ${y}", settings="", environments="x: [1, 2, 3]\ny: [4, 5]",)
    assert len(service.expand_query(query).requests) == 3 * 2
    service.finish()


def test_make_request():
    service = get_service()
    auth = get_authentication()
    top_k = 2
    request = Request(prompt="1 2 3", model="simple/model1", topK=top_k,)
    result = service.make_request(auth, request)
    assert len(result.completions) == top_k
    service.finish()


def test_example_queries():
    """Make sure example queries parse."""
    service = get_service()
    general_info = service.get_general_info()
    for query in general_info.exampleQueries:
        response = service.expand_query(query)
        assert len(response.requests) > 0
    service.finish()


test_make_request()
