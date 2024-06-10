from helm.clients.simple_client import SimpleClient
from helm.common.cache import BlackHoleCacheConfig
from helm.common.request import GeneratedOutput, Request, Token


def test_simple_client_make_request():
    client = SimpleClient(BlackHoleCacheConfig())
    request = Request(
        model="simple/model1",
        model_deployment="simple/model1",
        prompt="Elephants are one of the most",
        temperature=0.0,
        max_tokens=10,
    )
    result = client.make_request(request)
    assert result.success
    assert not result.cached
    assert result.embedding == []
    assert result.completions == [GeneratedOutput(text="most", logprob=0, tokens=[Token(text="most", logprob=0)])]
