import getpass

from src.common.authentication import Authentication
from src.common.perspective_api_request import PerspectiveAPIRequest, PerspectiveAPIRequestResult
from src.common.request import Request, RequestResult, TokenEstimationRequestResult
from src.proxy.accounts import Account
from proxy.remote_service import RemoteService

# An example of how to use the request API.
# api_key = getpass.getpass(prompt="Enter a valid API key: ")
# auth = Authentication(api_key=api_key)
# service = RemoteService("http://crfm-models.stanford.edu")

auth = Authentication(api_key="crfm")
service = RemoteService("http://127.0.0.1:1959")

# Access account and show my current quotas and usages
account: Account = service.get_account(auth)
print(account.usages)

# Make a request
request = Request(prompt="Life is like a box of")
request_result: RequestResult = service.make_request(auth, request)
print(request_result.completions[0].text)

# Expect different responses for the same request but with different values for `random`.
# Passing in the same value for `random` guarantees the same results.
request = Request(prompt="Life is like a box of", random="1")
request_result: RequestResult = service.make_request(auth, request)
print(request_result.completions[0].text)

# Estimate the number of tokens for a request
request = Request(prompt="Life is like a box of")
request_result: TokenEstimationRequestResult = service.estimate_tokens(request)
print(request_result.num_tokens)

# Calculate toxicity scores
text = "you suck."
request = PerspectiveAPIRequest(text_batch=[text])
request_result: PerspectiveAPIRequestResult = service.get_toxicity_scores(auth, request)
print(f"{text} - toxicity score: {request_result.text_to_toxicity_attributes[text].toxicity_score}")
