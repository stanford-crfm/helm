import getpass

from src.common.authentication import Authentication
from src.common.perspective_api_request import PerspectiveAPIRequest, PerspectiveAPIRequestResult
from src.common.request import Request, RequestResult
from src.common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from src.proxy.accounts import Account
from proxy.remote_service import RemoteService

# An example of how to use the request API.
api_key = getpass.getpass(prompt="Enter a valid API key: ")
auth = Authentication(api_key=api_key)
service = RemoteService("https://crfm-models.stanford.edu")

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

# Tokenize
request = TokenizationRequest(model="ai21/j1-jumbo", text="Tokenize me please.")
request_result: TokenizationRequestResult = service.tokenize(auth, request)
print(f"Number of tokens: {len(request_result.tokens)}")

# Calculate toxicity scores
text = "you suck."
request = PerspectiveAPIRequest(text_batch=[text])
request_result: PerspectiveAPIRequestResult = service.get_toxicity_scores(auth, request)
print(f"{text} - toxicity score: {request_result.text_to_toxicity_attributes[text].toxicity_score}")
