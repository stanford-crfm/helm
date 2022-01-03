import getpass

from src.common.authentication import Authentication
from src.common.request import Request, RequestResult
from src.proxy.accounts import Account
from src.proxy.service import RemoteService

# An example of how to use the request API.
api_key = getpass.getpass(prompt="Enter a valid API key: ")
auth = Authentication(api_key=api_key)
service = RemoteService("http://crfm-models.stanford.edu")
service = RemoteService("http://127.0.0.1:1959")

# Access account
account: Account = service.get_account(auth)
print(account)

# Make a request
request = Request(prompt="Life is like a box of")
request_result: RequestResult = service.make_request(auth, request)
print(request_result)