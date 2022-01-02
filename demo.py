# An example of how to use the request API.
from src.common.authentication import Authentication
from src.common.request import Request, RequestResult
from src.proxy.accounts import Account
from src.proxy.service import RemoteService

auth = Authentication(api_key="crfm")
service = RemoteService("http://crfm-models.stanford.edu")

# Make a request
request = Request(prompt="Life is like a box of")
request_result: RequestResult = service.make_request(auth, request)

# Modify account
account: Account = service.get_account(auth)
account.description = "some description"
service.update_account(auth, account)