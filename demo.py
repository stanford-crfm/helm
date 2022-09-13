import getpass
import sys
import pandas as pd

# sys.path.append('/juice/scr/katezhou/benchmarking')
# sys.path.append('/juice/scr/katezhou/benchmarking/src')
# sys.path.append('/juice/scr/katezhou/benchmarking/src/common')

from src.common.authentication import Authentication
from src.common.perspective_api_request import PerspectiveAPIRequest, PerspectiveAPIRequestResult
from src.common.request import Request, RequestResult
from src.common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from src.proxy.accounts import Account
from proxy.services.remote_service import RemoteService

# An example of how to use the request API.
#api_key = getpass.getpass(prompt="Enter a valid API key: ")
api_key = pd.read_csv("prod_env/api_key.csv", header=None)[0].values[0]
auth = Authentication(api_key=api_key)
service = RemoteService("https://crfm-models.stanford.edu")

# Access account and show my current quotas and usages
# account: Account = service.get_account(auth)
# print(account.usages)

print()
print("----")
print()

# Make a request
request = Request(prompt="Question: What is the numerical probability of \"toss-up\"? Answer: 50%. Question: What is the numerical probability of \"possibly\"? Answer:", temperature=0.7, model="openai/text-davinci-001", num_completions=100, max_tokens=50)
request_result: RequestResult = service.make_request(auth, request)
    
for x in request_result.completions:
    print(x.text)
    print("----")
print()
print("----")
print()

# Calculate toxicity scores
text = "you suck."
request = PerspectiveAPIRequest(text_batch=[text])
request_result: PerspectiveAPIRequestResult = service.get_toxicity_scores(auth, request)
print(f"{text} - toxicity score: {request_result.text_to_toxicity_attributes[text].toxicity_score}")

# Get-embedding example
request = Request(model="openai/ada", prompt="Life is like a box of", echo_prompt=True, embedding=True)
