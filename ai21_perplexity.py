import getpass
import os.path

import torch
from tqdm import trange
from src.common.authentication import Authentication
from src.common.perspective_api_request import PerspectiveAPIRequest, PerspectiveAPIRequestResult
from src.common.request import Request, RequestResult
from src.common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from src.proxy.accounts import Account
from proxy.services.remote_service import RemoteService
import torch
import random

# An example of how to use the request API.
# api_key = getpass.getpass(prompt="Enter a valid API key: ")
api_key = "Tpw4JnmcG7Zn6vKdEjRcs2gnM2grlFDL"
auth = Authentication(api_key=api_key)
service = RemoteService("https://crfm-models.stanford.edu")

# Access account and show my current quotas and usages
account: Account = service.get_account(auth)
print(account.usages)

# Load PTB data
with open("../data/ptb/ptb_test.txt", "r") as f:
    text = f.read().strip()

# Very Raw Tokenize
text = text.split(" ")

# moving window approach
# all_result_tokens = []
# for request_start in trange(0, len(text), request_stride):
#     request_text = " ".join(text[request_start:request_start+1024])
#     request = Request(model="ai21/j1-large", prompt=request_text, max_tokens=0, echo_prompt=True)
#     request_result: RequestResult = service.make_request(auth, request)
#     tokens = request_result.completions[0].tokens
#     tokens = [{"text": t.text, "logprob": t.logprob, "top_logprobs": t.top_logprobs} for t in tokens]
#     all_result_tokens.append(tokens)
#
# torch.save(all_result_tokens, "../data/results/j1_grande_ptb.pt")

# tokenziing first
if not os.path.exists("../data/ptb_test_tokens.pt"):
    request_stride = 512
    all_result_tokens = []
    all_result_detokens = []
    for request_start in trange(0, len(text), request_stride):
        request_text = " ".join(text[request_start:request_start+request_stride])
        request = TokenizationRequest(model="ai21/j1-jumbo", text=request_text)
        request_result: TokenizationRequestResult = service.tokenize(auth, request)
        tokens = request_result.tokens
        detoks = []
        for tok in tokens:
            text_range = tok.text_range
            start = text_range.start
            end = text_range.end
            detoks.append(
                request_text[start:end]
            )
        all_result_tokens += tokens
        all_result_detokens += detoks

    torch.save((all_result_tokens, all_result_detokens), "../data/ptb_test_tokens.pt")
else:
    all_result_tokens, all_result_detokens = torch.load("../data/ptb_test_tokens.pt")

# build dictionary from token index to list of logprobs
# we are going to draw random window over the long text so some tokens can correspond to multiple logprobs
window_size = 1024
random.seed(0)
for rand_draw_i in range(100):
    rand_window_start = random.randint(0, len(all_result_tokens)-window_size)
    rand_window = all_result_detokens[rand_window_start:rand_window_start+window_size]
    request_text = "".join(rand_window)
    request = Request(model="ai21/j1-large", prompt=request_text, max_tokens=0, echo_prompt=True)
    request_result: RequestResult = service.make_request(auth, request)
    tokens = request_result.completions[0].tokens
    tokens = [{"text": t.text, "logprob": t.logprob, "top_logprobs": t.top_logprobs} for t in tokens]
    text_tokens = [t["text"] for t in tokens]
    for tok in tokens:
        if len(tok["text"]) == 0:
            print("Encountering emtpy token")
            print(tok)
    breakpoint()
    for text_tok, detok in zip(text_tokens, rand_window):
        if text_tok != detok:
            print(text_tok, detok)
            breakpoint()
    breakpoint()
