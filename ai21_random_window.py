import getpass
import os.path

import torch
from tqdm import trange
from src.common.authentication import Authentication
from src.common.perspective_api_request import PerspectiveAPIRequest, PerspectiveAPIRequestResult
from src.common.request import Request, RequestResult
from src.common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from src.proxy.accounts import Account
from proxy.remote_service import RemoteService
import torch
import random

from pathlib import Path
import argparse

# An example of how to use the request API.
# api_key = getpass.getpass(prompt="Enter a valid API key: ")
api_key = "Tpw4JnmcG7Zn6vKdEjRcs2gnM2grlFDL"
auth = Authentication(api_key=api_key)
service = RemoteService("https://crfm-models.stanford.edu")
# Access account and show my current quotas and usages
account: Account = service.get_account(auth)
print(account.usages)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default="squad", choices=["ptb", "squad"])
args = parser.parse_args()

base_path = Path("../data")
if args.dataset_name == "ptb":
    file_name = "ptb_test.txt"
elif args.dataset_name == "squad":
    file_name = "squad_dev.json"
else:
    raise ValueError("Invalid Dataset Name.")
file_name = base_path / args.dataset_name / file_name

# Load PTB data
with open(file_name, "r") as f:
    text = f.read().strip()

# Very Raw Tokenize
text = text.split(" ")

# tokenziing first
tokens_cache_dir = base_path / args.dataset_name / "tokenized.pt"
if not os.path.exists(tokens_cache_dir):
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

    torch.save((all_result_tokens, all_result_detokens), tokens_cache_dir)
else:
    all_result_tokens, all_result_detokens = torch.load(tokens_cache_dir)

# build dictionary from token index to list of logprobs
# we are going to draw random window over the long text so some tokens can correspond to multiple logprobs
window_size = 1024
random_windows = []
random.seed(0)
all_logprobs = []
for rand_draw_i in trange(1000):
    rand_window_start = random.randint(0, len(all_result_tokens)-window_size)
    rand_window = all_result_detokens[rand_window_start:rand_window_start+window_size]
    request_text = "".join(rand_window)
    request = Request(model="ai21/j1-large", prompt=request_text, max_tokens=0, echo_prompt=True)
    request_result: RequestResult = service.make_request(auth, request)
    results = request_result.completions[0].tokens
    results = [{"text": t.text, "logprob": t.logprob, "top_logprobs": t.top_logprobs} for t in results]
    all_logprobs.append(results)
torch.save(all_logprobs, base_path / args.dataset_name / "logprobs.pt")