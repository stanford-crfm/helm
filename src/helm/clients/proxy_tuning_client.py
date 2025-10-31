"""
Proxy-tuned HELM client
=======================

This module implements a HELM Client that routes generation through
decoding-time strategies for domain-level adaptation. 
It runs multiple models (base, expert, anti-expert).
This is experimental code to test different decoding-time strategies. 

Main classes
------------
- AnyModel: Imported from any_model.py in proxy_tuning directory. Performs step-wise generation under three modes:
  1) **Base** (single base model, runs using HF generate() function),
  2) **Unite** (merge base + expert via vocabulary union arithmetic):
    - adapted in from [this codebase](https://github.com/starrYYxuan/UniTE/)
  3) **Proxy**:
     - Original method adapted from [this codebase](https://github.com/alisawuffles/proxy-tuning/tree/main):
         - base + alpha(expert − anti-expert) at the logit level with models of same vocabulary.
     - Cross-Architecture Proxy Tuning (our novel method) 
         - same formula as above using log-probs with models of differing vocabulary

- ProxyTuningClient: A HELM client that parses the deployment tag to
  configure `AnyModel`, runs generation for a given `Request`, and logs
  per-request outputs and token-wise outputs for the proxy tuning method.

Deployment/tag format
---------------------
The `model_name` (a.k.a. deployment tag) is expected to be of the form:
    "proxy_tuning/{base}_{expert}_{antiexpert}_{alpha}_{score_type}_{k}"

Examples:
    proxy_tuning/mellama-13b-chat_none_none_1.0_logits_20 (base)         
    proxy_tuning/qwen3-30b_mellama-13b-chat_none_1.0_logprobs_20 (unite)
    proxy_tuning/llama-70b-chat_mellama-13b-chat_llama-13b-base_0.7_logits_10 (Original proxy, logits) 
    proxy_tuning/qwen3-30b_mellama-13b-chat_llama-13b-base_0.7_logits_10 (CAPT proxy, logprobs) 

Each sub-tag meaning:
- base / expert / antiexpert: keys that must exist in `MODEL_PATHS` below
  (use "none" to disable that role).
  - if only base is not "none" --> base method
  - if base and expert are not "none" --> unite method
  - if base, expert, and antiexpert are not "none"  --> proxy method
- alpha: float, strength of expert vs anti-expert adjustment.
- score_type: "logits" (original proxy tuning) or "logprobs" (CAPT).
- k: top-k token pool size when building the union vocabulary (for Unite
  and CAPT).

Artifacts
---------
A results directory is created under:

    LOCAL_RESULTS_DIR/<safe_tag>_<YYYYMMDD_HHMMSS>/

Files inside:
- `<safe_tag>_<stamp>.csv`  : One row per HELM request with columns:
    timestamp, request_id, model_name, prompt, output, logits_path
- `logits_analysis/`        : Optional per-request tensors (when
  `return_logits_for_analysis=True`) saved via `torch.save(...)` as:
    logits_<runid>_r####.pt
    
"""

from helm.clients.client import Client
from helm.common.request import Request, RequestResult, GeneratedOutput
import os
import sys
import torch
from typing import Optional
from datetime import datetime

sys.path.insert(0, "/share/pi/ema2016/users/sronaghi/proxy_tuning")

from any_model import load_any_model

LOCAL_RESULTS_DIR = "/share/pi/ema2016/users/sronaghi/proxy_tuning/results/medhelm"


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# proxy tuning helpers
def _safe_tag(model_name: str) -> str:
    # e.g. "proxy_tuning/llama70b_mellama13bchat" -> "proxy_tuning_llama70b_mellama13bchat"
    return model_name.replace("/", "_").replace(" ", "").replace(".", "").replace("-", "")


def setup_run_dirs(model_name: str, root=LOCAL_RESULTS_DIR):
    """
    Creates:
      <root>/<TAG>_<YYYYMMDD_HHMMSS>/
          ├─ <TAG>_<YYYYMMDD_HHMMSS>.csv
          └─ logits_analysis/
    Returns: (run_dir, csv_path, logits_dir)
    """
    ensure_dir(root)
    tag = _safe_tag(model_name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root, f"{tag}_{stamp}")
    ensure_dir(run_dir)

    csv_name = f"{tag}_{stamp}.csv"
    csv_path = os.path.join(run_dir, csv_name)
    with open(csv_path, "w") as f:
        f.write("timestamp,request_id,model_name,prompt,output,logits_path\n")

    logits_dir = os.path.join(run_dir, "logits_analysis")
    ensure_dir(logits_dir)

    print(f"[TokenLog] created run dir: {run_dir}")
    print(f"[TokenLog] csv: {csv_path}")
    print(f"[TokenLog] logits dir: {logits_dir}")
    return run_dir, csv_path, logits_dir


def append_request_row(
    csv_path: str, request_id: str, model_name: str, prompt: str, output: str, logits_path: str | None
):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def esc(s: str) -> str:
        if s is None:
            return ""
        return s.replace("\n", "\\n").replace(",", "&#44;")

    with open(csv_path, "a") as f:
        f.write(f"{ts},{request_id},{esc(model_name)},{esc(prompt)},{esc(output)},{esc(logits_path or '')}\n")


class ProxyTuningClient(Client):
    """
    A HELM client that uses ProxyTuning for inference instead of directly calling the model.
    """

    def __init__(
        self,
        model_name: str,
    ):
        """
        Initializes the ProxyTuningClient.

        """
        self.run_dir, self.token_log_path, self.logits_dir = setup_run_dirs(model_name)
        self.model_name = model_name
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.req_seq = 0
        tag = model_name.split("/")[-1]
        self.return_logits_for_analysis = True

        parts = tag.split("_")
        base_name, expert_name, antiexpert_name, self.alpha, self.score_type, k_str = (
            parts[0],
            parts[1],
            parts[2],
            float(parts[3]),
            parts[4],
            parts[5],
        )
        self.k = int(k_str)
        self.is_unite = False
        self.is_proxy = False
        if expert_name != "none":
            if antiexpert_name == "none":
                self.is_unite = True
            else:
                self.is_proxy = True

        self.any_model = load_any_model(
            base_name=base_name,
            expert_name=expert_name,
            antiexpert_name=antiexpert_name,
            alpha=self.alpha,
            proxy=self.is_proxy,
            unite=self.is_unite,
        )

    def make_request(self, request: Request) -> RequestResult:

        prompt_text = request.prompt
        max_new_tokens = 750

        if request.max_tokens:
            max_new_tokens = request.max_tokens
            print("max_new_tokens: ", max_new_tokens)

        if request.messages:
            print(request.messages)
            prompt_text = " ".join(msg["content"] for msg in request.messages if msg.get("role") != "system")

        # progress = tqdm.tqdm(total=1, desc="Generating Completions")
        print("doing a generation", flush=True)

        generation, logit_results = self.any_model.generate(
            prompt=prompt_text,
            max_new_tokens=max_new_tokens,
            alpha=self.alpha,
            return_logits_for_analysis=self.return_logits_for_analysis,
            score_type=self.score_type,
            k=self.k,
            unite=self.is_unite,
            proxy=self.is_proxy,
        )

        print("generation: ", generation, flush=True)

        self.req_seq += 1
        request_id = f"{self.run_id}_r{self.req_seq:04d}"

        logits_path = None
        if self.return_logits_for_analysis and logit_results:
            logits_path = os.path.join(self.logits_dir, f"logits_{request_id}.pt")
            torch.save(logit_results, logits_path)
            print(f"[Logits] wrote {logits_path}")

        append_request_row(
            csv_path=self.token_log_path,
            request_id=request_id,
            model_name=self.model_name,
            prompt=prompt_text,
            output=generation,
            logits_path=logits_path,
        )

        # Return a HELM-compatible RequestResult
        output = GeneratedOutput(text=generation, logprob=0.0, tokens=[])
        return RequestResult(success=True, cached=False, completions=[output], embedding=[])
