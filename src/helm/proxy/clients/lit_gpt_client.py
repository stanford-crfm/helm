import json
import logging
import time
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.model import Block
from lit_gpt.utils import check_valid_checkpoint_dir, lazy_load, quantization

from helm.common.cache import Cache, CacheConfig
from helm.common.request import Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult, TokenizationToken,
)
from .client import Client
from .lit_gpt_generate import generate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LitGPTClient(Client):
    """Implements some "models" that just generate silly things quickly just to debug the infrastructure."""

    def __init__(
            self,
            cache_config: CacheConfig,
            checkpoint_dir: str = "",
            precision: str = "bf16-true",
            device="auto",
            devices: int = 1,
            strategy: str = "auto",
            quantize: Optional[
                Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    ):
        torch.set_float32_matmul_precision("high")

        self.cache = Cache(cache_config)
        if strategy == "fsdp":
            strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False)
        fabric = L.Fabric(devices=devices, accelerator=device, precision=precision, strategy=strategy)
        fabric.launch()
        logger.info("Using device: {}".format(fabric.device))

        checkpoint_dir = Path(checkpoint_dir)
        check_valid_checkpoint_dir(checkpoint_dir)

        with open(checkpoint_dir / "lit_config.json") as fp:
            config = Config(**json.load(fp))

        checkpoint_path = checkpoint_dir / "lit_model.pth"
        logger.info(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
        with fabric.init_module(empty_init=True), quantization(quantize):
            model = GPT(config)

        with lazy_load(checkpoint_path) as checkpoint:
            model.load_state_dict(checkpoint, strict=quantize is None)

        model.eval()
        self.model = fabric.setup(model)
        self.tokenizer = Tokenizer(checkpoint_dir)
        self.fabric = fabric

    def make_request(self, request: Request) -> RequestResult:
        model = self.model
        tokenizer = self.tokenizer
        fabric = self.fabric
        encoded = tokenizer.encode(request.prompt, bos=True, eos=False, device=fabric.device)
        prompt_length = encoded.size(0)
        max_returned_tokens = prompt_length + request.max_tokens
        assert max_returned_tokens <= model.config.block_size, (
            max_returned_tokens,
            model.config.block_size,
        )  # maximum rope cache length

        model.reset_cache()
        t0 = time.perf_counter()
        # helm doesn't have anything equivalent to top_k at the moment
        # TODO: allow temperature=0, pick the top token rather than sampling.
        eos_ids = [tokenizer.encode(e) for e in request.stop_sequences]
        tokens, logprobs, top_logprobs = generate(
            model,
            encoded,
            max_returned_tokens,
            max_seq_length=max_returned_tokens,
            temperature=max(request.temperature, 1e-11),
            eos_ids=eos_ids,
        )

        t = time.perf_counter() - t0
        model.reset_cache()
        if request.echo_prompt is False:
            output = tokenizer.decode(tokens[prompt_length:])
        else:
            output = tokenizer.decode(tokens)
        tokens_generated = tokens.size(0) - prompt_length

        logger.debug(f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec")
        logger.debug(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

        generated_tokens = []
        for token in tokens:
            generated_tokens.append(Token(text=tokenizer.decode(token), logprob=0, top_logprobs={}))
        tokens = generated_tokens
        completions = [Sequence(text=output, logprob=0, tokens=tokens)]

        return RequestResult(
            success=True,
            cached=False,
            error=None,
            completions=completions,
            embedding=[],
            request_time=None,
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        fabric = self.fabric
        logger.debug("Using device: {}".format(fabric.device))
        t0 = time.perf_counter()
        encoded = self.tokenizer.encode(request.text, bos=True, eos=False, device=fabric.device)
        tokens = encoded.tolist()
        tokens = [TokenizationToken(value=token) for token in tokens]
        t = time.perf_counter() - t0
        return TokenizationRequestResult(success=True, cached=False, tokens=tokens, text=request.text, request_time=t)

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        t0 = time.perf_counter()
        text = self.tokenizer.decode(torch.as_tensor(request.tokens, dtype=torch.int))
        t = time.perf_counter() - t0
        return DecodeRequestResult(success=True, cached=False, text=text, request_time=t)
