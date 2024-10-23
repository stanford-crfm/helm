import json
import logging
import time
from pathlib import Path
from threading import Lock
from typing import List, Literal, Optional, Dict, Union

import torch

from helm.common.cache import CacheConfig
from helm.common.optional_dependencies import OptionalDependencyNotInstalled
from helm.common.request import Request, RequestResult, GeneratedOutput, Token
from helm.tokenizers.tokenizer import Tokenizer

from helm.clients.client import CachingClient
from helm.clients.lit_gpt_generate import generate  # type: ignore

try:
    import lightning as L
    from lightning.fabric.strategies import FSDPStrategy
    from lit_gpt import GPT, Config
    from lit_gpt.model import Block
    from lit_gpt.utils import check_valid_checkpoint_dir, lazy_load, quantization
except ModuleNotFoundError as e:
    # Provide manual instructions for installing lit-gpt from GitHub
    # because PyPI does not allow installing dependencies directly from GitHub.
    raise OptionalDependencyNotInstalled(
        f"Optional dependency {e.name} is not installed. "
        "Please run `pip install lit-gpt@git+https://github.com/Lightning-AI/lit-gpt@main` to install it."
    ) from e

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

QuantizationType = Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]


class SingletonMeta(type):
    _instances: Dict[type, type] = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class LitGPT(metaclass=SingletonMeta):
    def __init__(
        self,
        checkpoint_dir: Path = Path(""),
        precision: str = "bf16-true",
        device: str = "auto",
        devices: int = 1,
        strategy: Union[str, FSDPStrategy] = "auto",
        quantize: Optional[QuantizationType] = None,
    ):
        torch.set_float32_matmul_precision("high")

        if strategy == "fsdp":
            strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False)
        fabric = L.Fabric(devices=devices, accelerator=device, precision=precision, strategy=strategy)  # type: ignore
        fabric.launch()
        logger.info("Using device: {}".format(fabric.device))

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
        self.fabric = fabric


class LitGPTClient(CachingClient):
    """Client for evaluating Lit-GPT (from Lightning AI) supported LLMs"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        checkpoint_dir: Path = Path(""),
        precision: str = "bf16-true",
        device: str = "auto",
        devices: int = 1,
        strategy: str = "auto",
        quantize: Optional[QuantizationType] = None,
    ):
        super().__init__(cache_config=cache_config)
        self.tokenizer = tokenizer
        lit_gpt = LitGPT(checkpoint_dir, precision, device, devices, strategy, quantize)
        self.model = lit_gpt.model
        self.fabric = lit_gpt.fabric

    def make_request(self, request: Request) -> RequestResult:
        model = self.model

        # TODO: Fix this, it's a hack
        # Check is self.tokenizer is of type LitGPTTokenizer
        assert hasattr(self.tokenizer, "_tokenizer"), "Tokenizer must be of type LitGPTTokenizer"
        tokenizer = self.tokenizer._tokenizer  # type: ignore

        fabric = self.fabric
        encoded = tokenizer.encode(request.prompt, bos=True, eos=False, device=fabric.device)
        prompt_length = encoded.size(0)
        max_returned_tokens: int = prompt_length + request.max_tokens
        assert max_returned_tokens <= model.config.block_size, (
            max_returned_tokens,
            model.config.block_size,
        )  # maximum rope cache length

        model.clear_kv_cache()

        with fabric.init_tensor():
            # set the max_seq_length to limit the memory usage to what we need
            model.max_seq_length = max_returned_tokens

        t0 = time.perf_counter()
        # helm doesn't have anything equivalent to top_k at the moment
        # TODO: allow temperature=0, pick the top token rather than sampling.
        stop_tokens: List[torch.Tensor] = [tokenizer.encode(e, device=fabric.device) for e in request.stop_sequences]

        with fabric.init_tensor():
            # enable the kv cache
            model.set_kv_cache(batch_size=1)
        tokens = generate(
            model,
            encoded,
            max_returned_tokens,
            temperature=max(request.temperature, 1e-11),
            stop_tokens=stop_tokens,
        )

        t = time.perf_counter() - t0
        model.clear_kv_cache()
        if request.echo_prompt is False:
            output = tokenizer.decode(tokens[prompt_length:])
        else:
            output = tokenizer.decode(tokens)
        tokens_generated = tokens.size(0) - prompt_length

        logger.debug(f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec")
        logger.debug(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

        generated_tokens = []
        for token in tokens:
            generated_tokens.append(Token(text=tokenizer.decode(token), logprob=0))
        completions = [GeneratedOutput(text=output, logprob=0, tokens=generated_tokens)]

        return RequestResult(
            success=True,
            cached=False,
            error=None,
            completions=completions,
            embedding=[],
            request_time=t,
        )
