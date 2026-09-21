import hashlib
import tempfile
from pathlib import Path
from typing import Optional

from helm.common.optional_dependencies import handle_module_not_found_error

try:
    import dspy
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["dspy"])

from helm.benchmark.runner import _get_current_run_spec_name
from helm.clients.client import Client
from helm.common.cache import Cache, CacheConfig
from helm.common.general import ensure_file_downloaded
from helm.common.hierarchical_logger import hlog
from helm.common.request import Request, RequestResult, GeneratedOutput
from helm.proxy.retry import NonRetriableException
from helm.tokenizers.tokenizer import Tokenizer


class DSPyClient(Client):
    """
    A HELM client that uses DSPy for inference instead of directly calling the model.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        model_name: Optional[str] = None,
        dspy_agent_url: Optional[str] = None,
        dspy_module: Optional[str] = None,
        dspy_api_model: Optional[str] = None,
        dspy_api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initializes the DSPyClient.

        Args:
            cache_config (CacheConfig): Configuration for caching.
            tokenizer (Tokenizer): Tokenizer instance (unused but required by HELM interface).
            tokenizer_name (str): Name of the tokenizer (unused but required by HELM interface).
            model_name (str): The official model name used within HELM.
            dspy_agent_url (str): URL for the DSPy agent JSON configuration.
            dspy_module (str): The module to use with DSPy (Predict, ChainOfThought).
            dspy_api_model (str): The actual model name (API) to use with DSPy.
            dspy_api_base (str): Base URL for the model API.
            api_key (str): API key for the DSPy model provider.
        """

        if not model_name:
            raise NonRetriableException("Please specify the model name in model_deployments.yaml")
        if not dspy_api_model:
            raise NonRetriableException("Please specify dspy_api_model in model_deployments.yaml")

        if ("o3-mini" in model_name) or ("deepseek-r1" in model_name):
            self.lm = dspy.LM(
                model=dspy_api_model, api_base=dspy_api_base, api_key=api_key, temperature=1.0, max_tokens=100000
            )
        else:
            self.lm = dspy.LM(model=dspy_api_model, api_base=dspy_api_base, api_key=api_key, temperature=0.0)

        run_spec_name = _get_current_run_spec_name()
        self.scenario_name = run_spec_name.split(":")[0] if run_spec_name else "unknown"
        self.model_name = model_name
        self.dspy_agent_url_template = dspy_agent_url
        self.dspy_module = dspy_module
        self.dspy_api_model = dspy_api_model
        self.dspy_api_base = dspy_api_base
        self.api_key = api_key
        self.cache_dir = Path(tempfile.gettempdir()) / "dspy_agent_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self._current_cache_file = None
        self._load_agent_for_scenario()
        self.cache = Cache(cache_config) if cache_config else None

    def _get_cache_file(self, url):
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"agent_{url_hash}.json"

    def _load_agent_for_scenario(self):
        if self.dspy_module == "ChainOfThought":
            self.agent = dspy.ChainOfThought("inputs -> output")
        elif self.dspy_module == "Predict":
            self.agent = dspy.Predict("inputs -> output")
        else:
            raise ValueError(f"Unknown dspy_module: {self.dspy_module}")

        dspy_agent_url = (
            self.dspy_agent_url_template.format(scenario=self.scenario_name) if self.dspy_agent_url_template else None
        )
        if dspy_agent_url:
            cache_file = self._get_cache_file(dspy_agent_url)
            if self._current_cache_file == cache_file:
                return
            try:
                ensure_file_downloaded(source_url=dspy_agent_url, target_path=str(cache_file))
                self.agent.load(str(cache_file))
                self._current_cache_file = cache_file

            except Exception as e:
                raise NonRetriableException(f"Failed to load DSPy agent from URL {dspy_agent_url}: {str(e)}")
        hlog(
            f"DSPy client initialized - HELM Model: {self.model_name}, DSPy Model: {self.dspy_api_model}, API Base: {self.dspy_api_base}, API Key: {'***' if self.api_key else None}, DSPy Agent: {dspy_agent_url}, DSPy Module: {self.dspy_module}"
        )

    def make_request(self, request: Request) -> RequestResult:
        run_spec_name = _get_current_run_spec_name()
        current_scenario = run_spec_name.split(":")[0] if run_spec_name else "unknown"
        if current_scenario != self.scenario_name:
            self.scenario_name = current_scenario
            if self.dspy_agent_url_template:
                self._load_agent_for_scenario()
            else:
                hlog(
                    f"DSPy client initialized - HELM Model: {self.model_name}, DSPy Model: {self.dspy_api_model}, API Base: {self.dspy_api_base}, API Key: {'***' if self.api_key else None}, DSPy Agent: None, DSPy Module: {self.dspy_module}"
                )

        prompt_text = request.prompt
        if request.messages:
            prompt_text = " ".join(msg["content"] for msg in request.messages if msg.get("role") != "system")

        try:
            with dspy.context(lm=self.lm):
                prediction = self.agent(inputs=prompt_text)
            output_text = prediction.output if hasattr(prediction, "output") else str(prediction)
        except Exception as e:
            return RequestResult(success=False, cached=False, completions=[], embedding=[], error=str(e))

        output = GeneratedOutput(text=output_text, logprob=0.0, tokens=[])
        return RequestResult(success=True, cached=False, completions=[output], embedding=[])
