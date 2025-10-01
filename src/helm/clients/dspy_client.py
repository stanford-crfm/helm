# File: helm/clients/dspy_client.py
from helm.clients.client import Client
from helm.common.cache import CacheConfig
from helm.tokenizers.tokenizer import Tokenizer
from helm.common.cache import Cache
from helm.common.request import Request, RequestResult, GeneratedOutput
from helm.proxy.retry import NonRetriableException
from helm.common.hierarchical_logger import hlog
import dspy
import requests
import threading
import os
import json
import tempfile
import fcntl
import hashlib
from pathlib import Path


class DSPyClient(Client):
    """
    A HELM client that uses DSPy for inference instead of directly calling the model.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        model_name: str = None,
        dspy_agent_url: str = None,
        dspy_module: str = None,
        api_model: str = None,
        api_base: str = None,
        api_key: str = None,
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
            api_model (str): The actual model name (API) to use with DSPy.
            api_base (str): Base URL for the model API.
            api_key (str): API key for the DSPy model provider.
        """

        if not model_name:
            raise NonRetriableException("Please specify the model name in model_deployments.yaml")
        if not api_model:
            raise NonRetriableException("Please specify the model name according to the API in model_deployments.yaml")
        if not api_key:
            api_provider = model_name.split("/")[0]
            raise NonRetriableException(f"Please provide {api_provider}ApiKey key through credentials.conf")

        if ("o3-mini" in model_name) or ("deepseek-r1" in model_name):
            self.lm = dspy.LM(model=api_model, api_base=api_base, api_key=api_key, temperature=1.0, max_tokens=100000)
        else:
            self.lm = dspy.LM(model=api_model, api_base=api_base, api_key=api_key, temperature=0.0)

        self.scenario_name = os.environ.get("HELM_CURRENT_SCENARIO", "unknown")
        self.model_name = model_name
        self.dspy_agent_url_template = dspy_agent_url
        self.dspy_module = dspy_module
        self.api_model = api_model
        self.api_base = api_base
        self.api_key = api_key
        self.cache_dir = Path(tempfile.gettempdir()) / "dspy_agent_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self._current_cache_file = None
        self._load_agent_for_scenario()
        self.cache = Cache(cache_config) if cache_config else None

    def _get_cache_file(self, url):
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"agent_{url_hash}.json"

    def _download_with_lock(self, url, cache_file):
        lock_file = cache_file.with_suffix(".lock")
        try:
            with open(lock_file, "w") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if cache_file.exists():
                    try:
                        with open(cache_file, "r") as f:
                            return json.load(f)
                    except:
                        cache_file.unlink(missing_ok=True)

                response = requests.get(url, timeout=30)
                response.raise_for_status()
                agent_config = response.json()

                temp_file = cache_file.with_suffix(".tmp")
                with open(temp_file, "w") as f:
                    json.dump(agent_config, f)
                temp_file.rename(cache_file)
                return agent_config
        except:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        finally:
            lock_file.unlink(missing_ok=True)

    def _load_agent_for_scenario(self):
        if self.dspy_module == "ChainOfThought":
            self.agent = dspy.ChainOfThought("inputs -> output")
        else:
            self.agent = dspy.Predict("inputs -> output")

        dspy_agent_url = (
            self.dspy_agent_url_template.format(scenario=self.scenario_name) if self.dspy_agent_url_template else None
        )
        if dspy_agent_url:
            cache_file = self._get_cache_file(dspy_agent_url)
            if self._current_cache_file == cache_file:
                return
            try:
                if cache_file.exists():
                    try:
                        with open(cache_file, "r") as f:
                            agent_config = json.load(f)
                    except:
                        agent_config = self._download_with_lock(dspy_agent_url, cache_file)
                else:
                    agent_config = self._download_with_lock(dspy_agent_url, cache_file)

                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
                    json.dump(agent_config, temp_file)
                    temp_file_path = temp_file.name

                self.agent.load(temp_file_path)
                os.unlink(temp_file_path)
                self._current_cache_file = cache_file

            except Exception as e:
                raise NonRetriableException(f"Failed to load DSPy agent from URL {dspy_agent_url}: {str(e)}")
        hlog(
            f"DSPy client initialized - HELM Model: {self.model_name}, DSPy Model: {self.api_model}, API Base: {self.api_base}, API Key: {'***' if self.api_key else None}, DSPy Agent: {dspy_agent_url}, DSPy Module: {self.dspy_module}"
        )

    def make_request(self, request: Request) -> RequestResult:
        current_scenario = os.environ.get("HELM_CURRENT_SCENARIO", "unknown")
        if current_scenario != self.scenario_name:
            self.scenario_name = current_scenario
            if self.dspy_agent_url_template:
                self._load_agent_for_scenario()
            else:
                hlog(
                    f"DSPy client initialized - HELM Model: {self.model_name}, DSPy Model: {self.api_model}, API Base: {self.api_base}, API Key: {'***' if self.api_key else None}, DSPy Agent: None, DSPy Module: {self.dspy_module}"
                )

        prompt_text = request.prompt
        if request.messages:
            prompt_text = " ".join(msg["content"] for msg in request.messages if msg.get("role") != "system")

        current_agent_url = (
            self.dspy_agent_url_template.format(scenario=self.scenario_name) if self.dspy_agent_url_template else None
        )

        try:
            with dspy.context(lm=self.lm):
                prediction = self.agent(inputs=prompt_text)
            output_text = prediction.output if hasattr(prediction, "output") else str(prediction)
        except Exception as e:
            return RequestResult(success=False, cached=False, completions=[], embedding=[], error=str(e))

        output = GeneratedOutput(text=output_text, logprob=0.0, tokens=[])
        return RequestResult(success=True, cached=False, completions=[output], embedding=[])
