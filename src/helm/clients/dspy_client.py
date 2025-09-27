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
            api_provider = model_name.split('/')[0]
            raise NonRetriableException(f"Please provide {api_provider}ApiKey key through credentials.conf")

        if ("o3-mini" in model_name) or ("deepseek-r1" in model_name):
            self.lm = dspy.LM(model=api_model, api_base=api_base, api_key=api_key, temperature=1.0, max_tokens=100000)
        else:
            self.lm = dspy.LM(model=api_model, api_base=api_base, api_key=api_key, temperature=0.0)
        
        self.scenario_name = os.environ.get('HELM_CURRENT_SCENARIO', 'unknown')
        self.dspy_agent_url_template = dspy_agent_url
        self.model_name = model_name
        self.api_model = api_model
        self.api_base = api_base
        self.api_key = api_key
        self.dspy_module = dspy_module
        
        self._load_agent_for_scenario()
        self.cache = Cache(cache_config) if cache_config else None

    def _load_agent_for_scenario(self):    
        """Load DSPy agent configuration for the current scenario."""
        if self.dspy_module == "ChainOfThought":
            self.agent = dspy.ChainOfThought("inputs -> output")
        else:
            self.agent = dspy.Predict("inputs -> output")
        dspy_agent_url = self.dspy_agent_url_template.format(scenario=self.scenario_name) if self.dspy_agent_url_template else None
        if dspy_agent_url:
            try:
                response = requests.get(dspy_agent_url)
                response.raise_for_status()
                agent_config = response.json()

                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                    json.dump(agent_config, temp_file)
                    temp_file_path = temp_file.name
                
                self.agent.load(temp_file_path)
                os.unlink(temp_file_path)

            except Exception as e:
                raise NonRetriableException(f"Failed to load DSPy agent from URL {dspy_agent_url}: {str(e)}")     
        hlog(f"DSPy client initialized - HELM Model: {self.model_name}, DSPy Model: {self.api_model}, API Base: {self.api_base}, API Key: {'***' if self.api_key else None}, DSPy Agent: {dspy_agent_url}, DSPy Module: {self.dspy_module}")

    def make_request(self, request: Request) -> RequestResult:
        """
        Handles a request by sending the prompt to DSPy.

        Args:
            request (Request): The request object containing the prompt.

        Returns:
            RequestResult: A HELM-compatible response object.
        """
        current_scenario = os.environ.get('HELM_CURRENT_SCENARIO', 'unknown')
        self.scenario_name = current_scenario if current_scenario != self.scenario_name else self.scenario_name
        if self.dspy_agent_url_template:
            self._load_agent_for_scenario()
        prompt_text = request.prompt

        if request.messages:
            prompt_text = " ".join(msg["content"] for msg in request.messages if msg.get("role") != "system")

        current_agent_url = self.dspy_agent_url_template.format(scenario=self.scenario_name) if self.dspy_agent_url_template else None
        
        try:
            with dspy.context(lm=self.lm):
                prediction = self.agent(inputs=prompt_text)
            output_text = prediction.output if hasattr(prediction, "output") else str(prediction)
        except Exception as e:
            return RequestResult(success=False, cached=False, completions=[], embedding=[], error=str(e))

        # Return a HELM-compatible RequestResult
        output = GeneratedOutput(text=output_text, logprob=0.0, tokens=[])
        return RequestResult(success=True, cached=False, completions=[output], embedding=[])