from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService
from helm.benchmark.tokenizer_config_registry import TokenizerConfig
from helm.benchmark.model_deployment_registry import ModelDeployment


class ConfigurableTokenizerWindowService(LocalWindowService):
    """A LocalWindowService that takes its parameters from ModelDeployment and TokenizerConfig"""

    def __init__(self, service: TokenizerService, tokenizer_config: TokenizerConfig, model_deployment: ModelDeployment):
        super().__init__(service)
        if model_deployment.max_sequence_length is not None:
            self._max_sequence_length = model_deployment.max_sequence_length
        else:
            raise Exception(f"`max_sequence_length` needs to be set for ModelDeployment {model_deployment.name}")

        self._max_request_length = model_deployment.max_request_length
        if model_deployment.max_request_length is not None:
            self._max_request_length = model_deployment.max_request_length
        else:
            self._max_request_length = self._max_sequence_length

        if tokenizer_config.end_of_text_token is not None:
            self._end_of_text_token = tokenizer_config.end_of_text_token
        else:
            self._end_of_text_token = ""

        if tokenizer_config.prefix_token is not None:
            self._prefix_token = tokenizer_config.prefix_token
        else:
            self._prefix_token = self._end_of_text_token

        self._tokenizer_name = tokenizer_config.name

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length of this tokenizer."""
        return self._max_request_length

    @property
    def max_request_length(self) -> int:
        """Return the max request length of GPT-2."""
        return self._max_request_length

    @property
    def end_of_text_token(self) -> str:
        """The end of text token."""
        return self._end_of_text_token

    @property
    def tokenizer_name(self) -> str:
        """Name of the tokenizer to use when sending a request."""
        return self._tokenizer_name

    @property
    def prefix_token(self) -> str:
        """The prefix token for models that uses the GPT-2 tokenizer is the end of text token."""
        return self._prefix_token
