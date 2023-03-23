from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService
from helm.common.tokenization_request import TokenizerInfo


class RemoteWindowService(LocalWindowService):
    def __init__(self, service: TokenizerService, model_name: str):
        super().__init__(service)
        self.model_name = model_name
        info = self.service.get_info(model_name)
        self._max_sequence_length = info.max_sequence_length
        self._max_request_length = info.max_request_length
        self._end_of_text_token = info.end_of_text_token
        self._prefix_token = info.prefix_token

    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    @property
    def max_request_length(self) -> int:
        return self._max_request_length

    @property
    def end_of_text_token(self) -> str:
        return self._end_of_text_token

    @property
    def prefix_token(self) -> str:
        return self._prefix_token

    @property
    def tokenizer_name(self) -> str:
        """Name of the tokenizer to use when sending a request."""
        return self.model_name