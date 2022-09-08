from .gpt2_window_service import GPT2WindowService
from .tokenizer_service import TokenizerService


class MTNLGWindowService(GPT2WindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """
        The max length of the model input. MT-NLG does not predict the logprob of the first
        input token so `max_sequence_length` is one token shorter than `max_request_length`.
        """
        return self.max_request_length - 1

    @property
    def max_request_length(self) -> int:
        """
        The max request length for the MT-NLG models is 2048.
        Source: https://github.com/microsoft/turing-academic-TNLG
        """
        return 2048

    @property
    def prefix_token(self) -> str:
        return "<<"
