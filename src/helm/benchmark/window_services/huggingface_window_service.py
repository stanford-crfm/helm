from typing import Optional
from helm.proxy.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer
from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService


class HuggingFaceWindowService(LocalWindowService):
    def __init__(
        self,
        service: TokenizerService,
        tokenizer_name: str,
        pretrained_model_name_or_path: Optional[str] = None,
        revision: Optional[str] = None,
        max_sequence_length: Optional[int] = None,
        max_reqeust_length: Optional[int] = None,
        end_of_text_token: Optional[str] = None,
        prefix_token: Optional[str] = None,
    ):
        super().__init__(service)
        self._tokenizer_name = tokenizer_name
        tokenizer = HuggingFaceTokenizer.get_tokenizer(
            helm_tokenizer_name=tokenizer_name,
            pretrained_model_name_or_path=pretrained_model_name_or_path or tokenizer_name,
            revision=revision,
        )
        # Override max_sequence_length, max_request_length, end_of_text_token
        # and prefix_token if provided as an argument.
        # Otherwise, auto-infer them from the Hugging Face tokenizer.
        #
        # Note that many Hugging Face tokenizers have incorrect sequence lengths,
        # so it is recommended to set this manually.
        self._max_sequence_length = max_sequence_length or tokenizer.model_max_length
        self._max_request_length = max_reqeust_length or tokenizer.model_max_length
        self._end_of_text_token = end_of_text_token or tokenizer.eos_token
        self._prefix_token = prefix_token or tokenizer.bos_token

    @property
    def tokenizer_name(self) -> str:
        """Name of the tokenizer to use when sending a request."""
        return self._tokenizer_name

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length of this tokenizer."""
        return self._max_sequence_length

    @property
    def max_request_length(self) -> int:
        """Return the max request length of this tokenizer."""
        return self._max_request_length

    @property
    def end_of_text_token(self) -> str:
        """The end of text token."""
        return self._end_of_text_token

    @property
    def prefix_token(self) -> str:
        """The prefix token."""
        return self._prefix_token
