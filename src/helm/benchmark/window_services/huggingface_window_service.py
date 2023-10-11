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
    ):
        super().__init__(service)
        self._tokenizer_name = tokenizer_name
        tokenizer = HuggingFaceTokenizer.get_tokenizer(
            helm_tokenizer_name=tokenizer_name,
            pretrained_model_name_or_path=pretrained_model_name_or_path or tokenizer_name,
            revision=revision,
        )
        self._prefix_token = tokenizer.bos_token
        self._end_of_text_token = tokenizer.eos_token
        # Override max_sequence_length if provided as an argument.
        # Otherwise, auto-infer max_sequence_length from the Hugging Face tokenizer.
        # Note that many Hugging Face tokenizers have incorrect sequence lengths,
        # so it is recommended to set this manually.
        if max_sequence_length:
            self._max_sequence_length = max_sequence_length
        else:
            self._max_sequence_length = tokenizer.model_max_length
        self._max_request_length = max_reqeust_length

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length of this tokenizer."""
        return self._max_sequence_length

    @property
    def max_request_length(self) -> int:
        """Return the max request length of this tokenizer."""
        return self._max_request_length or self._max_sequence_length

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
        """The prefix token."""
        return self._prefix_token
