from helm.proxy.tokenizers.tokenizer import TokenizerModel, TokenizerSpec, EncodeResult
from helm.common.tokenization_request import TokenizationToken

from typing import List, Optional
import tiktoken

class TiktokenTokenizerModel(TokenizerModel):

    def __init__(self):
        super().__init__()
        self.tokenizer: Optional[tiktoken.Encoding] = None

    def set_spec(self, spec: TokenizerSpec):
        super().set_spec(spec)
        self.tokenizer = tiktoken.get_encoding(self._get_tokenizer_name(self.spec.name))

    @property
    def provider_name(self) -> str:
        return "tiktoken"
    
    @staticmethod
    def _get_tokenizer_name(tokenizer_full_name: str) -> str:
        return tokenizer_full_name.split("/")[1]
    
    def tokenize(self, text: str) -> List[str]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        tokens: List[str] = [self.tokenizer.decode([token]) for token in self.tokenizer.encode(text)]
        return tokens
    
    def encode(self, text: str, truncation: bool = False, max_length: Optional[int] = None) -> EncodeResult:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        tokens: List[int] = self.tokenizer.encode(text)
        if truncation:
            tokens = tokens[:max_length]
        return EncodeResult(text = text, tokens = [TokenizationToken(token) for token in tokens])
    
    def decode(self, tokens: List[TokenizationToken], normalized_text: Optional[str] = None) -> str:
        """Ignores normalized_text"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        tokens: List[int] = [token if isinstance(token, int) else self.tokenizer.encode(token)[0] for token in tokens]
        text: str = self.tokenizer.decode(tokens)
        return text
