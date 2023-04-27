from helm.proxy.tokenizers.tokenizer import TokenizerModel, TokenizerSpec, EncodeResult
from helm.common.tokenization_request import TokenizationToken

from typing import List, Optional

class HuggingFaceTokenizerModel(TokenizerModel):

    def set_spec(self, spec: TokenizerSpec):
        super().set_spec(spec)
        # TODO: Initialize the tokenizer here

    @property
    def provider_name(self) -> str:
        return "huggingface"
    
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError("HuggingFaceTokenizerModel.tokenize() is not implemented")
    
    def encode(self, text: str, truncation: bool = False, max_length: Optional[int] = None) -> EncodeResult:
        raise NotImplementedError("HuggingFaceTokenizerModel.encode() is not implemented")
    
    def decode(self, tokens: List[TokenizationToken], normalized_text: Optional[str] = None) -> str:
        raise NotImplementedError("HuggingFaceTokenizerModel.decode() is not implemented")