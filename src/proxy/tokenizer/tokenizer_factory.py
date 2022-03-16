from benchmark.tokenizer_service import TokenizerService
from .ai21_tokenizer import AI21Tokenizer
from .tokenizer import Tokenizer
from .openai_tokenizer import OpenAITokenizer


class TokenizerFactory:
    @staticmethod
    def get_tokenizer(model: str, service: TokenizerService) -> Tokenizer:
        """Returns a `Tokenizer` given the model."""
        organization: str = model.split("/")[0]

        tokenizer: Tokenizer
        if organization == "openai" or organization == "simple":
            tokenizer = OpenAITokenizer()
        elif organization == "ai21":
            tokenizer = AI21Tokenizer(model=model, service=service)
        else:
            raise Exception(f"Unsupported model: {model}")

        return tokenizer
