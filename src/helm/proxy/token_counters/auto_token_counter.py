from typing import List
from helm.benchmark.model_deployment_registry import ModelDeployment, get_model_deployment

from helm.common.request import Request, GeneratedOutput
from helm.tokenizers.auto_tokenizer import AutoTokenizer
from helm.common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from helm.proxy.token_counters.token_counter import TokenCounter


class AutoTokenCounter(TokenCounter):
    """Automatically count tokens based on the model_deployment."""

    def __init__(self, auto_tokenizer: AutoTokenizer):
        self.auto_tokenizer: AutoTokenizer = auto_tokenizer

    def count_tokens(self, request: Request, completions: List[GeneratedOutput]) -> int:
        """Counts tokens based on the model deployment.

        This counts the number of tokens in the request and completions.
        Both input and output tokens are counted. For some model providers,
        this method will return a larger number of tokens than the actual
        token count used for billing. For example, GooseAI only charges for
        (output_tokens - 25) rather than (input_tokens + output_tokens)."""
        model_deployment: ModelDeployment = get_model_deployment(request.model_deployment)
        assert model_deployment.tokenizer_name
        tokenizer_name = model_deployment.tokenizer_name

        num_completion_tokens = 0
        for completion in completions:
            if completion.tokens:
                num_completion_tokens += len(completion.tokens)
            else:
                tokenized_completion: TokenizationRequestResult = self.auto_tokenizer.tokenize(
                    TokenizationRequest(request.prompt, tokenizer=tokenizer_name)
                )
                num_completion_tokens += len(tokenized_completion.tokens)

        tokenized_prompt: TokenizationRequestResult = self.auto_tokenizer.tokenize(
            TokenizationRequest(request.prompt, tokenizer=tokenizer_name)
        )
        num_prompt_tokens = len(tokenized_prompt.tokens)
        return num_prompt_tokens + num_completion_tokens
