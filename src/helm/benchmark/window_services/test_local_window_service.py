from helm.benchmark.window_services.default_window_service import DefaultWindowService
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
)


class _CountingTokenizerService:
    """
    Test tokenizer where each character counts as one token.
    Used to test window truncation logic.
    """

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        text = request.text[: request.max_length] if request.truncation else request.text
        tokens = [TokenizationToken(value=i) for i, _ in enumerate(text)]
        return TokenizationRequestResult(
            success=True,
            cached=False,
            text=text,
            tokens=tokens,
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        return DecodeRequestResult(
            success=True,
            cached=False,
            text="x" * len(request.tokens),
        )


def test_local_window_service_uses_combined_prompt_completion_budget():
    """
    MWE for a bug in prompt budgeting.

    Given:
      max_request_length = 2048
      max_sequence_and_generated_tokens_length = 2040
      expected_completion_token_length = 5

    The real prompt budget is:
      min(2048, 2040 - 5) = 2035

    So:
      2035-token prompt -> fits
      2036-token prompt -> does not fit
      truncation target -> 2035
    """
    ws = DefaultWindowService(
        service=_CountingTokenizerService(),
        tokenizer_name="huggingface/gpt2",
        max_sequence_length=2048,
        max_request_length=2048,
        max_sequence_and_generated_tokens_length=2040,
    )

    completion = 5
    got_fit_2035 = ws.fits_within_context_window("x" * 2035, completion)
    got_fit_2036 = ws.fits_within_context_window("x" * 2036, completion)
    got_trunc_len = len(ws.truncate_from_right("x" * 3000, completion))

    assert got_fit_2035, "fits(2035, completion=5) == {got_fit_2035}  expected=True"
    assert not got_fit_2036, "fits(2036, completion=5) == {got_fit_2036}  expected=False"
    assert got_trunc_len == 2035, "truncate_len == {got_trunc_len}  expected=2035"
