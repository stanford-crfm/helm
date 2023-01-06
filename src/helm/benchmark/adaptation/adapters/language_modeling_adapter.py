from typing import List, Tuple, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.scenarios.scenario import Instance, EVAL_SPLITS
from helm.benchmark.window_services.window_service import EncodeResult
from helm.common.general import flatten_list, parallel_map
from helm.common.hierarchical_logger import hlog, htrack
from helm.common.request import Request
from helm.common.tokenization_request import TokenizationToken
from .adapter import Adapter


class LanguageModelingAdapter(Adapter):
    """
    Each `Instance` in a `Scenario` looks like this:

        <input> -> <reference1>
                   <reference2>
                   <reference3> [correct]
                   <reference4>

    For language modeling, we don't use the references (even if they exist), just feed the input:

        <input>
    """

    @htrack(None)
    def adapt(self, instances: List[Instance], parallelism: int) -> ScenarioState:
        """
        Takes a a list of `Instance`s and builds a list of corresponding `RequestState`s.
        Only requires eval instances.
        """
        # Pick out evaluation instances. This includes both valid and test splits.
        eval_instances: List[Instance] = [instance for instance in instances if instance.split in EVAL_SPLITS]
        hlog(f"{len(eval_instances)} eval instances")

        all_request_states: List[RequestState] = flatten_list(
            parallel_map(self.generate_requests, instances, parallelism)
        )
        hlog(f"{len(all_request_states)} requests")

        return ScenarioState(self.adapter_spec, all_request_states)

    def generate_requests(self, eval_instance: Instance) -> List[RequestState]:
        """
        Adapted from https://github.com/EleutherAI/lm_perplexity/blob/main/lm_perplexity/utils.py.
        """
        max_sequence_length: int = self.window_service.max_sequence_length
        max_request_length: int = self.window_service.max_request_length
        prefix_token: str = self.window_service.prefix_token

        encode_result: EncodeResult = self.window_service.encode(eval_instance.input.text)
        tokens: List[TokenizationToken] = encode_result.tokens
        text: str = encode_result.text

        request_states: List[RequestState] = []
        num_predicted_tokens: int = 0

        # Special handling for first window: predict all tokens
        # Example for GPT-3:
        # Raw token sequence format: [<str_tok1>, <str_tok2>, ..., <byte_tok1>, ...]
        # (total length <= max_sequence_length)
        # Convert it to: [<eot>, <str_tok1>, <str_tok2>, ...]
        # (total length <= max_req_len = max_sequence_length+1 for GPT-3)
        # Num_conditioning_tokens = 1
        # Example: ["Hello", " world", "bytes:\xe2\x80"] => "<eot>Hello world"
        #
        # Note: There are trailing byte tokens in the raw sequence because some subwords/symbols might translate to
        # multiple tokens (e.g. ’ => ["bytes:\xe2\x80", "bytes:\x99"]) and we chunk documents by token, not by word.

        # Uses `max_sequence_length` instead of `max_request_length` here because `prefix_token` will be prepended
        # to the sequence later. This is the only place where `max_sequence_length` is used.
        first_seq_len: int = min(max_sequence_length, len(tokens))
        prompt_text, num_conditioning_tokens = self.construct_language_modeling_prompt(
            self.window_service.encode(prefix_token).tokens, tokens[:first_seq_len], max_request_length, text
        )
        request = Request(
            model=self.adapter_spec.model,
            prompt=prompt_text,
            num_completions=1,
            temperature=0,
            max_tokens=self.adapter_spec.max_tokens,  # usually this is zero
            stop_sequences=self.adapter_spec.stop_sequences,
            echo_prompt=True,
            random=self.adapter_spec.random,
        )
        request_state = RequestState(
            instance=eval_instance,
            reference_index=None,
            request_mode=None,
            train_trial_index=0,
            output_mapping=None,
            request=request,
            result=None,
            num_conditioning_tokens=1 if len(prefix_token) > 0 else 0,
            num_train_instances=self.adapter_spec.max_train_instances,
            prompt_truncated=False,
        )
        request_states.append(request_state)
        num_predicted_tokens += first_seq_len

        while num_predicted_tokens < len(tokens):
            # Example for GPT-3:
            # Raw token sequence format:
            # [<cond_byte1>, ..., <cond_str_tok1>, <cond_str_tok2>, ..., <pred_str_tok1>, ..., <pred_byte1>, ...]
            # (total length <= max_req_len = max_sequence_length+1 for GPT-3)
            #
            # Convert it to: [<cond_str_tok1>, <cond_str_tok2>, ..., <pred_str_tok1>, <pred_str_tok2>. ...]
            # (total length <= max_req_len = max_sequence_length+1 for GPT-3)
            #
            # Example: conditioning_tokens=["bytes:\x99", "Exc"], pred_tokens=["use", " me", "bytes:\xe2\x80"] =>
            # prompt="Excuse me", num_conditioning_tokens = 1

            # The upper bound is `max_req_len - 1` because there will be at least 1 conditioning tokens.
            window_pred_len: int = min(len(tokens) - num_predicted_tokens, max_request_length - 1)
            window_end: int = num_predicted_tokens + window_pred_len
            conditioning_tokens: List[TokenizationToken] = tokens[
                window_end - max_request_length : num_predicted_tokens
            ]
            pred_tokens: List[TokenizationToken] = tokens[num_predicted_tokens:window_end]
            prompt_text, num_conditioning_tokens = self.construct_language_modeling_prompt(
                conditioning_tokens, pred_tokens, max_request_length, text
            )

            request = Request(
                model=self.adapter_spec.model,
                prompt=prompt_text,
                num_completions=1,
                temperature=0,
                max_tokens=self.adapter_spec.max_tokens,  # usually this is zero
                stop_sequences=self.adapter_spec.stop_sequences,
                echo_prompt=True,
            )
            request_state = RequestState(
                instance=eval_instance,
                reference_index=None,
                request_mode=None,
                train_trial_index=0,
                output_mapping=None,
                request=request,
                result=None,
                num_conditioning_tokens=num_conditioning_tokens,
                num_train_instances=self.adapter_spec.max_train_instances,
                prompt_truncated=False,
            )
            request_states.append(request_state)
            num_predicted_tokens += window_pred_len

        return request_states

    def construct_language_modeling_prompt(
        self,
        conditioning_tokens: List[TokenizationToken],
        pred_tokens: List[TokenizationToken],
        max_req_len: int,
        text: str,
    ) -> Tuple[str, int]:
        """
        Some subwords/symbols might translate to multiple tokens. e.g. ’ => ["bytes:\xe2\x80", "bytes:\x99"].

        When a subword of this type happens to be the last token of a chunk, we need to strip the leading and
        trailing bytes to ensure the prompt is a valid string.

        Since some tokens are removed, we also need to recompute num_conditioning_tokens.

        For models using the GPT-2 tokenizer, conditioning_tokens and pred_tokens are integers; for AI21
        models, the tokens are TokenizationTokens.

        text is the normalized text fed to decode(). Some tokenizers (e.g. AI21) need this field for decoding.
        """
        raw_prompt: str
        raw_prompt, pred_tokens = self.fits_tokens_within_context_window(
            conditioning_tokens, pred_tokens, max_req_len, text
        )

        prompt: str = raw_prompt.strip("\ufffd")
        # If there are no byte tokens, avoid API calls
        if len(prompt) == len(raw_prompt):
            num_conditioning_tokens = len(conditioning_tokens)
        else:
            num_leading_byte_tokens: int = max_req_len - len(
                self.window_service.encode(raw_prompt.lstrip("\ufffd")).tokens
            )
            num_trailing_byte_tokens: int = max_req_len - len(
                self.window_service.encode(raw_prompt.rstrip("\ufffd")).tokens
            )

            # There are no string tokens to predict
            if num_trailing_byte_tokens >= len(pred_tokens):
                num_conditioning_tokens = len(self.window_service.encode(prompt).tokens)
            # There are no conditioning string tokens
            elif num_leading_byte_tokens >= len(conditioning_tokens):
                num_conditioning_tokens = 1
            else:
                num_conditioning_tokens = len(conditioning_tokens) - num_leading_byte_tokens
        return prompt, num_conditioning_tokens

    def fits_tokens_within_context_window(
        self,
        conditioning_tokens: List[TokenizationToken],
        pred_tokens: List[TokenizationToken],
        max_req_len: int,
        text: Optional[str] = None,
    ) -> Tuple[str, List[TokenizationToken]]:
        """
        This method is used for adapting instances for language modeling scenarios.
        For some tokenizers (e.g. AI21), decoding then encoding k tokens may result
        in > k tokens. This method trims the tokens and check with the tokenizer
        repeatedly until they fit in the context window.

        For models using the GPT-2 tokenizer, conditioning_tokens and pred_tokens
        are integers; for AI21 models, the tokens are TokenizationTokens.
        """
        prompt: str = self.window_service.decode(conditioning_tokens + pred_tokens, text)
        prompt_length: int = len(self.window_service.encode(prompt).tokens)

        # If the prompt is too long, removes the overflowing tokens.
        # Since encoding might generate extra tokens, we need to repeat this until prompt_length <= max_req_len.
        # For AI21, for example, this happens especially frequently when a document contains different types of
        # whitespace characters because some whitespaces are tokenized to multiple tokens and the others
        # are tokenized to a single token. However, the AI21 tokenizer seems to normalize all types
        # of whitespaces to the same whitespace character.
        #
        # e.g. original text: ",  (", which is tokenized to:
        # [('▁', 0, 0), (',', 0, 1), ('▁▁', 1, 3), ('(', 3, 4)]
        # normalized text: ",  (", which is tokenized to:
        # [('▁', 0, 0), (',', 0, 1), ('▁', 1, 2), ('▁', 2, 3), ('(', 3, 4)]
        while prompt_length > max_req_len:
            # Trims the extra (prompt_length - max_req_len) tokens
            pred_tokens = pred_tokens[: -(prompt_length - max_req_len)]
            prompt = self.window_service.decode(conditioning_tokens + pred_tokens, text)
            prompt_length = len(self.window_service.encode(prompt).tokens)

            # When the input text contains languages the tokenizer cannot process, the input text
            # might be inflated so the truncation cannot work properly.
            # e.g.
            # With the OpenAI tokenizer:
            # >>> tokenizer.decode(tokenizer.encode("行星运转"))
            # '行星运转'
            # With the YaLM tokenizer:
            # >>> tokenizer.decode(tokenizer.tokenize("行星运转"))
            # '行<0xE6><0x98><0x9F><0xE8><0xBF><0x90><0xE8><0xBD><0xAC>'
            if len(pred_tokens) == 0:
                raise ValueError(
                    "Truncating pred_tokens to fit them in the context window, "
                    "got len(pred_tokens) == 0, which will lead to an infinite loop."
                )

        return prompt, pred_tokens
