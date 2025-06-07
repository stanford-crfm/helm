import spacy
import spacy.cli
import numpy as np
from dataclasses import replace
from typing import Dict, List, Tuple, Any, Optional

from transformers import AutoTokenizer, GPT2Tokenizer
import spacy.language

from helm.common.hierarchical_logger import hlog
from helm.benchmark.model_deployment_registry import get_model_deployment, ModelDeployment
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.benchmark.metrics.metric import Stat
from helm.benchmark.metrics.metric_name import MetricName
from helm.common.tokenization_request import TokenizationRequest

from .prompt_translations import TS_GUESSING_BASE, TS_GUESSING_MULTICHOICE

PROMPT_CONFIGS_MASTER = {
    "ts_guessing_question_base": TS_GUESSING_BASE,
    "ts_guessing_question_multichoice": TS_GUESSING_MULTICHOICE,
}


class ContaminationUtils:
    """
    Utilities for contamination detection strategies. Provides common functionalities for
    handling HELM objects, model configurations, prompt management, and result formatting.
    """

    DEFAULT_MODEL_MAX_CONTEXT_TOKENS: int = 4096
    AVG_CHARS_PER_TOKEN_HEURISTIC: int = 4
    CONSERVATIVE_CHARS_PER_TOKEN_DIVISOR: int = max(1, AVG_CHARS_PER_TOKEN_HEURISTIC - 1)
    MAX_REASONABLE_CONTEXT_LENGTH: int = 2_000_000
    SPACY_MODEL_MAP: Dict[str, str] = {
        "en": "en_core_web_sm",
        "pt": "pt_core_news_sm",
        "zh": "zh_core_web_sm",
    }

    _gpt2_tokenizer_cache: Optional[GPT2Tokenizer] = None
    _spacy_models_cache: Dict[str, Optional[spacy.language.Language]] = {}

    @staticmethod
    def determine_model_max_length(
        model_deployment_name: str, default_max_len: int = DEFAULT_MODEL_MAX_CONTEXT_TOKENS
    ) -> int:
        """
        Determines the maximum sequence length for a model with a clear order of priority.
        Priority: ModelDeployment -> Hugging Face Tokenizer Config -> Default Value.
        """
        try:
            deployment = get_model_deployment(model_deployment_name)
            max_len = ContaminationUtils._get_max_length_from_deployment(deployment)
            if max_len:
                hlog(f"INFO: Max length {max_len} found in ModelDeployment for '{model_deployment_name}'")
                return max_len
        except (KeyError, ImportError):
            pass

        tokenizer_name = ContaminationUtils.get_tokenizer_name(model_deployment_name) or model_deployment_name
        max_len_from_hf = ContaminationUtils._get_max_length_from_hf(tokenizer_name)
        if max_len_from_hf:
            hlog(f"INFO: Max length {max_len_from_hf} found in AutoTokenizer config for '{tokenizer_name}'")
            return max_len_from_hf

        hlog(f"INFO: Using default max length {default_max_len} for '{model_deployment_name}'")
        return default_max_len

    @staticmethod
    def check_prompt_length(
        prompt_text: str, model_deployment_name: str, tokenizer_service: TokenizerService, max_allowable_tokens: int
    ) -> Tuple[bool, int]:
        """
        Checks if a prompt exceeds the token limit using the primary TokenizerService.
        Raises RuntimeError on failure, allowing the caller to implement fallback logic.
        """
        tokenizer_name = ContaminationUtils.get_tokenizer_name(model_deployment_name)
        if not tokenizer_name:
            raise RuntimeError(f"Could not resolve tokenizer name for '{model_deployment_name}'")

        token_count = ContaminationUtils.get_token_count(prompt_text, tokenizer_name, tokenizer_service)
        return token_count <= max_allowable_tokens, token_count

    @staticmethod
    def check_prompt_length_fallback(prompt_text: str, max_allowable_tokens: int) -> Tuple[bool, int]:
        """Fallback for checking prompt length using GPT-2 or character heuristics."""
        num_tokens = -1
        gpt2_tokenizer = ContaminationUtils.get_gpt2_fallback_tokenizer()
        if gpt2_tokenizer:
            try:
                num_tokens = len(gpt2_tokenizer.encode(prompt_text, add_special_tokens=False))
            except Exception:
                num_tokens = -1

        if num_tokens == -1:
            num_tokens = (
                len(prompt_text) + ContaminationUtils.CONSERVATIVE_CHARS_PER_TOKEN_DIVISOR - 1
            ) // ContaminationUtils.CONSERVATIVE_CHARS_PER_TOKEN_DIVISOR

        return num_tokens <= max_allowable_tokens, num_tokens

    @staticmethod
    def get_tokenizer_name(model_deployment_name: str) -> Optional[str]:
        """Gets the specific tokenizer name from a ModelDeployment."""
        try:
            deployment = get_model_deployment(model_deployment_name)
            return deployment.tokenizer_name or None
        except KeyError:
            return None
        except Exception as e:
            hlog(f"ERROR: Unexpected error getting tokenizer name for '{model_deployment_name}': {e}")
            return None

    @staticmethod
    def get_token_count(text: str, tokenizer_name: str, tokenizer_service: TokenizerService) -> int:
        """Uses HELM's TokenizerService to get the token count. Raises RuntimeError on failure."""
        request = TokenizationRequest(text=text, tokenizer=tokenizer_name)
        result = tokenizer_service.tokenize(request)
        if result.success and result.tokens is not None:
            return len(result.tokens)
        raise RuntimeError(f"TokenizerService failed for '{tokenizer_name}': {getattr(result, 'error', 'Unknown')}")

    @staticmethod
    def get_spacy_tagger(language: str) -> Optional[spacy.language.Language]:
        """Loads and returns a cached spaCy language model for POS tagging."""
        normalized_lang = language.lower().split("_")[0].split("-")[0]
        if normalized_lang in ContaminationUtils._spacy_models_cache:
            return ContaminationUtils._spacy_models_cache[normalized_lang]

        model_name = ContaminationUtils.SPACY_MODEL_MAP.get(normalized_lang)
        if not model_name:
            hlog(f"WARNING: No spaCy model in SPACY_MODEL_MAP for language '{language}'.")
            ContaminationUtils._spacy_models_cache[normalized_lang] = None
            return None

        try:
            nlp = spacy.load(model_name, disable=["parser", "ner"])
            hlog(f"INFO: spaCy model '{model_name}' loaded successfully.")
        except OSError:
            hlog(f"INFO: spaCy model '{model_name}' not found. Attempting to download...")
            try:
                spacy.cli.download(model_name)
                nlp = spacy.load(model_name, disable=["parser", "ner"])
                hlog(f"INFO: spaCy model '{model_name}' loaded after download.")
            except Exception as e:
                hlog(f"CRITICAL: Failed to download or load spaCy model '{model_name}': {e}")
                nlp = None

        ContaminationUtils._spacy_models_cache[normalized_lang] = nlp
        return nlp

    @staticmethod
    def get_question_text(example: Any) -> str:
        """Extracts the main question text or context from a HELM Instance or dictionary."""

        # Primary: `instance.input.text`.
        if hasattr(example, "input") and hasattr(example.input, "text") and isinstance(example.input.text, str):
            return example.input.text

        # Fallback: Search common keys in dictionaries.
        if isinstance(example, dict):
            for key in [
                "question",
                "text",
                "prompt",
                "context",
                "passage",
                "sentence",
                "goal",
                "query",
                "input",
                "inputs",
                "problem",
                "story",
                "article",
            ]:
                if key in example and isinstance(example.get(key), str) and example[key].strip():
                    return example[key]
                if key == "question" and "stem" in example.get(key, {}) and isinstance(example[key]["stem"], str):
                    return example[key]["stem"]
        hlog(f"WARNING: Could not extract question text from example of type: {type(example)}.")
        return ""

    @staticmethod
    def get_choices(example: Any) -> List[str]:
        """Extracts a list of choices from a HELM Instance or a dictionary."""
        if hasattr(example, "references") and example.references:
            choices = [
                str(ref.output.text)
                for ref in example.references
                if hasattr(ref, "output") and hasattr(ref.output, "text")
            ]
            if choices:
                return choices
        if hasattr(example, "output_mapping") and isinstance(getattr(example, "output_mapping", None), dict):
            return [str(val) for val in example.output_mapping.values()]
        if isinstance(example, dict):
            for key in ["choices", "options", "endings"]:
                if key in example and isinstance(example.get(key), list):
                    return [str(item) for item in example[key]]
            # Fallback for lettered/numbered keys
            options = []
            for i in range(26):
                key = chr(ord("A") + i)
                if key in example:
                    options.append(str(example[key]))
                elif key.lower() in example:
                    options.append(str(example[key.lower()]))
                else:
                    break
            if options:
                return options
        return []

    @staticmethod
    def get_answer_index(example: Any) -> int:
        """Extracts the 0-based index of the correct answer from a HELM Instance or dict."""
        if hasattr(example, "references") and example.references:
            for i, ref in enumerate(example.references):
                if hasattr(ref, "tags") and "correct" in ref.tags:
                    return i
        if isinstance(example, dict):
            for key in ["label", "answer", "answerKey"]:
                if key in example:
                    val = example[key]
                    if isinstance(val, int):
                        return val
                    if isinstance(val, str):
                        val_norm = val.strip().lower()
                        if val_norm.isdigit():
                            return int(val_norm)
                        if len(val_norm) == 1 and "a" <= val_norm <= "z":
                            return ord(val_norm) - ord("a")
            if "answer" in example and "choices" in example and isinstance(example["answer"], str):
                try:
                    return example["choices"].index(example["answer"])
                except ValueError:
                    pass
        return -1

    @staticmethod
    def get_prompt_fragments(strategy_key: str, language: str) -> Dict[str, str]:
        """Loads prompt fragments for a given strategy and language, with fallback to English."""
        if strategy_key not in PROMPT_CONFIGS_MASTER:
            hlog(f"ERROR: Prompt configuration for '{strategy_key}' not found.")
            return {}
        lang_prompts = PROMPT_CONFIGS_MASTER[strategy_key]
        normalized_lang = language.lower().split("_")[0].split("-")[0]
        if normalized_lang in lang_prompts:
            return lang_prompts[normalized_lang]
        if "en" in lang_prompts:
            hlog(f"WARNING: Prompts for language '{language}' not found. Falling back to English.")
            return lang_prompts["en"]
        hlog(f"ERROR: No prompts for '{language}' and no English fallback for strategy '{strategy_key}'.")
        return {}

    @staticmethod
    def create_generation_adapter_spec(original_adapter_spec: Any, generation_params: Dict[str, Any]) -> Any:
        """Creates a new AdapterSpec configured for generation."""
        params = generation_params.copy()
        params["method"] = "generation"
        return replace(original_adapter_spec, **params)

    @staticmethod
    def format_helm_stats(
        calculated_metrics: Dict[str, float], strategy_metric_prefix: str, split: str = "test"
    ) -> List[Stat]:
        """Formats calculated metrics into a list of HELM Stat objects, preserving rounding."""
        stats: List[Stat] = []
        for name, value in calculated_metrics.items():
            value_rounded = np.round(value, 2)
            stat = Stat(
                name=MetricName(name=f"{strategy_metric_prefix} {name})", split=split),
                count=1,
                sum=value_rounded,
                sum_squared=np.round(value_rounded**2, 2),
                min=value_rounded,
                max=value_rounded,
                mean=value_rounded,
                variance=0.0,
                stddev=0.0,
            )
            stats.append(stat)
        return stats

    @staticmethod
    def get_gpt2_fallback_tokenizer() -> Optional[GPT2Tokenizer]:
        """Loads and caches a GPT-2 tokenizer for fallback length estimation."""
        if ContaminationUtils._gpt2_tokenizer_cache is None:
            try:
                hlog("INFO: Loading GPT-2 tokenizer for fallback length check.")
                ContaminationUtils._gpt2_tokenizer_cache = GPT2Tokenizer.from_pretrained("gpt2", trust_remote_code=True)
            except Exception as e:
                hlog(f"WARNING: Could not load GPT-2 fallback tokenizer: {e}")
                ContaminationUtils._gpt2_tokenizer_cache = False
        return ContaminationUtils._gpt2_tokenizer_cache if ContaminationUtils._gpt2_tokenizer_cache else None

    @classmethod
    def clear_caches(cls) -> None:
        """Clear all internal caches."""
        cls._gpt2_tokenizer_cache = None
        cls._spacy_models_cache.clear()
        hlog("INFO: ContaminationUtils caches cleared.")

    @staticmethod
    def _get_max_length_from_deployment(deployment: ModelDeployment) -> Optional[int]:
        """Helper to extract max_length from a ModelDeployment object."""
        if deployment.max_sequence_length and deployment.max_sequence_length > 0:
            return deployment.max_sequence_length
        if deployment.max_request_length and deployment.max_request_length > 0:
            return deployment.max_request_length
        return None

    @staticmethod
    def _get_max_length_from_hf(tokenizer_name: str) -> Optional[int]:
        """Helper to get max_length from a Hugging Face tokenizer config."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            max_len = getattr(tokenizer, "model_max_length", 0)
            del tokenizer
            if 0 < max_len < ContaminationUtils.MAX_REASONABLE_CONTEXT_LENGTH:
                return max_len
        except Exception as e:
            hlog(f"DEBUG: Could not load AutoTokenizer for '{tokenizer_name}' to find max length: {e}")
        return None
