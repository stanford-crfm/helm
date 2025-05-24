import spacy
import spacy.cli
import numpy as np
import importlib.util
from dataclasses import replace
from typing import Dict, List, Tuple, Any, Optional

from helm.common.hierarchical_logger import hlog
from helm.benchmark.model_deployment_registry import get_model_deployment, ModelDeployment
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.benchmark.metrics.metric import Stat
from helm.benchmark.metrics.metric_name import MetricName
from helm.common.tokenization_request import TokenizationRequest, TokenizationRequestResult

from .prompt_translations import TS_GUESSING_BASE, TS_GUESSING_MULTICHOICE

TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
AutoTokenizer_class: Optional[type] = None

if TRANSFORMERS_AVAILABLE:
    try:
        from transformers import AutoTokenizer
        AutoTokenizer_class = AutoTokenizer
    except ImportError as e_transformers_import:
        hlog(f"UTIL WARNING: 'transformers' module found but could not be imported. GPT-2 fallback might not work. Error: {e_transformers_import}")
        TRANSFORMERS_AVAILABLE = False


PROMPT_CONFIGS_MASTER = {
    "ts_guessing_question_base": TS_GUESSING_BASE,
    "ts_guessing_question_multichoice": TS_GUESSING_MULTICHOICE,
}


class UtilsContamination:
    """
    Utilities for contamination detection strategies.
    Provides common functionalities for handling HELM objects, model configurations,
    prompt management, and result formatting.
    """

    DEFAULT_MODEL_MAX_CONTEXT_TOKENS_UTIL: int = 4096
    AVG_CHARS_PER_TOKEN_HEURISTIC: int = 4
    CONSERVATIVE_CHARS_PER_TOKEN_DIVISOR: int = max(1, AVG_CHARS_PER_TOKEN_HEURISTIC -1)

    SPACY_MODEL_MAP: Dict[str, str] = {
        "en": "en_core_web_sm",
        "pt": "pt_core_news_sm",
        "zh": "zh_core_web_sm",
    }
    gpt2_tokenizer_fallback_cache: Optional[Any] = None

    @staticmethod
    def get_gpt2_fallback_tokenizer() -> Optional[Any]:
        """
        Loads and caches a GPT-2 tokenizer for fallback length estimation.
        Returns the tokenizer instance, None if transformers is not available,
        or False if loading failed previously.
        """
        if not TRANSFORMERS_AVAILABLE or AutoTokenizer_class is None:
            if UtilsContamination.gpt2_tokenizer_fallback_cache is None:
                 hlog("UTIL INFO: 'transformers' library not available or AutoTokenizer not imported. GPT-2 tokenizer fallback cannot be used.")
                 UtilsContamination.gpt2_tokenizer_fallback_cache = False
            return None

        if UtilsContamination.gpt2_tokenizer_fallback_cache is None:
            try:
                hlog("UTIL INFO: Loading GPT-2 tokenizer for fallback length check (first time).")
                UtilsContamination.gpt2_tokenizer_fallback_cache = AutoTokenizer_class.from_pretrained("gpt2", trust_remote_code=True)
            except Exception as e_load_gpt2:
                hlog(f"UTIL WARNING: Failed to load GPT-2 tokenizer for fallback: {e_load_gpt2}")
                UtilsContamination.gpt2_tokenizer_fallback_cache = False
        
        return UtilsContamination.gpt2_tokenizer_fallback_cache if UtilsContamination.gpt2_tokenizer_fallback_cache is not False else None


    @staticmethod
    def get_choices(example: Any) -> List[str]:
        """
        Extracts a list of choices from a HELM Instance or a dictionary.
        Tries various common structures for multiple-choice questions.

        Args:
            example (Any): The input object or dictionary containing choices.

        Returns:
            List[str]: A list of choice strings. Returns empty list if no choices found.
        """

        choices_found: List[str] = []

        if hasattr(example, "references") and example.references:
            choices_found = [
                str(ref.output.text)
                for ref in example.references
                if hasattr(ref, "output") and hasattr(ref.output, "text")
            ]
            if choices_found:
                return choices_found

        if hasattr(example, "output_mapping") and example.output_mapping and isinstance(example.output_mapping, dict):
            choices_found = [str(val) for val in example.output_mapping.values()]
            if choices_found:
                return choices_found

        if isinstance(example, dict):
            if "choices" in example:
                choices_val = example["choices"]
                if isinstance(choices_val, list):
                    return [str(c) for c in choices_val]
                elif isinstance(choices_val, dict) and "text" in choices_val and isinstance(choices_val["text"], list):
                    return [str(c) for c in choices_val["text"]]
            if "endings" in example and isinstance(example["endings"], list):
                return [str(e) for e in example["endings"]]
            if "options" in example and isinstance(example["options"], list):
                return [str(o) for o in example["options"]]

            alpha_keys = [chr(ord("A") + i) for i in range(26)]  # A-Z
            num_keys_0_idx = [str(i) for i in range(10)]  # 0-9

            current_options = []
            for key in alpha_keys:
                if key in example and isinstance(example[key], str):
                    current_options.append(example[key])
                elif key.lower() in example and isinstance(example[key.lower()], str):
                    current_options.append(example[key.lower()])
                elif current_options:
                    break
            if current_options:
                return current_options

            for key in num_keys_0_idx:
                if key in example and isinstance(example[key], str):
                    current_options.append(example[key])
                elif current_options:
                    break
            if current_options:
                return current_options

        # Fallback to optionN as object attributes (less common for dicts)
        if not isinstance(example, dict):
            for i in range(1, 7):
                opt_key = f"option{i}"
                if hasattr(example, opt_key):
                    val = getattr(example, opt_key)
                    if isinstance(val, str):
                        choices_found.append(val)
                    if not hasattr(example, f"option{i+1}") and choices_found:
                        break
                elif choices_found:
                    break
            if choices_found:
                return choices_found

        hlog(
            f"UTIL WARNING: Could not extract choices from example of type: {type(example)}. "
            f"Consider inspecting its structure. Keys/attrs might be: {list(example.keys()) if isinstance(example, dict) else dir(example)}"
        )
        return []

    @staticmethod
    def get_answer_index(example: Any) -> int:
        """
        Extracts the 0-based index of the correct answer from a HELM Instance or dict.
        Handles various ways correct answers are specified.

        Args:
            example (Any): A HELM instance or dictionary containing the answer.

        Returns:
            int: The 0-based index of the correct answer, or -1 if not found.
        """

        alphabet_options = "abcdefghijklmnopqrstuvwxyz"
        numeric_options_1_indexed = "123456789"

        if hasattr(example, "references") and example.references:
            for i, ref in enumerate(example.references):
                if hasattr(ref, "tags") and "correct" in ref.tags:
                    return i

        if (
            hasattr(example, "output_mapping")
            and example.output_mapping
            and hasattr(example, "references")
            and example.references
        ):
            correct_text_from_ref = None
            for ref in example.references:
                if (
                    hasattr(ref, "tags")
                    and "correct" in ref.tags
                    and hasattr(ref, "output")
                    and hasattr(ref.output, "text")
                ):
                    correct_text_from_ref = str(ref.output.text)
                    break
            if correct_text_from_ref and isinstance(example.output_mapping, dict):
                choices_list = UtilsContamination.get_choices(example)
                if choices_list:
                    try:
                        return choices_list.index(correct_text_from_ref)
                    except ValueError:
                        hlog(
                            f"UTIL WARNING: Correct text '{correct_text_from_ref}' from reference not found in choices extracted by get_choices."
                        )
                else:  # Fallback to try to interpret keys from output_mapping
                    for key, text_val in example.output_mapping.items():
                        if str(text_val) == correct_text_from_ref:
                            if isinstance(key, int):
                                return key
                            if isinstance(key, str):
                                key_lower = key.lower()
                                if key_lower.isdigit():
                                    return int(key_lower)
                                if len(key_lower) == 1 and key_lower in alphabet_options:
                                    return alphabet_options.index(key_lower)
                            hlog(
                                f"UTIL WARNING: Found correct text in output_mapping for key '{key}', but key is not a recognized index format."
                            )

        if isinstance(example, dict):
            if "answerKey" in example:
                key_val = str(example["answerKey"]).strip().lower()
                if len(key_val) == 1 and key_val in alphabet_options:
                    return alphabet_options.index(key_val)
                elif key_val in numeric_options_1_indexed:
                    return int(key_val) - 1
                elif key_val.isdigit():
                    return int(key_val)

            if "label" in example:
                try:
                    return int(example["label"])
                except (ValueError, TypeError):
                    hlog(
                        f"UTIL WARNING: Could not convert 'label' field ('{example['label']}') to int for answer index."
                    )

            if "answer" in example:
                ans_val = example["answer"]
                if isinstance(ans_val, int):
                    return ans_val
                if isinstance(ans_val, str):
                    key_ans = ans_val.strip().lower()
                    if len(key_ans) == 1 and key_ans in alphabet_options:
                        return alphabet_options.index(key_ans)
                    if key_ans in numeric_options_1_indexed:
                        return int(key_ans) - 1
                    if key_ans.isdigit():
                        return int(key_ans)

            if (
                "answer" in example
                and "choices" in example
                and isinstance(example["answer"], str)
                and isinstance(example["choices"], list)
            ):
                try:
                    return example["choices"].index(example["answer"])
                except ValueError:
                    hlog(f"UTIL WARNING: Answer text '{example['answer']}' not found in 'choices' list.")

        hlog(
            f"UTIL WARNING: Could not determine answer index for example of type: {type(example)}. "
            f"Consider inspecting its structure. Keys/attrs might be: {list(example.keys()) if isinstance(example, dict) else dir(example)}"
        )
        return -1

    @staticmethod
    def get_question_text(example: Any) -> str:
        """
        Extracts the main question text or context from a HELM Instance or dictionary.

        Args:
            example (Any): A HELM instance or dictionary.

        Returns:
            str: The question text/context or a default warning string.
        """

        if hasattr(example, "input") and hasattr(example.input, "text") and isinstance(example.input.text, str):
            return example.input.text

        if isinstance(example, dict):
            preferred_keys = [
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
            ]
            for key in preferred_keys:
                if key in example:
                    val = example[key]
                    if isinstance(val, str) and val.strip():
                        return val
                    if isinstance(val, dict) and "stem" in val and isinstance(val["stem"], str) and val["stem"].strip():
                        return val["stem"]

            other_potential_keys = ["problem", "story", "article"]
            for key in other_potential_keys:
                if key in example and isinstance(example[key], str) and example[key].strip():
                    return example[key]

        hlog(
            f"UTIL WARNING: Could not extract question text from example of type: {type(example)}. "
            f"Consider inspecting its structure. Keys/attrs might be: {list(example.keys()) if isinstance(example, dict) else dir(example)}"
        )
        return "Unknown question or context"

    @staticmethod
    def get_prompt_fragments(strategy_key: str, language: str) -> Dict[str, str]:
        """
        Loads prompt fragments for a given strategy and language.
        Falls back to English if the specified language is not found.

        Args:
            strategy_key (str): Key identifying the contamination strategy (e.g., "base", "multichoice").
                                Should match keys in PROMPT_CONFIGS_MASTER.
            language (str): Language code (e.g., "en", "pt").

        Returns:
            Dict[str, str]: A dictionary with prompt fragments. Empty if not found.
        """

        if strategy_key not in PROMPT_CONFIGS_MASTER:
            hlog(
                f"UTIL ERROR: Prompt configuration for strategy_key '{strategy_key}' not found in PROMPT_CONFIGS_MASTER. "
                f"Available keys: {list(PROMPT_CONFIGS_MASTER.keys())}"
            )
            return {}

        lang_prompts_for_strategy = PROMPT_CONFIGS_MASTER[strategy_key]
        normalized_lang = language.lower().split("_")[0].split("-")[0]

        if normalized_lang in lang_prompts_for_strategy:
            return lang_prompts_for_strategy[normalized_lang]
        elif "en" in lang_prompts_for_strategy:
            hlog(
                f"UTIL WARNING: Language '{language}' (normalized to '{normalized_lang}') not found for strategy '{strategy_key}'. "
                "Falling back to English prompts."
            )
            return lang_prompts_for_strategy["en"]
        else:
            hlog(
                f"UTIL ERROR: English fallback prompts not found for strategy '{strategy_key}' when language '{language}' (normalized: {normalized_lang}) is missing."
            )
            return {}

    @staticmethod
    def determine_model_max_length(
        model_deployment_name: str, default_max_len: int = DEFAULT_MODEL_MAX_CONTEXT_TOKENS_UTIL
    ) -> int:
        """
        Determines the maximum sequence length for a model.
        Prioritizes ModelDeployment, then AutoTokenizer, then a default.

        Args:
            model_deployment_name (str): Name of the deployed model from HELM's registry.
            default_max_len (int, optional): Default context length if detection fails.
                                             Defaults to UtilsContamination.DEFAULT_MODEL_MAX_CONTEXT_TOKENS_UTIL.

        Returns:
            int: The determined maximum number of tokens for the model's context.
        """

        model_max_len = default_max_len
        primary_source_found = False
        try:
            model_deployment: ModelDeployment = get_model_deployment(model_deployment_name)
            if model_deployment.max_sequence_length is not None and model_deployment.max_sequence_length > 0:
                model_max_len = model_deployment.max_sequence_length
                primary_source_found = True
                hlog(
                    f"UTIL INFO: Using model_max_length from ModelDeployment.max_sequence_length: {model_max_len} for {model_deployment_name}"
                )
            elif model_deployment.max_request_length is not None and model_deployment.max_request_length > 0:
                model_max_len = model_deployment.max_request_length
                primary_source_found = True
                hlog(
                    f"UTIL INFO: Using model_max_length from ModelDeployment.max_request_length: {model_max_len} for {model_deployment_name}"
                )

            if not primary_source_found:
                hlog(
                    f"UTIL WARNING: max_sequence_length/max_request_length not set or invalid in ModelDeployment for '{model_deployment_name}'. Attempting fallback to AutoTokenizer."
                )

        except KeyError as e_model_reg:
            hlog(
                f"UTIL INFO: Could not get ModelDeployment (model not in registry or other KeyError) for '{model_deployment_name}': {e_model_reg}. Falling back to AutoTokenizer."
            )
        except Exception as e_model_reg_other:
            hlog(
                f"UTIL WARNING: Error accessing ModelDeployment for '{model_deployment_name}': {e_model_reg_other}. Falling back to AutoTokenizer."
            )

        if not primary_source_found:
            temp_tokenizer_for_max_len = None
            try:
                from transformers import AutoTokenizer

                hlog(
                    f"UTIL INFO: Fallback: Loading AutoTokenizer for '{model_deployment_name}' to get model_max_length."
                )
                temp_tokenizer_for_max_len = AutoTokenizer.from_pretrained(
                    model_deployment_name, trust_remote_code=True
                )

                tokenizer_max_len_attr = getattr(temp_tokenizer_for_max_len, "model_max_length", model_max_len)

                if (
                    isinstance(tokenizer_max_len_attr, int)
                    and tokenizer_max_len_attr > 0
                    and tokenizer_max_len_attr < 2_000_000
                ):
                    model_max_len = tokenizer_max_len_attr
                    hlog(
                        f"UTIL INFO: Using model_max_length from AutoTokenizer.model_max_length: {model_max_len} for {model_deployment_name}"
                    )
                else:
                    hlog(
                        f"UTIL WARNING: Fallback AutoTokenizer for {model_deployment_name} reported an unusual max length ({tokenizer_max_len_attr}). "
                        f"Using previously determined or default value: {model_max_len}."
                    )
            except ImportError:
                hlog(
                    f"UTIL WARNING: 'transformers' library not installed. Cannot use AutoTokenizer fallback for '{model_deployment_name}'. Using: {model_max_len}."
                )
            except Exception as e_hf_tokenizer:
                hlog(
                    f"UTIL WARNING: Fallback using AutoTokenizer for '{model_deployment_name}' also failed: {e_hf_tokenizer}. "
                    f"Using previously determined or default value: {model_max_len}."
                )
            finally:
                if temp_tokenizer_for_max_len:
                    del temp_tokenizer_for_max_len

        if not (isinstance(model_max_len, int) and model_max_len > 0):
            hlog(
                f"UTIL WARNING: Determined model_max_length ({model_max_len}) for '{model_deployment_name}' is invalid. Resetting to util default: {UtilsContamination.DEFAULT_MODEL_MAX_CONTEXT_TOKENS_UTIL}."
            )
            model_max_len = UtilsContamination.DEFAULT_MODEL_MAX_CONTEXT_TOKENS_UTIL
        return model_max_len

    @staticmethod
    def create_generation_adapter_spec(original_adapter_spec: Any, generation_method_params: Dict[str, Any]) -> Any:
        """
        Creates a new AdapterSpec configured for generation, updating specified parameters.
        Ensures 'method' is always set to 'generation'.

        Args:
            original_adapter_spec (Any): The base AdapterSpec (or any object with similar structure that supports dataclasses.replace).
            generation_method_params (Dict[str, Any]): Parameters to override/set in the new AdapterSpec.
                                                        'method' will be set to 'generation'. Any other
                                                        AdapterSpec fields (like instructions, input_prefix, etc.)
                                                        should be explicitly provided in this dict if they
                                                        need to be changed from the original_adapter_spec
                                                        or set to specific values for generation.

        Returns:
            Any: A new AdapterSpec-like instance.
        """

        params_to_update = generation_method_params.copy()
        params_to_update["method"] = "generation"

        return replace(original_adapter_spec, **params_to_update)

    @staticmethod
    def check_prompt_length_tokenization_request(
        prompt_text: str,
        model_name_for_tokenizer: str,
        tokenizer_service: TokenizerService,
        max_allowable_prompt_tokens: int,
    ) -> Tuple[bool, int]:
        """
        Checks if the tokenized prompt_text fits within max_allowable_prompt_tokens using HELM TokenizerService.
        """

        if not isinstance(prompt_text, str):
            hlog(f"UTIL ERROR: prompt_text must be a string, got {type(prompt_text)}. Length check aborted.")
            return False, -1 

        if max_allowable_prompt_tokens <= 0:
            hlog(f"UTIL WARNING: max_allowable_prompt_tokens ({max_allowable_prompt_tokens}) is non-positive.")
            return False, 0
        
        try:
            tokenization_request = TokenizationRequest(text=prompt_text, tokenizer=model_name_for_tokenizer)
            tokenization_result: TokenizationRequestResult = tokenizer_service.tokenize(tokenization_request)

            if tokenization_result.success and tokenization_result.tokens is not None:
                num_prompt_tokens = len(tokenization_result.tokens)
                fits_within_limit = num_prompt_tokens <= max_allowable_prompt_tokens
                return fits_within_limit, num_prompt_tokens
            else:
                error_msg = getattr(tokenization_result, 'error', 'Unknown error')
                hlog(f"HELM TokenizerService failed for '{model_name_for_tokenizer}': {error_msg}. Switching to fallback.")
                raise Exception(f"HELM tokenization failed: {error_msg}")
                
        except Exception as e_helm_service:
            hlog(f"HELM TokenizerService exception for '{model_name_for_tokenizer}': {type(e_helm_service).__name__}. Switching to fallback.")
            raise

    @staticmethod  
    def check_prompt_length_fallback_gpt2(
        prompt_text: str,
        model_name_for_tokenizer: str,
        max_allowable_prompt_tokens: int,
    ) -> Tuple[bool, int]:
        """
        Fallback tokenization using GPT-2 tokenizer and character heuristic.
        """
        
        num_prompt_tokens = -1

        # Try GPT-2 tokenizer first
        gpt2_tokenizer = UtilsContamination.get_gpt2_fallback_tokenizer()
        if gpt2_tokenizer:
            try:
                num_prompt_tokens = len(gpt2_tokenizer.encode(prompt_text, add_special_tokens=False))
            except Exception as e_gpt2_fallback:
                hlog(f"GPT-2 Fallback failed for '{model_name_for_tokenizer}': {type(e_gpt2_fallback).__name__}. Using character heuristic.")
                num_prompt_tokens = -1

        # Character heuristic fallback
        if num_prompt_tokens == -1:
            try:
                if not prompt_text:
                    num_prompt_tokens = 0
                else:
                    num_prompt_tokens = (len(prompt_text) + UtilsContamination.CONSERVATIVE_CHARS_PER_TOKEN_DIVISOR - 1) // UtilsContamination.CONSERVATIVE_CHARS_PER_TOKEN_DIVISOR
            except Exception as e_char_fallback:
                hlog(f"Character heuristic failed for '{model_name_for_tokenizer}': {type(e_char_fallback).__name__}.")
                return False, -1 

        if num_prompt_tokens == -1:
            hlog(f"All tokenization methods failed for '{model_name_for_tokenizer}'.")
            return False, -1

        fits_within_limit = num_prompt_tokens <= max_allowable_prompt_tokens
        return fits_within_limit, num_prompt_tokens

    @staticmethod
    def format_helm_stats(
        calculated_metrics: Dict[str, float], strategy_metric_name_prefix: str, split: str = "test"
    ) -> List[Stat]:
        """
        Formats calculated metrics into a list of HELM Stat objects.

        Args:
            calculated_metrics (Dict[str, float]): Metric names and values.
            strategy_metric_name_prefix (str): Prefix for each metric name.
            split (str, optional): Dataset split (e.g., "test").

        Returns:
            List[Stat]: List of Stat objects.
        """

        final_helm_stats: List[Stat] = []
        for metric_name_suffix, metric_value in calculated_metrics.items():
            metric_value_rounded = np.round(metric_value, 2)
            sum_squared_rounded = np.round(metric_value_rounded**2, 2)

            full_metric_name_str = f"{strategy_metric_name_prefix} {metric_name_suffix})"
            
            metric_name_obj = MetricName(name=full_metric_name_str, split=split)

            stat_obj = Stat(
                name=metric_name_obj,
                count=1,
                sum=metric_value_rounded,
                sum_squared=sum_squared_rounded,
                min=metric_value_rounded,
                max=metric_value_rounded,
                mean=metric_value_rounded,
                variance=0.0,
                stddev=0.0
            )
            final_helm_stats.append(stat_obj)
            
        return final_helm_stats

    @staticmethod
    def get_spacy_tagger(language: str) -> Any:
        """
        Loads and returns a spaCy language model for POS tagging.
        Attempts to download the model if not found.
        Disables unnecessary components (parser, NER) for speed.

        Args:
            language (str): Language code (e.g., "en", "pt", "zh").

        Returns:
            spacy.language.Language: A spaCy language model instance.

        Raises:
            ValueError: If the language code is not supported or model name cannot be determined.
            ImportError: If spaCy library is not installed.
            Exception: For other errors during model download or loading.
        """

        normalized_lang = language.lower().split("_")[0].split("-")[0]
        model_name = UtilsContamination.SPACY_MODEL_MAP.get(normalized_lang)

        if not model_name:
            hlog(
                f"UTIL WARNING: No spaCy model mapped in SPACY_MODEL_MAP for language '{language}' "
                f"(normalized to '{normalized_lang}'). Falling back to English model."
            )
            normalized_lang = "en"
            model_name = UtilsContamination.SPACY_MODEL_MAP.get(normalized_lang)
            
            if not model_name:
                hlog("UTIL ERROR: English fallback model not found in SPACY_MODEL_MAP.")
                raise ValueError("English fallback model not configured in SPACY_MODEL_MAP")

        try:
            hlog(f"UTIL INFO: Attempting to load spaCy model: '{model_name}' for language '{normalized_lang}'")

            model_is_installed = False
            try:
                spacy.load(model_name, disable=["parser", "ner"])
                model_is_installed = True
                hlog(f"UTIL INFO: spaCy model '{model_name}' found and loadable.")
            except OSError:
                hlog(f"UTIL INFO: spaCy model '{model_name}' not found by spacy.load(). Attempting download...")

            if not model_is_installed:
                try:
                    spacy.cli.download(model_name)
                    hlog(f"UTIL INFO: spaCy model '{model_name}' downloaded successfully via spacy.cli.download.")
                except SystemExit as e_download:
                    if e_download.code == 0:
                        hlog(
                            f"UTIL INFO: spacy.cli.download for '{model_name}' exited with code 0 (likely success or already present)."
                        )
                    else:
                        hlog(
                            f"UTIL ERROR: Failed to download spaCy model '{model_name}'. spacy.cli.download exited with code: {e_download.code}"
                        )
                        raise Exception(f"Failed to download spaCy model {model_name}") from e_download
                except Exception as e_download_other:
                    hlog(
                        f"UTIL ERROR: An unexpected error occurred during spaCy model download for '{model_name}': {e_download_other}"
                    )
                    raise Exception(f"Unexpected error downloading spaCy model {model_name}") from e_download_other

            return spacy.load(model_name, disable=["parser", "ner"])

        except ImportError:
            hlog("UTIL CRITICAL: spaCy library not installed. Please install it: pip install spacy")
            raise
        except Exception as e:
            hlog(f"UTIL CRITICAL: Failed to ensure availability of or load spaCy model '{model_name}': {e}")
            raise Exception(f"Could not load or download spaCy model {model_name}") from e