import spacy
import spacy.cli
from typing import Dict, List, Any, Optional
import spacy.language

from helm.common.hierarchical_logger import hlog
from helm.benchmark.scenarios.scenario import Instance
from .prompt_translations import TS_GUESSING_BASE, TS_GUESSING_MULTICHOICE

PROMPT_CONFIGS_MASTER = {
    "ts_guessing_question_base": TS_GUESSING_BASE,
    "ts_guessing_question_multichoice": TS_GUESSING_MULTICHOICE,
}


class ContaminationUtils:
    """
    Utilities for contamination detection strategies. Provides common functionalities for
    handling HELM objects, parsing instances, and managing NLP models (spaCy).
    """

    SPACY_MODEL_MAP: Dict[str, str] = {
        "en": "en_core_web_sm",
        "pt": "pt_core_news_sm",
        "zh": "zh_core_web_sm",
    }

    _spacy_models_cache: Dict[str, Optional[spacy.language.Language]] = {}

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
    def get_question_text(instance: Instance) -> str:
        """Extracts the main question text from a HELM Instance."""
        if hasattr(instance, "input") and hasattr(instance.input, "text"):
            return instance.input.text
        hlog(f"WARNING: Instance {getattr(instance, 'id', 'unknown')} has no input.text")
        return ""

    @staticmethod
    def get_choices(instance: Instance) -> List[str]:
        """Extracts a list of choices from a HELM Instance."""
        if not hasattr(instance, "references") or not instance.references:
            return []

        return [
            str(ref.output.text)
            for ref in instance.references
            if hasattr(ref, "output") and hasattr(ref.output, "text")
        ]

    @staticmethod
    def get_answer_index(instance: Instance) -> int:
        """Extracts the 0-based index of the correct answer from a HELM Instance."""
        if not hasattr(instance, "references") or not instance.references:
            return -1

        for i, ref in enumerate(instance.references):
            if hasattr(ref, "tags") and "correct" in ref.tags:
                return i

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

    @classmethod
    def clear_caches(cls) -> None:
        """Clear all internal caches."""
        cls._spacy_models_cache.clear()
        hlog("INFO: ContaminationUtils caches cleared.")
