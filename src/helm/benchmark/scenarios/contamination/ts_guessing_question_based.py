from typing import List, Optional
import random
from dataclasses import replace

from helm.benchmark.scenarios.contamination.contamination_base import ContaminationStrategy
from helm.benchmark.scenarios.scenario import Instance, Reference, Output
from helm.common.hierarchical_logger import hlog
from helm.benchmark.scenarios.contamination.contamination_utils import ContaminationUtils


class TSGuessingQuestionBaseStrategy(ContaminationStrategy):
    """
    Implementation of the 'base' TS-Guessing strategy.
    Masks an important word (NOUN, ADJ, VERB) in a sentence and asks the model
    to predict the masked word. Contamination is measured via exact match.
    Reference: https://aclanthology.org/2024.naacl-long.482/
    """

    STRATEGY_NAME = "ts_guessing_question_base"
    STRATEGY_DISPLAY_NAME = "TS-Guessing Base"
    POS_TAGS_TO_MASK = ["NOUN", "ADJ", "VERB"]

    def transform_instances(
        self,
        instances: List[Instance],
    ) -> List[Instance]:
        """
        Applies single-word masking to each instance.
        Returns a list of new instances containing the masked prompt.
        """

        # Load the spaCy tagger for the specified language.
        tagger = ContaminationUtils.get_spacy_tagger(self.language)
        if not tagger:
            hlog(
                f"WARNING: Could not load spaCy tagger for '{self.language}'. "
                f"Returning original instances without transformation."
            )
            return instances

        # Load prompt components for the current language.
        prompt_components = ContaminationUtils.get_prompt_fragments(self.STRATEGY_NAME, self.language)
        if not prompt_components:
            hlog(f"WARNING: Could not load prompt components for '{self.language}'. " f"Returning original instances.")
            return instances

        # Transform each instance by applying the masking procedure.
        transformed_instances = []
        skipped = 0

        for instance in instances:
            try:
                masked_instance = self._mask_instance(instance, tagger, prompt_components)
                if masked_instance:
                    transformed_instances.append(masked_instance)
                else:
                    skipped += 1
            except Exception as e:
                hlog(f"ERROR while transforming instance {instance.id}: {e}")
                skipped += 1

        return transformed_instances

    def _mask_instance(self, instance: Instance, tagger, prompt_components) -> Optional[Instance]:
        """
        Masks an important word (NOUN, ADJ, VERB) in the instance's text.
        Returns a new instance with the masked prompt and the correct word as a reference.
        """

        # Extract text content from the question.
        text = ContaminationUtils.get_question_text(instance)
        if not text or len(text.split()) < 5:
            return None

        # Use spaCy to identify candidate tokens for masking.
        doc = tagger(text)
        candidate_tokens = [
            token
            for token in doc
            if token.pos_ in self.POS_TAGS_TO_MASK
            and not token.is_stop
            and not token.is_punct
            and len(token.text.strip()) > 1
        ]

        if not candidate_tokens:
            return None

        # Randomly select one candidate word to mask.
        selected_token = random.choice(candidate_tokens)
        word_to_mask = selected_token.text

        # Get character indices of the selected token.
        start_char = selected_token.idx
        end_char = start_char + len(word_to_mask)

        # Replace the selected word with [MASK].
        if not (0 <= start_char < end_char <= len(text)):
            # Fallback: simple replace method.
            masked_text = text.replace(word_to_mask, "[MASK]", 1)
            if masked_text == text:
                return None
        else:
            masked_text = text[:start_char] + "[MASK]" + text[end_char:]

        # Build the final prompt using the loaded template components.
        instruction = prompt_components.get("instruction", "Fill in the [MASK]:")
        sentence_template = prompt_components.get("sentence_template", '"{masked_sentence}"')
        answer_prefix = prompt_components.get("answer_prefix", "Answer:")

        sentence_formatted = sentence_template.format(masked_sentence=masked_text)
        final_prompt = f"{instruction}\n{sentence_formatted}\n\n{answer_prefix}"

        # Create a reference containing the original (masked) word as ground truth.
        masked_word_ref = Reference(output=Output(text=word_to_mask.strip()), tags=["correct"])

        # Create a new instance with the masked input and reference.
        new_input = replace(instance.input, text=final_prompt)
        new_instance = replace(instance, input=new_input, references=[masked_word_ref])

        return new_instance
