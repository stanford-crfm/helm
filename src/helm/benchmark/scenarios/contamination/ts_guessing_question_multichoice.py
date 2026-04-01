from typing import List, Optional
import numpy as np
from dataclasses import replace

from helm.benchmark.scenarios.contamination.contamination_base import ContaminationStrategy
from helm.benchmark.scenarios.scenario import Instance, Reference, Output
from helm.common.hierarchical_logger import hlog
from helm.benchmark.scenarios.contamination.contamination_utils import ContaminationUtils


class TSGuessingQuestionMultiChoiceStrategy(ContaminationStrategy):
    """
    Implements a question-based multi-choice guessing test for contamination detection.
    Prompts are constructed with a randomly masked incorrect option. The model is asked
    to predict the content of the mask. This prediction is then compared against the
    original text of that specific masked (incorrect) option.
    Reference: https://aclanthology.org/2024.naacl-long.482/
    """

    STRATEGY_NAME = "ts_guessing_question_multichoice"
    STRATEGY_DISPLAY_NAME = "TS-Guessing Multichoice"

    def __init__(self):
        """
        Initializes the TS-Guessing multichoice contamination strategy.
        """
        super().__init__()
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def transform_instances(
        self,
        instances: List[Instance],
    ) -> List[Instance]:
        """
        Applies masking to one incorrect option in multiple-choice questions.
        Generates modified instances containing the masked prompt and a reference
        with the original text of the masked option.
        """

        # Load prompt components based on language configuration
        prompt_components = ContaminationUtils.get_prompt_fragments(self.STRATEGY_NAME, self.language)
        if not prompt_components:
            hlog("WARNING: Could not load prompt components. Returning original instances.")
            return instances

        transformed_instances = []
        skipped = 0

        # Apply masking transformation to each instance
        for instance in instances:
            try:
                masked_instance = self._mask_instance(instance, prompt_components)
                if masked_instance:
                    transformed_instances.append(masked_instance)
                else:
                    skipped += 1
            except Exception as e:
                hlog(f"ERROR while transforming instance {instance.id}: {e}")
                skipped += 1

        return transformed_instances

    def _mask_instance(self, instance: Instance, prompt_components) -> Optional[Instance]:
        """
        Masks one incorrect option in a multiple-choice instance.
        Creates a new instance where the masked prompt replaces the original question,
        and the original masked text is stored as a reference.
        """

        # Extract necessary information from the instance
        question_text = ContaminationUtils.get_question_text(instance)
        choices = ContaminationUtils.get_choices(instance)
        correct_idx = ContaminationUtils.get_answer_index(instance)

        # Validate instance data
        if not question_text or not choices or len(choices) <= 1 or correct_idx < 0 or correct_idx >= len(choices):
            return None

        # Identify incorrect options (eligible for masking)
        incorrect_indices = [i for i in range(len(choices)) if i != correct_idx]
        if not incorrect_indices:
            return None

        # Randomly select one incorrect option to mask
        masked_idx = np.random.choice(incorrect_indices)
        masked_letter = self.alphabet[masked_idx % len(self.alphabet)]

        # Store the original text of the masked option (ground truth)
        masked_option_text = str(choices[masked_idx])

        # Build prompt with masked option
        prompt = self._build_masked_prompt(question_text, choices, masked_idx, masked_letter, prompt_components)

        # Create a reference object with the original (unmasked) text
        masked_option_ref = Reference(output=Output(text=masked_option_text.strip()), tags=["correct"])

        # Create a new instance with the modified input and reference
        new_input = replace(instance.input, text=prompt)
        new_instance = replace(instance, input=new_input, references=[masked_option_ref])  # Ground truth for comparison

        return new_instance

    def _build_masked_prompt(
        self, question: str, choices: List[str], masked_idx: int, masked_letter: str, prompt_components: dict
    ) -> str:
        """
        Constructs a prompt where one incorrect option is replaced with [MASK].
        Combines prompt components with the question and options to form the
        complete masked prompt for model evaluation.
        """

        # Prompt template components
        instr_fill = prompt_components.get("instruction_fill_option", "Fill in [MASK] in option")
        instr_knowledge = prompt_components.get("instruction_knowledge", "Use your knowledge.")
        instr_rule = prompt_components.get("instruction_rule", "Provide only the masked text.")
        header_q = prompt_components.get("header_question", "Question:")
        header_opts = prompt_components.get("header_options", "Options:")
        footer = prompt_components.get("footer_reply", "Answer:")
        answer_prefix = prompt_components.get("answer_prefix", "Answer: ")

        # Instruction section
        instruction = f"{instr_fill} {masked_letter} {instr_knowledge}\n{instr_rule}\n"

        # Build options list, masking the selected incorrect option
        options_text = ""
        for i, choice in enumerate(choices):
            if i >= len(self.alphabet):
                break
            letter = self.alphabet[i]
            content = "[MASK]" if i == masked_idx else str(choice)
            options_text += f"\n{letter}: {content}"

        # Assemble the complete prompt
        prompt = (
            f"{instruction}\n"
            f"{header_q} {question}\n"
            f"{header_opts}{options_text}\n"
            f"{footer}\n\n"
            f"{answer_prefix}"
        )

        return prompt
