from dataclasses import dataclass, field
from typing import List, Optional

from helm.common.image_generation_parameters import ImageGenerationParameters
from helm.common.reeval_parameters import REEvalParameters


# Adaptation methods
ADAPT_GENERATION: str = "generation"
ADAPT_CHAT: str = "chat"
ADAPT_LANGUAGE_MODELING: str = "language_modeling"
ADAPT_MULTIPLE_CHOICE_JOINT: str = "multiple_choice_joint"
ADAPT_MULTIPLE_CHOICE_JOINT_CHAIN_OF_THOUGHT: str = "multiple_choice_joint_chain_of_thought"
ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL: str = "multiple_choice_separate_original"
ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED: str = "multiple_choice_separate_calibrated"
ADAPT_RANKING_BINARY: str = "ranking_binary"
ADAPT_EHR_INSTRUCTION: str = "ehr_instruction"
ADAPT_MULTIPLE_CHOICE_SEPARATE_METHODS: List[str] = [
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED,
]

# Multimodal adaptation methods
ADAPT_GENERATION_MULTIMODAL: str = "generation_multimodal"
ADAPT_MULTIPLE_CHOICE_JOINT_MULTIMODAL: str = "multiple_choice_joint_multimodal"


@dataclass(frozen=True)
class Substitution:
    """Represents a regular expression search/replace."""

    source: str
    target: str


@dataclass(frozen=True)
class AdapterSpec:
    """
    Specifies how to take a `Scenario` (a list of `Instance`s) and produce a
    `ScenarioState` (set of `Request`s ). Instead of having free-form prompt
    hacking, we try to make the process more declarative and systematic.
    Note that an `Instance` could produce many `Request`s (e.g., one for each `Reference`).
    """

    method: str = ""
    """The high-level strategy for converting instances into a prompt for the language model."""

    global_prefix: str = ""
    """The string that is prepended to the entire prompt."""

    global_suffix: str = ""
    """The string that is appended to the entire prompt."""

    instructions: str = ""
    """The description of the task that is included at the very beginning of the prompt."""

    input_prefix: str = "Input: "
    """The string that is included before each input (e.g., 'Question:')."""

    input_suffix: str = "\n"
    """The string that is included after each input (e.g., '\\n')."""

    reference_prefix: str = "A. "
    """The string that is included before each reference (for multiple-choice questions)."""

    reference_suffix: str = "\n"
    """The string that is included after each reference (for multiple-choice questions)."""

    chain_of_thought_prefix: str = ""
    """The string that is included before each chain of thought. (e.g., 'Let\'s think step by step')"""

    chain_of_thought_suffix: str = "\n"
    """The string that is included after each chain of thought. (e.g., 'The correct answer is')"""

    output_prefix: str = "Output: "
    """The string that is included before the correct answer/predicted output (e.g., 'Answer:')."""

    output_suffix: str = "\n"
    """The string that is included after the correct answer/predicted output (e.g., '\\n')."""

    instance_prefix: str = "\n"
    """The string that is included before each instance (e.g., '\\n\\n')."""

    substitutions: List[Substitution] = field(default_factory=list, hash=False)
    """A list of regular expression substitutions (e.g., replacing '\\n' with ';\\n')
    to perform at the very end on the prompt."""

    max_train_instances: int = 5
    """Maximum number of training instances to include in the prompt (currently by randomly sampling)."""

    max_eval_instances: Optional[int] = None
    """Maximum number of instances to evaluate on (over all splits - test, valid, etc.)."""

    num_outputs: int = 5
    """Maximum number of possible outputs to generate by sampling multiple outputs."""

    num_train_trials: int = 1
    """Number of trials, where in each trial we choose an independent, random set of training instances.
    Used to compute variance."""

    num_trials: int = 1
    """Number of trials, where we query the model with the same requests, but different random seeds."""

    sample_train: bool = True
    """If true, randomly sample N training examples; if false, select N consecutive training examples"""

    # Decoding parameters (inherited by `Request`)

    model_deployment: str = ""
    """Name of the language model deployment (<host_organization>/<model name>) to send requests to."""

    model: str = ""
    """Name of the language model (<creator_organization>/<model name>) to send requests to."""

    temperature: float = 1
    """Temperature parameter used in generation."""

    max_tokens: int = 100
    """Maximum number of tokens to generate."""

    # Set hash=False to make `AdapterSpec` hashable
    stop_sequences: List[str] = field(default_factory=list, hash=False)
    """List of stop sequences. Output generation will be stopped if any stop sequence is encountered."""

    # Random string (used concretely to bypass cache / see diverse results)
    random: Optional[str] = None
    """Random seed (string), which guarantees reproducibility."""

    multi_label: bool = False
    """If true, for instances with multiple correct reference, the gold answer should be considered to be all
    of the correct references rather than any of the correct references."""

    image_generation_parameters: Optional[ImageGenerationParameters] = None
    """Parameters for image generation."""

    reeval_parameters: Optional[REEvalParameters] = None
    """Parameters for reeval evaluation."""

    # Set hash=False to make `AdapterSpec` hashable
    eval_splits: Optional[List[str]] = field(default=None, hash=False)
    """The splits from which evaluation instances will be drawn."""
