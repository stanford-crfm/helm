from abc import ABC, abstractmethod
from dataclasses import replace
from typing import List, Dict, Optional, Tuple, Type

from helm.proxy.models import (
    get_all_instruction_following_models,
    get_all_code_models,
    get_all_models,
    get_all_text_models,
    get_model_names_with_tag,
    FULL_FUNCTIONALITY_TEXT_MODEL_TAG,
    LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
    GPT2_TOKENIZER_TAG,
    AI21_TOKENIZER_TAG,
    COHERE_TOKENIZER_TAG,
    OPT_TOKENIZER_TAG,
    GPTJ_TOKENIZER_TAG,
    GPTNEO_TOKENIZER_TAG,
    GPT4_TOKENIZER_TAG,
    ABLATION_MODEL_TAG,
)
from .runner import RunSpec
from helm.benchmark.adaptation.adapter_spec import AdapterSpec, Substitution
from .augmentations.perturbation import PerturbationSpec
from .augmentations.data_augmenter import DataAugmenterSpec


class RunExpander(ABC):
    """
    A `RunExpander` takes a `RunSpec` and returns a list of `RunSpec`s.
    For example, it might fill out the `model` field with a variety of different models.
    """

    name: str

    @abstractmethod
    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        pass


class ReplaceValueRunExpander(RunExpander):
    """
    Replace a single field (e.g., max_train_instances) with a list of values (e.g., 0, 1, 2).
    """

    def __init__(self, value):
        """
        `value` is either the actual value to use or a lookup into the values dict.
        """
        self.name = type(self).name
        if value in type(self).values_dict:
            self.values = type(self).values_dict[value]
        else:
            self.values = [value]

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        def sanitize(value):
            return str(value).replace("/", "_")

        return [
            replace(
                run_spec,
                name=f"{run_spec.name}{',' if ':' in run_spec.name else ':'}{self.name}={sanitize(value)}",
                adapter_spec=replace(run_spec.adapter_spec, **{self.name: value}),
            )
            for value in self.values
        ]


class ReplaceRunSpecValueRunExpander(RunExpander):
    """
    Replace a single field (e.g., max_train_instances) with a list of values (e.g., 0, 1, 2).
    """

    def __init__(self, value):
        """
        `value` is either the actual value to use or a lookup into the values dict.
        """
        self.name = type(self).name
        if value in type(self).values_dict:
            self.values = type(self).values_dict[value]
        else:
            self.values = [value]

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        def sanitize(value):
            return str(value).replace("/", "_")

        return [
            replace(
                run_spec,
                name=f"{run_spec.name},{self.name}={sanitize(value)}",
                metrics=value,
            )
            for value in self.values
        ]


class InstructionsRunExpander(RunExpander):
    """
    Set the instructions of the prompt.
    """

    name = "instructions"

    def __init__(self, value):
        self.value = value

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        adapter_spec = run_spec.adapter_spec
        if self.value == "none":
            adapter_spec = replace(
                adapter_spec,
                instructions="",
            )
        elif self.value == "expert":
            adapter_spec = replace(
                adapter_spec,
                instructions="I am an expert AI assistant who is here to help you with the following. "
                + adapter_spec.instructions,
            )
        else:
            raise Exception("Unknown value: {self.value}")
        return [
            replace(run_spec, name=f"{run_spec.name},{self.name}={self.value}", adapter_spec=adapter_spec),
        ]


class PromptRunExpander(RunExpander):
    """
    Set the prompt.
    """

    name = "prompt"

    def __init__(self, value):
        self.value = value

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        adapter_spec = run_spec.adapter_spec
        if self.value == "human_assistant":
            adapter_spec = replace(
                adapter_spec,
                input_prefix='Human: What is the answer to "',
                input_suffix='"?\n',
                output_prefix='Assistant: The answer is "',
                output_suffix='".\n',
                stop_sequences=['".'],
            )
        elif self.value == "question_answer":
            adapter_spec = replace(adapter_spec, input_prefix="Question: ", output_prefix="Answer: ")
        elif self.value == "qa":
            adapter_spec = replace(adapter_spec, input_prefix="Q: ", output_prefix="A: ")
        elif self.value == "input_output_html":
            adapter_spec = replace(
                adapter_spec,
                input_prefix="<input>",
                input_suffix="</input>\n",
                output_prefix="<output>",
                output_suffix="</output>\n",
            )
        elif self.value == "input_output":
            adapter_spec = replace(adapter_spec, input_prefix="Input: ", output_prefix="Output: ")
        elif self.value == "i_o":
            adapter_spec = replace(adapter_spec, input_prefix="I: ", output_prefix="O: ")
        else:
            raise Exception("Unknown value: {self.value}")
        return [
            replace(
                run_spec,
                name=f"{run_spec.name},{self.name}={self.value}",
                adapter_spec=adapter_spec,
            ),
        ]


class NewlineRunExpander(RunExpander):
    """
    Set the newline delimiter (what's inserted before each newline).
    """

    name = "newline"

    def __init__(self, value):
        self.value = value

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        adapter_spec = run_spec.adapter_spec
        if self.value == "semicolon":
            adapter_spec = replace(
                adapter_spec,
                substitutions=[Substitution("\n", ";\n")],
            )
        elif self.value == "br":
            adapter_spec = replace(
                adapter_spec,
                substitutions=[Substitution("\n", "<br>\n")],
            )
        else:
            raise Exception("Unknown value: {self.value}")
        return [
            replace(
                run_spec,
                name=f"{run_spec.name},{self.name}={self.value}",
                adapter_spec=adapter_spec,
            ),
        ]


class StopRunExpander(RunExpander):
    """
    Set the stop sequence to something (e.g., ###) with new lines.
    """

    name = "stop"

    def __init__(self, value):
        """
        Args:
            value(str): Either the actual value to use or a lookup into the values dict.
        """
        self.value = value

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        if self.value == "hash":
            stop = "###"
        elif self.value == "semicolon":
            stop = ";"
        elif self.value == "br":
            stop = "<br>"
        else:
            raise Exception(f"Unknown value: {self.value}")
        return [
            replace(
                run_spec,
                name=f"{run_spec.name},{self.name}={self.value}",
                adapter_spec=replace(run_spec.adapter_spec, instance_prefix=f"{stop}\n\n", stop_sequences=[stop]),
            ),
        ]


class AddToStopRunExpander(RunExpander):
    """
    Add a stop sequence to the stop sequences. (Not like StopRunExpander, which replaces the stop sequences.)
    """

    name = "add_to_stop"

    def __init__(self, value):
        """
        Args:
            value(str): Either the actual value to use or a lookup into the values dict.
        """
        self.value = value

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        return [
            replace(
                run_spec,
                name=run_spec.name,
                adapter_spec=replace(
                    run_spec.adapter_spec, stop_sequences=run_spec.adapter_spec.stop_sequences + [self.value]
                ),
            ),
        ]


class GlobalPrefixRunExpander(RunExpander):
    """For overriding global prefix for specific models."""

    name = "global_prefix"

    def __init__(self, value):
        self.value = value

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        if self.value == "nlg":
            prefix = "[NLG]"
        else:
            raise Exception(f"Unknown value: {self.value}")

        return [
            replace(
                run_spec,
                name=f"{run_spec.name},{self.name}={self.value}",
                adapter_spec=replace(run_spec.adapter_spec, global_prefix=prefix),
            )
        ]


class FormatPromptRunExpander(RunExpander):
    """Adds a prefix and suffix to the prompt."""

    name = "format_prompt"

    def __init__(self, prefix: str = "", suffix: str = ""):
        self.prefix = prefix
        self.suffix = suffix

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        return [
            replace(
                run_spec,
                name=run_spec.name,
                adapter_spec=replace(
                    run_spec.adapter_spec,
                    global_prefix=self.prefix,
                    output_prefix=self.suffix,
                ),
            ),
        ]


class NumTrainTrialsRunExpander(ReplaceValueRunExpander):
    """For estimating variance across runs."""

    name = "num_train_trials"
    values_dict = {
        "1": [1],
        "2": [2],
        "3": [3],
        "4": [4],
        "5": [5],
    }


class MaxTrainInstancesRunExpander(ReplaceValueRunExpander):
    """For getting learning curves."""

    name = "max_train_instances"
    values_dict = {
        "zero": [0],
        "one": [1],
        "all": [0, 1, 2, 4, 8, 16],  # Cap at 16 due to limited context length
        "big_bench_few_shot_setting": [0, 1, 2, 3],  # Commonly used few-shot setting in BIG-bench
    }


class NumOutputsRunExpander(ReplaceValueRunExpander):
    """For overriding num_outputs."""

    name = "num_outputs"
    values_dict = {
        "default": [1],
        "copyright_sweep": [1, 10],
    }


class ModelRunExpander(ReplaceValueRunExpander):
    """
    For specifying different models.
    Note: we're assuming we don't have to change the decoding parameters for different models.
    """

    name = "model"

    def __init__(self, value):
        """
        `value` is either the actual value to use or a lookup into the values dict.
        """
        if value in self.values_dict:
            self.values = self.values_dict[value]
        else:
            self.values = [value]

    @property
    def values_dict(self):
        values_dict = {
            "full_functionality_text": get_model_names_with_tag(FULL_FUNCTIONALITY_TEXT_MODEL_TAG),
            "ai21/j1-jumbo": ["ai21/j1-jumbo"],
            "openai/curie": ["openai/curie"],
            "chat_run": ["openai/chat-gpt", "openai/text-davinci-003"],  # Compare ChatGPT to text-davinci-003
            "all": get_all_models(),
            "text_code": get_all_text_models() + get_all_code_models(),
            "text": get_all_text_models(),
            "code": get_all_code_models(),
            "instruction_following": get_all_instruction_following_models(),
            "limited_functionality_text": get_model_names_with_tag(LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG),
            "gpt2_tokenizer": get_model_names_with_tag(GPT2_TOKENIZER_TAG),
            "ai21_tokenizer": get_model_names_with_tag(AI21_TOKENIZER_TAG),
            "cohere_tokenizer": get_model_names_with_tag(COHERE_TOKENIZER_TAG),
            "opt_tokenizer": get_model_names_with_tag(OPT_TOKENIZER_TAG),
            "summarization_zs": ["openai/davinci", "openai/curie", "openai/text-davinci-002", "openai/text-curie-001"],
            "biomedical": ["openai/text-davinci-003"],  # TODO: add https://huggingface.co/stanford-crfm/BioMedLM
            "interactive_qa": ["openai/text-davinci-001", "openai/davinci", "ai21/j1-jumbo", "openai/text-babbage-001"],
            "opinions_qa_openai": [
                "openai/ada",
                "openai/davinci",
                "openai/text-ada-001",
                "openai/text-davinci-001",
                "openai/text-davinci-002",
                "openai/text-davinci-003",
            ],
            "opinions_qa_ai21": ["ai21/j1-grande", "ai21/j1-jumbo", "ai21/j1-grande-v2-beta"],
        }

        # For each of the keys above (e.g., "text"), create a corresponding ablation (e.g., "ablation_text")
        # which contains the subset of models with the ablation tag.
        ablation_models = set(get_model_names_with_tag(ABLATION_MODEL_TAG))
        ablation_values_dict = {}
        for family_name, models in values_dict.items():
            ablation_values_dict["ablation_" + family_name] = list(ablation_models & set(models))
        for family_name, models in ablation_values_dict.items():
            if family_name == "ablation_all":
                values_dict["ablation"] = models
            else:
                values_dict[family_name] = models
        return values_dict


############################################################


# Helper functions to instantiate `PerturbationSpec`s.
def extra_space(num_spaces: int) -> PerturbationSpec:
    return PerturbationSpec(
        class_name="helm.benchmark.augmentations.extra_space_perturbation.ExtraSpacePerturbation",
        args={"num_spaces": num_spaces},
    )


def space(max_spaces: int) -> PerturbationSpec:
    return PerturbationSpec(
        class_name="helm.benchmark.augmentations.space_perturbation.SpacePerturbation",
        args={"max_spaces": max_spaces},
    )


def lower() -> PerturbationSpec:
    return PerturbationSpec(
        class_name="helm.benchmark.augmentations.lowercase_perturbation.LowerCasePerturbation", args={}
    )


def misspelling(prob: float) -> PerturbationSpec:
    return PerturbationSpec(
        class_name="helm.benchmark.augmentations.misspelling_perturbation.MisspellingPerturbation",
        args={"prob": prob},
    )


def synonym(prob: float) -> PerturbationSpec:
    return PerturbationSpec(
        class_name="helm.benchmark.augmentations.synonym_perturbation.SynonymPerturbation",
        args={"prob": prob},
    )


def contrast_sets() -> PerturbationSpec:
    return PerturbationSpec(
        class_name="helm.benchmark.augmentations.contrast_sets_perturbation.ContrastSetsPerturbation",
        args={},
    )


def typo(prob: float) -> PerturbationSpec:
    return PerturbationSpec(
        class_name="helm.benchmark.augmentations.typos_perturbation.TyposPerturbation",
        args={"prob": prob},
    )


def filler(prob: float) -> PerturbationSpec:
    return PerturbationSpec(
        class_name="helm.benchmark.augmentations.filler_words_perturbation.FillerWordsPerturbation",
        args={"insert_prob": prob, "speaker_ph": False},
    )


def mild_mix() -> PerturbationSpec:
    return PerturbationSpec(
        class_name="helm.benchmark.augmentations.mild_mix_perturbation.MildMixPerturbation", args={}
    )


def contract_and_expand() -> List[PerturbationSpec]:
    return [
        PerturbationSpec(
            class_name=f"helm.benchmark.augmentations.contraction_expansion_perturbation.{mode}Perturbation",
            args={},
        )
        for mode in ["Contraction", "Expansion"]
    ]


def dialect(
    prob: float, source_class: str, target_class: str, mapping_file_path: Optional[str] = None
) -> PerturbationSpec:
    return PerturbationSpec(
        class_name="helm.benchmark.augmentations.dialect_perturbation.DialectPerturbation",
        args={
            "prob": prob,
            "source_class": source_class,
            "target_class": target_class,
            "mapping_file_path": mapping_file_path,
        },
    )


def person_name(
    prob: float,
    source_class: Dict[str, str],
    target_class: Dict[str, str],
    name_file_path: Optional[str] = None,
    person_name_type: str = "first_name",
    preserve_gender: bool = True,
) -> PerturbationSpec:
    return PerturbationSpec(
        class_name="helm.benchmark.augmentations.person_name_perturbation.PersonNamePerturbation",
        args={
            "prob": prob,
            "source_class": source_class,
            "target_class": target_class,
            "name_file_path": name_file_path,
            "person_name_type": person_name_type,
            "preserve_gender": preserve_gender,
        },
    )


def gender(
    mode: str,
    prob: float,
    source_class: str,
    target_class: str,
    mapping_file_path: Optional[str] = None,
    mapping_file_genders: Tuple[str] = None,
    bidirectional: bool = False,
) -> PerturbationSpec:
    return PerturbationSpec(
        class_name="helm.benchmark.augmentations.gender_perturbation.GenderPerturbation",
        args={
            "mode": mode,
            "prob": prob,
            "source_class": source_class,
            "target_class": target_class,
            "mapping_file_path": mapping_file_path,
            "mapping_file_genders": mapping_file_genders,
            "bidirectional": bidirectional,
        },
    )


# Specifies the data augmentations that we're interested in trying out.
# Concretely, this is a mapping from the name (which is specified in a conf
# file or the CLI) to a list of options to try, where each option is a list of perturbations.
# Each option generates a RunSpec.
# For example, suppose:
# - We specify data_augmentation=foo
# - foo maps to {r1: [a, b], r2: [c, d, e]}
# Then we will create two RunSpecs:
# - r1: with perturbations [a, b]
# - r2: with perturbations [c, d, e]

ROBUSTNESS_PERTURBATION_SPECS: List[PerturbationSpec] = [mild_mix()]

FAIRNESS_PERTURBATION_SPECS: List[PerturbationSpec] = [
    dialect(prob=1.0, source_class="SAE", target_class="AAVE"),
    gender(mode="pronouns", prob=1.0, source_class="male", target_class="female"),
    person_name(
        prob=1.0,
        source_class={"race": "white_american"},
        target_class={"race": "black_american"},
        person_name_type="first_name",
        preserve_gender=True,
    ),
]

PERTURBATION_SPECS_DICT: Dict[str, Dict[str, List[PerturbationSpec]]] = {
    # Robustness
    "extra_space": {"extra_space2": [extra_space(num_spaces=2)]},
    "contrast_sets": {"contrast_sets": [contrast_sets()]},
    "space": {"space3": [space(max_spaces=3)]},
    "lower": {"lower": [lower()]},
    "contract": {"contract": contract_and_expand()},
    "filler": {"filler0.3": [filler(0.3)]},
    "misspelling_mild": {"misspelling0.05": [misspelling(prob=0.05)]},
    "misspelling_medium": {"misspelling0.20": [misspelling(prob=0.20)]},
    "misspelling_hard": {"misspelling0.5": [misspelling(prob=0.5)]},
    "misspelling_sweep": {f"misspelling{prob}": [misspelling(prob=prob)] for prob in [0, 0.05, 0.2, 0.5]},
    "typo_easy": {"typo0.1": [typo(prob=0.1)]},
    "typo_medium": {"typo0.3": [typo(prob=0.3)]},
    "typo_hard": {"typo0.5": [typo(prob=0.5)]},
    "synonym": {"synonym0.5": [synonym(prob=0.5)]},
    # Fairness
    "dialect_easy": {
        "dialect_easy_prob=0.1_source=SAE_target=AAVE": [dialect(prob=0.1, source_class="SAE", target_class="AAVE")]
    },
    "dialect_medium": {
        "dialect_prob=0.3_source=SAE_target=AAVE": [dialect(prob=0.3, source_class="SAE", target_class="AAVE")]
    },
    "dialect_hard": {
        "dialect_prob=0.5_source=SAE_target=AAVE": [dialect(prob=0.5, source_class="SAE", target_class="AAVE")]
    },
    "dialect_deterministic": {
        "dialect_prob=1.0_source=SAE_target=AAVE": [dialect(prob=1.0, source_class="SAE", target_class="AAVE")]
    },
    "person_name_first_hard_preserve_gender": {
        "person_name_first_prob=0.5_source=white_target=black_preserve_gender=True": [
            person_name(
                prob=0.5,
                source_class={"race": "white_american"},
                target_class={"race": "black_american"},
                person_name_type="first_name",
                preserve_gender=True,
            )
        ],
    },
    "person_name_first_hard_dont_preserve_gender": {
        "person_name_first_prob=0.5_source=white_target=black_preserve_gender=False": [
            person_name(
                prob=0.5,
                source_class={"race": "white_american"},
                target_class={"race": "black_american"},
                person_name_type="first_name",
                preserve_gender=False,
            )
        ],
    },
    "person_name_first_deterministic": {
        "person_name_first_prob=1.0_source=white_target=black_preserve_gender=True": [
            person_name(
                prob=1.0,
                source_class={"race": "white_american"},
                target_class={"race": "black_american"},
                person_name_type="first_name",
                preserve_gender=True,
            )
        ],
    },
    "person_name_last_deterministic": {
        "person_name_last_prob=1.0_source=white_target=hispanic": [
            person_name(
                prob=1.0,
                source_class={"race": "white"},
                target_class={"race": "hispanic"},
                person_name_type="last_name",
                preserve_gender=False,
            )
        ],
    },
    "person_name_last_hard": {
        "person_name_last_prob=0.5_source=white_target=hispanic": [
            person_name(
                prob=0.5,
                source_class={"race": "white"},
                target_class={"race": "hispanic"},
                person_name_type="last_name",
                preserve_gender=False,
            )
        ],
    },
    "gender_terms_easy": {
        "gender_terms_prob=0.1_source=male_target=female": [
            gender(mode="terms", prob=0.1, source_class="male", target_class="female")
        ]
    },
    "gender_terms_medium": {
        "gender_terms_prob=0.3_source=male_target=female": [
            gender(mode="terms", prob=0.3, source_class="male", target_class="female")
        ]
    },
    "gender_terms_hard": {
        "gender_terms_prob=0.5_source=male_target=female": [
            gender(mode="terms", prob=0.5, source_class="male", target_class="female")
        ]
    },
    "gender_terms_deterministic": {
        "gender_terms_prob=1.0_source=male_target=female": [
            gender(mode="terms", prob=1.0, source_class="male", target_class="female")
        ]
    },
    "gender_pronouns_easy": {
        "gender_pronouns_prob=0.1_source=male_target=female": [
            gender(mode="pronouns", prob=0.1, source_class="male", target_class="female")
        ]
    },
    "gender_pronouns_medium": {
        "gender_pronouns_prob=0.3_source=male_target=female": [
            gender(mode="pronouns", prob=0.3, source_class="male", target_class="female")
        ]
    },
    "gender_pronouns_hard": {
        "gender_pronouns_prob=0.5_source=male_target=female": [
            gender(mode="pronouns", prob=0.5, source_class="male", target_class="female")
        ]
    },
    "gender_pronouns_deterministic": {
        "gender_pronouns_prob=1.0_source=male_target=female": [
            gender(mode="pronouns", prob=1.0, source_class="male", target_class="female")
        ]
    },
    "robustness": {"robustness": ROBUSTNESS_PERTURBATION_SPECS},
    "fairness": {"fairness": FAIRNESS_PERTURBATION_SPECS},
    "canonical": {"canonical": ROBUSTNESS_PERTURBATION_SPECS + FAIRNESS_PERTURBATION_SPECS},
    "robustness_all": {
        "robustness_all": [
            *contract_and_expand(),
            filler(0.1),
            lower(),
            misspelling(prob=0.1),
            space(max_spaces=3),
            synonym(prob=0.1),
            typo(prob=0.01),
        ]
    },
}


class DataAugmentationRunExpander(RunExpander):
    """
    Applies a list of data augmentations, where the list of data augmentations
    is given by a name (see the keys to `PERTURBATION_SPECS_DICT` above).

    **Example:**

        data_augmentation=all

    Note that some names map to a single data augmentation with multiple
    perturbations (e.g., all), and others map to a list of data augmentations
    each with one perturbation (e.g., misspelling_sweep).
    """

    name = "data_augmentation"

    def __init__(self, value):
        """
        Args:
            value (str): Comma-separated list of perturbations."""
        self.value = value

        if self.value not in PERTURBATION_SPECS_DICT:
            raise ValueError(
                f"Unknown data_augmentation: {self.value}; possible choices: {PERTURBATION_SPECS_DICT.keys()}"
            )

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        """Return `run_spec` with data augmentations."""

        def create_run_spec(aug_name: str, perturbation_specs: List[PerturbationSpec]) -> RunSpec:
            data_augmenter_spec: DataAugmenterSpec = DataAugmenterSpec(
                perturbation_specs=perturbation_specs,
                # Always include original and perturbed instances together so that
                # we can compute the normal and robustness metrics in the same run.
                should_augment_train_instances=False,
                should_include_original_train=True,  # irrelevant
                should_skip_unchanged_train=True,  # irrelevant
                should_augment_eval_instances=True,
                should_include_original_eval=True,
                should_skip_unchanged_eval=True,
                seeds_per_instance=1,
            )
            return replace(
                run_spec,
                name=f"{run_spec.name},{DataAugmentationRunExpander.name}={aug_name}",
                data_augmenter_spec=data_augmenter_spec,
            )

        return [
            create_run_spec(aug_name, perturbation_specs)
            for aug_name, perturbation_specs in PERTURBATION_SPECS_DICT[self.value].items()
        ]


class ScenarioSpecRunExpander(RunExpander):
    """
    Run expander which mutates ScenarioSpec arguments (e.g., can be used to set the prompt size or tokenizer
    in the SyntheticEfficiencyScenario).
    """

    def __init__(self, value):
        """
        `value` is either the actual value to use or a lookup into the values dict.
        """
        self.name = type(self).name
        if value in type(self).values_dict:
            self.values = type(self).values_dict[value]
        else:
            self.values = [value]

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        def sanitize(value):
            return str(value).replace("/", "_")

        # Change field in scenario_spec object as appropriate.
        return [
            replace(
                run_spec,
                name=f"{run_spec.name}{',' if ':' in run_spec.name else ':'}{self.name}={sanitize(value)}",
                scenario_spec=replace(
                    run_spec.scenario_spec, args=dict(run_spec.scenario_spec.args, **{self.name: value})
                ),
            )
            for value in self.values
        ]


class TokenizerRunExpander(ScenarioSpecRunExpander):
    """
    Run expander for specifying tokenizer for SyntheticEfficiencyScenario.
    """

    name = "tokenizer"
    # Compute model-to-tokenizer mapping.
    # TODO: Consider moving this to Model class.
    model_to_tokenizer_mapping = {
        "together/yalm": ["yandex/yalm"],
        "together/bloom": ["bigscience/bloom"],
        "together/t0pp": ["bigscience/t0pp"],
        "together/t5-11b": ["google/t5"],
        "together/ul2": ["google/ul2"],
        "together/glm": ["tsinghua/glm"],
        "AlephAlpha/luminous-base": ["AlephAlpha/luminous-base"],
        "AlephAlpha/luminous-extended": ["AlephAlpha/luminous-extended"],
        "AlephAlpha/luminous-supreme": ["AlephAlpha/luminous-supreme"],
        "AlephAlpha/luminous-world": ["AlephAlpha/luminous-world"],
        "huggingface/santacoder": ["bigcode/santacoder"],
        "huggingface/starcoder": ["bigcode/starcoder"],
    }
    model_tags_and_tokenizers = [
        (GPT2_TOKENIZER_TAG, "huggingface/gpt2"),
        (AI21_TOKENIZER_TAG, "ai21/j1"),
        (COHERE_TOKENIZER_TAG, "cohere/cohere"),
        (OPT_TOKENIZER_TAG, "meta/opt"),
        (GPTJ_TOKENIZER_TAG, "eleutherai/gptj"),
        (GPT4_TOKENIZER_TAG, "openai/cl100k_base"),
        (GPTNEO_TOKENIZER_TAG, "eleutherai/gptneox"),
    ]
    for model_tag, tokenizer in model_tags_and_tokenizers:
        for model in get_model_names_with_tag(model_tag):
            model_to_tokenizer_mapping[model] = [tokenizer]
    # tokenizer=default will map to using the right tokenizer for a given model.
    values_dict = {"default": model_to_tokenizer_mapping}

    def __init__(self, value):
        """
        Args:
            value (str): Either the actual value to use or a lookup into the values dict.
        """
        self.name = type(self).name
        if value in type(self).values_dict:
            self.all_values = type(self).values_dict[value]
        else:
            self.all_values = [value]

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        # Find right tokenizer given model.
        if isinstance(self.all_values, dict):
            model: str = run_spec.adapter_spec.model
            self.values = self.all_values[model] if model in self.all_values else []
        else:
            self.values = self.all_values
        return super().expand(run_spec)


class NumPromptTokensRunExpander(ScenarioSpecRunExpander):
    """
    Run expander for specifying number of prompt tokens. This is used in the SyntheticEfficiencyScenario
    to control the size of the prompt (in terms of number of tokens).
    """

    name = "num_prompt_tokens"
    values_dict = {"default_sweep": [1, 256, 512, 1024, 1536]}


class NumOutputTokensRunExpander(RunExpander):
    """
    Run expander for specifying number of output tokens. This is used in the SyntheticEfficiencyScenario
    to control the number of output tokens.
    """

    name = "num_output_tokens"
    adapter_spec_name = "max_tokens"
    values_dict = {"default_sweep": [1, 2, 4, 8, 16, 32, 64]}

    def __init__(self, value):
        """
        Args:
            value (str): Either the actual value to use or a lookup into the values dict.
        """
        self.name = type(self).name
        self.adapter_spec_name = type(self).adapter_spec_name
        if value in type(self).values_dict:
            self.values = type(self).values_dict[value]
        else:
            self.values = [value]

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        # Change run_spec name (num_output_tokens field), and adapter_spec (max_tokens field).
        return [
            replace(
                run_spec,
                name=f"{run_spec.name}{',' if ':' in run_spec.name else ':'}{self.name}={value}",
                adapter_spec=replace(run_spec.adapter_spec, **{self.adapter_spec_name: value}),
            )
            for value in self.values
        ]


class IncreaseMaxTokensRunExpander(RunExpander):
    """
    Run expander for increasing the number of max tokens.
    """

    name = "increase_max_tokens"

    def __init__(self, value: int):
        """
        Args:
            value (int): The number of tokens to increase max tokens by
        """
        self.value = value

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        adapter_spec: AdapterSpec = run_spec.adapter_spec
        adapter_spec = replace(adapter_spec, max_tokens=adapter_spec.max_tokens + self.value)
        return [
            replace(
                run_spec,
                adapter_spec=adapter_spec,
            ),
        ]


class IncreaseTemperatureRunExpander(RunExpander):
    """
    Run expander for increasing the temperature.
    """

    name = "increase_temperature"

    def __init__(self, value: float):
        """
        Args:
            value (float): The amount to increase temperature by
        """
        self.value = value

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        adapter_spec: AdapterSpec = run_spec.adapter_spec
        adapter_spec = replace(adapter_spec, temperature=adapter_spec.temperature + self.value)
        return [
            replace(
                run_spec,
                adapter_spec=adapter_spec,
            ),
        ]


class ChatMLRunExpander(RunExpander):
    """
    Adapt to ChatML: https://github.com/openai/openai-python/blob/main/chatml.md
    A 1-shot example:
    <|im_start|>system
    Translate from English to French
    <|im_end|>
    <|im_start|>user
    How are you?
    <|im_end|>
    <|im_start|>user
    Comment allez-vous?
    <|im_end|>
    <|im_start|>user
    {{user input here}}<|im_end|>
    """

    name = "chatml"

    def __init__(self):
        self.name = type(self).name

    def expand(self, run_spec: RunSpec) -> List[RunSpec]:
        adapter_spec = run_spec.adapter_spec
        # according to https://github.com/openai/openai-python/blob/main/chatml.md#few-shot-prompting
        # few-shot examples should do `<|im_start|>system name=example_user`
        # or `<|im_start|>system name=example_assistant`
        # but it is also possible to put examples into a user message.

        scenario_name = run_spec.name.split(":")[0]

        if scenario_name in ("msmarco",):
            # output_prefix:
            #     Does the passage answer the query?
            #     Answer:
            #
            # new_output_prefix:
            #     Does the passage answer the query?<|im_end|>
            #     <|im_start|>assistant
            #     Answer:

            new_output_prefix = (
                adapter_spec.output_prefix.split("\n")[0]
                + "<|im_end|>\n<|im_start|>assistant\n"
                + adapter_spec.output_prefix.split("\n")[1]
            )

        elif scenario_name in ("summarization_cnndm", "summarization_xsum"):
            # output_prefix:
            #     Summarize the above article in 1 sentence.
            #
            # new_output_prefix:
            #     Summarize the above article in 1 sentence.<|im_end|>
            #     <|im_start|>assistant
            #

            new_output_prefix = adapter_spec.output_prefix + "<|im_end|>\n<|im_start|>assistant\n"

        else:
            # output_prefix:
            #     {output_prefix}
            #
            # new_output_prefix:
            #     <|im_end|>
            #     <|im_start|>assistant
            #     {output_prefix}

            new_output_prefix = "<|im_end|>\n<|im_start|>assistant\n" + adapter_spec.output_prefix

        adapter_spec = replace(
            adapter_spec,
            # This is a hack to make sure <|im_start|>user goes before the reference.
            instructions=(
                f"<|im_start|>system\n{adapter_spec.instructions}<|im_end|>\n<|im_start|>user\n"
                if adapter_spec.instructions != ""
                else "<|im_start|>user\n"
            ),
            instance_prefix="",
            output_prefix=new_output_prefix,
            output_suffix="<|im_end|>\n<|im_start|>user\n",
            stop_sequences=adapter_spec.stop_sequences + ["<|im_end|>"],
        )

        return [
            replace(
                run_spec,
                adapter_spec=adapter_spec,
            ),
        ]


RUN_EXPANDER_SUBCLASSES: List[Type[RunExpander]] = [
    InstructionsRunExpander,
    PromptRunExpander,
    NewlineRunExpander,
    StopRunExpander,
    FormatPromptRunExpander,
    AddToStopRunExpander,
    GlobalPrefixRunExpander,
    NumTrainTrialsRunExpander,
    MaxTrainInstancesRunExpander,
    NumOutputsRunExpander,
    ModelRunExpander,
    DataAugmentationRunExpander,
    TokenizerRunExpander,
    NumPromptTokensRunExpander,
    NumOutputTokensRunExpander,
    ChatMLRunExpander,
]


RUN_EXPANDERS = dict((expander.name, expander) for expander in RUN_EXPANDER_SUBCLASSES)
