from dataclasses import dataclass, field
from typing import List, Dict, Optional

TEXT_MODEL_TAG: str = "text"
CODE_MODEL_TAG: str = "code"
LIMITED_FUNCTIONALITY_MODEL_TAG: str = "limited_functionality"
WIDER_CONTEXT_WINDOW_TAG: str = "wider_context_window"


@dataclass
class Model:
    """Represents a model that we can make requests to."""

    # Model group, used to filter for quotas (e.g. "huggingface")
    group: str

    # Name of the specific model (e.g. "huggingface/gptj_6b")
    name: str

    # Organization that originally created the model (e.g. "EleutherAI")
    #   Note that this may be different from group or the prefix of the name
    #   ("huggingface" in "huggingface/gptj_6b") as the hosting organization
    #   may be different from the creator organization. We also capitalize
    #   this field properly to later display in the UI.
    creator_organization: str

    # Short description of the model
    description: str

    # Tags corresponding to the properties of the model
    tags: List[str] = field(default_factory=list)

    # Estimated training co2e cost of this model
    training_co2e_cost: Optional[float] = None

    @property
    def organization(self) -> str:
        """
        Extracts the organization from the model name.
        Example: 'ai21/j1-jumbo' => 'ai21'
        """
        return self.name.split("/")[0]

    @property
    def engine(self) -> str:
        """
        Extracts the model engine from the model name.
        Example: 'ai21/j1-jumbo' => 'j1-jumbo'
        """
        return self.name.split("/")[1]


# For the list of available models, see the following docs:
#
# OpenAI: https://help.openai.com/en/articles/5832130-what-s-changed-with-engine-names-and-best-practices
# AI21: https://studio.ai21.com/docs/jurassic1-language-models/
ALL_MODELS = [
    # AI21
    Model(
        group="jurassic",
        creator_organization="AI21 Labs",
        name="ai21/j1-jumbo",
        description="Jurassic J1-Jumbo (178B parameters)",
        tags=[TEXT_MODEL_TAG],
    ),
    # From AI21: "the new model is a mid-point in terms of size, cost and performance between Jumbo and Large.
    # We also implemented a few tweaks to its training process. Internal benchmarks suggest it can really
    # help the unit economics on your end compared to Jumbo, without compromising too much on quality."
    Model(
        group="jurassic",
        creator_organization="AI21 Labs",
        name="ai21/j1-grande",
        description="Jurassic J1-Large (17B parameters with a few tweaks to its training process)",
        tags=[TEXT_MODEL_TAG],
    ),
    Model(
        group="jurassic",
        creator_organization="AI21 Labs",
        name="ai21/j1-large",
        description="Jurassic J1-Large (7.5B parameters)",
        tags=[TEXT_MODEL_TAG],
    ),
    # OpenAI: https://beta.openai.com/docs/engines/gpt-3
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/davinci",
        description="GPT-3 (175B parameters)",
        tags=[TEXT_MODEL_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/curie",
        description="GPT-3 (6.7B parameters)",
        tags=[TEXT_MODEL_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/babbage",
        description="GPT-3 (1.3B parameters)",
        tags=[TEXT_MODEL_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/ada",
        description="GPT-3 (350M parameters)",
        tags=[TEXT_MODEL_TAG],
    ),
    # TODO: text-davinci-002 supports insertion. Support insertion in our framework.
    #       https://github.com/stanford-crfm/benchmarking/issues/359
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-davinci-002",
        description="GPT-3 from Instruct series 2nd generation (175B parameters) - 4000 max tokens",
        tags=[TEXT_MODEL_TAG, WIDER_CONTEXT_WINDOW_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-davinci-001",
        description="GPT-3 from Instruct series (175B parameters)",
        tags=[TEXT_MODEL_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-curie-001",
        description="GPT-3 from Instruct series (6.7B parameters)",
        tags=[TEXT_MODEL_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-babbage-001",
        description="GPT-3 from Instruct series (1.3B parameters)",
        tags=[TEXT_MODEL_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-ada-001",
        description="GPT-3 from Instruct series (350M parameters)",
        tags=[TEXT_MODEL_TAG],
    ),
    Model(
        group="codex",
        creator_organization="OpenAI",
        name="openai/code-davinci-002",
        description="Codex 2nd Generation (for natural language to code) - 4000 max tokens",
        tags=[CODE_MODEL_TAG, WIDER_CONTEXT_WINDOW_TAG],
    ),
    Model(
        group="codex",
        creator_organization="OpenAI",
        name="openai/code-davinci-001",
        description="Codex (for natural language to code) - 2048 max tokens",
        tags=[CODE_MODEL_TAG],
    ),
    Model(
        group="codex",
        creator_organization="OpenAI",
        name="openai/code-cushman-001",
        description="Codex (for natural language to code) - 2048 max tokens",
        tags=[CODE_MODEL_TAG],
    ),
    # GooseAI supported models
    Model(
        group="gooseai",
        creator_organization="EleutherAI",
        name="gooseai/gpt-neo-20b",
        description="GPT-NeoX (20B parameters trained by EleutherAI on The Pile)",
        tags=[TEXT_MODEL_TAG],
    ),
    Model(
        group="gooseai",
        creator_organization="EleutherAI",
        name="gooseai/gpt-j-6b",
        description="GPT-J (6B parameters trained by EleutherAI on The Pile)",
        tags=[TEXT_MODEL_TAG],
    ),
    # HuggingFace
    # TODO: Switch GPT-J over to GooseAPI
    #       https://github.com/stanford-crfm/benchmarking/issues/310
    Model(
        group="huggingface",
        creator_organization="EleutherAI",
        name="huggingface/gptj_6b",
        description="GPT-J (6B parameters)",
        tags=[TEXT_MODEL_TAG],
    ),
    Model(
        group="huggingface",
        creator_organization="OpenAI",
        name="huggingface/gpt2",
        description="GPT-2 (1.5B parameters)",
        tags=[TEXT_MODEL_TAG],
    ),
    # Anthropic
    # TODO: The API for the Anthropic LM is not production-ready. Update with the official name and description.
    Model(
        group="anthropic",
        creator_organization="Anthropic",
        name="anthropic/stanford-online-helpful-v4-s3",
        description="Anthropic model (52B parameters)",
        tags=[LIMITED_FUNCTIONALITY_MODEL_TAG, TEXT_MODEL_TAG],  # The Anthropic model has limited functionality
    ),
    # Microsoft
    Model(
        group="microsoft",
        creator_organization="Microsoft",
        name="microsoft/TNLGv2_530B",
        description="Megatron-Turing NLG (530B parameters)",
        tags=[LIMITED_FUNCTIONALITY_MODEL_TAG, TEXT_MODEL_TAG],  # The TNLGv2 models have limited functionality
    ),
    # TODO: The TNLGv2_7B model is unavailable to us at the moment, but simply uncomment the following when it's ready.
    # Model(
    #     group="microsoft",
    #     creator_organization="Microsoft",
    #     name="microsoft/TNLGv2_7B",
    #     description="Megatron-Turing NLG (7B parameters)",
    #     tags = [LIMITED_FUNCTIONALITY_MODEL_TAG, TEXT_MODEL_TAG],  # The TNLGv2 models have limited functionality
    # ),
    # For debugging
    Model(
        group="simple",
        creator_organization="Simple",
        name="simple/model1",
        description="Copy last tokens (for debugging)",
    ),
]


MODEL_NAME_TO_MODEL: Dict[str, Model] = {model.name: model for model in ALL_MODELS}


def get_model(model_name: str) -> Model:
    """Get the `Model` given the name."""
    if model_name not in MODEL_NAME_TO_MODEL:
        raise ValueError(f"No model with name: {model_name}")

    return MODEL_NAME_TO_MODEL[model_name]


def get_model_group(model_name: str) -> str:
    """Get the model's group given the name."""
    model: Model = get_model(model_name)
    return model.group


def get_all_models() -> List[str]:
    """Get all model names."""
    return list(MODEL_NAME_TO_MODEL.keys())


def get_model_names_with_tag(tag: str) -> List[str]:
    """Get all the name of the models with tag `tag`."""
    return [model.name for model in ALL_MODELS if tag in model.tags]


def get_all_text_models() -> List[str]:
    """Get all text model names."""
    return get_model_names_with_tag(TEXT_MODEL_TAG)


def get_all_code_models() -> List[str]:
    """Get all code model names."""
    return get_model_names_with_tag(CODE_MODEL_TAG)
