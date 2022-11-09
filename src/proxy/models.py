from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Different modalities
TEXT_MODEL_TAG: str = "text"
CODE_MODEL_TAG: str = "code"
EMBEDDING_MODEL_TAG: str = "embedding"

# Some model APIs have limited functionalities
FULL_FUNCTIONALITY_TEXT_MODEL_TAG: str = "full_functionality_text"
LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG: str = "limited_functionality_text"

# For OpenAI models with wider context windows
WIDER_CONTEXT_WINDOW_TAG: str = "wider_context_window"

# To fetch models that use these tokenizers
GPT2_TOKENIZER_TAG: str = "gpt2_tokenizer"
AI21_TOKENIZER_TAG: str = "ai21_tokenizer"
COHERE_TOKENIZER_TAG: str = "cohere_tokenizer"
OPT_TOKENIZER_TAG: str = "opt_tokenizer"
GPTJ_TOKENIZER_TAG: str = "gptj_tokenizer"
GPTNEO_TOKENIZER_TAG: str = "gptneo_tokenizer"

# Models that are used for ablations and fine-grained analyses.
# These models are selected specifically because of their low marginal cost to evaluate.
ABLATION_MODEL_TAG: str = "ablation"

# Some models (e.g., T5) have stripped newlines.
# So we cannot use \n as a stop sequence for these models.
NO_NEWLINES_TAG: str = "no_newlines"

# Some models (e.g., UL2) require a prefix (e.g., [NLG]) in the
# prompts to indicate the mode before doing inference.
NLG_PREFIX_TAG: str = "nlg_prefix_tag"


@dataclass
class Model:
    """Represents a model that we can make requests to."""

    # Model group, used to filter for quotas (e.g. "huggingface")
    group: str

    # Name of the specific model (e.g. "huggingface/gpt-j-6b")
    name: str

    # Display name of the specific model (e.g. "GPT-J-6B")
    display_name: str

    # Organization that originally created the model (e.g. "EleutherAI")
    #   Note that this may be different from group or the prefix of the name
    #   ("huggingface" in "huggingface/gpt-j-6b") as the hosting organization
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
# Note that schema.yaml has much of this information now.
# Over time, we should add more information there.
ALL_MODELS = [
    # AI21: https://studio.ai21.com/pricing
    Model(
        group="jurassic",
        creator_organization="AI21 Labs",
        name="ai21/j1-jumbo",
        display_name="Jurassic Jumbo (178B)",
        description="Jurassic J1-Jumbo (178B parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    ),
    # From AI21: "the new model is a mid-point in terms of size, cost and performance between Jumbo and Large.
    # We also implemented a few tweaks to its training process. Internal benchmarks suggest it can really
    # help the unit economics on your end compared to Jumbo, without compromising too much on quality."
    Model(
        group="jurassic",
        creator_organization="AI21 Labs",
        name="ai21/j1-grande",
        display_name="Jurassic Grande (17B)",
        description="Jurassic J1-Large (17B parameters with a few tweaks to its training process)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    ),
    Model(
        group="jurassic",
        creator_organization="AI21 Labs",
        name="ai21/j1-large",
        display_name="Jurassic Large (7.5B)",
        description="Jurassic J1-Large (7.5B parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    ),
    # Anthropic
    Model(
        group="anthropic",
        creator_organization="Anthropic",
        name="anthropic/stanford-online-all-v4-s3",
        display_name="Anthropic-LM (52B)",
        description="Anthropic model (52B parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG, ABLATION_MODEL_TAG],
    ),
    # BigScience
    Model(
        group="together",
        creator_organization="BigScience",
        name="together/bloom",
        display_name="BLOOM (176B)",
        # From https://bigscience.huggingface.co/blog/bloom
        description="BLOOM (176B parameters) is an autoregressive model similar to GPT-3 trained "
        "on 46 natural languages and 13 programming languages.",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG],
    ),
    Model(
        group="together",
        creator_organization="BigScience",
        name="together/t0pp",
        display_name="T0++ (11B)",
        # From https://huggingface.co/bigscience/T0pp
        description="T0pp (11B parameters) is an encoder-decoder model trained on a large set of different tasks "
        "specified in natural language prompts.",
        # Does not support echo=True
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG, NO_NEWLINES_TAG],
    ),
    # Cohere models
    # Model versioning and the possible versions are not documented here:
    # https://docs.cohere.ai/generate-reference#model-optional.
    # So, instead, we got the names of the models from the Cohere Playground.
    #
    # Note that their tokenizer and model were trained on English text and
    # they do not have a dedicated decode API endpoint, so the adaptation
    # step for language modeling fails for certain Scenarios:
    # the_pile:subset=ArXiv
    # the_pile:subset=Github
    # the_pile:subset=PubMed Central
    Model(
        group="cohere",
        creator_organization="Cohere",
        name="cohere/xlarge-20220609",
        display_name="Cohere xlarge (52.4B)",
        description="Cohere xlarge (52.4B parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    Model(
        group="cohere",
        creator_organization="Cohere",
        name="cohere/xlarge-20221108",
        display_name="Cohere xlarge - 11/08/2022 update (52.4B)",
        description="Cohere xlarge - 11/08/2022 update (52.4B parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    Model(
        group="cohere",
        creator_organization="Cohere",
        name="cohere/large-20220720",
        display_name="Cohere large (13.1B)",
        description="Cohere large (13.1B parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    Model(
        group="cohere",
        creator_organization="Cohere",
        name="cohere/medium-20220720",
        display_name="Cohere medium (6.1B)",
        description="Cohere medium (6.1B parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    Model(
        group="cohere",
        creator_organization="Cohere",
        name="cohere/medium-20221108",
        display_name="Cohere medium - 11/08/2022 update (6.1B)",
        description="Cohere medium - 11/08/2022 update (6.1B parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    Model(
        group="cohere",
        creator_organization="Cohere",
        name="cohere/small-20220720",
        display_name="Cohere small (410M)",
        description="Cohere small (410M parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    # EleutherAI
    Model(
        group="together",
        creator_organization="EleutherAI",
        name="together/gpt-j-6b",
        display_name="GPT-J (6B)",
        description="GPT-J (6B parameters) autoregressive language model trained on The Pile",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG, GPTJ_TOKENIZER_TAG],
    ),
    Model(
        group="together",
        creator_organization="EleutherAI",
        name="together/gpt-neox-20b",
        display_name="GPT-NeoX (20B)",
        description="GPT-NeoX (20B parameters) autoregressive language model trained on The Pile.",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG, GPTNEO_TOKENIZER_TAG],
    ),
    # GooseAI supported models
    Model(
        group="gooseai",
        creator_organization="EleutherAI",
        name="gooseai/gpt-neo-20b",
        display_name="GPT-NeoX (20B, GooseAI)",
        description="GPT-NeoX (20B parameters) autoregressive language model trained on The Pile.",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPTNEO_TOKENIZER_TAG],
    ),
    Model(
        group="gooseai",
        creator_organization="EleutherAI",
        name="gooseai/gpt-j-6b",
        display_name="GPT-J (6B, GooseAI)",
        description="GPT-J (6B parameters) autoregressive language model trained on The Pile.",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPTJ_TOKENIZER_TAG],
    ),
    # HuggingFace
    Model(
        group="huggingface",
        creator_organization="EleutherAI",
        name="huggingface/gpt-j-6b",
        display_name="GPT-J (6B, HuggingFace)",
        description="GPT-J (6B parameters) autoregressive language model trained on The Pile.",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPTJ_TOKENIZER_TAG],
    ),
    # Google
    Model(
        group="together",
        creator_organization="Google",
        name="together/t5-11b",
        display_name="T5 (11B)",
        # From https://huggingface.co/docs/transformers/v4.20.1/en/model_doc/t5
        description="T5 (11B parameters) is an encoder-decoder model pre-trained on a multi-task mixture of "
        "unsupervised and supervised tasks and for which each task is converted into a text-to-text "
        "format.",
        # Does not support echo=True
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG, NO_NEWLINES_TAG],
    ),
    Model(
        group="together",
        creator_organization="Google",
        name="together/ul2",
        display_name="UL2 (20B)",
        # From https://huggingface.co/google/ul2
        description="UL2 (20B parameters) is an encoder-decoder model trained on the C4 corpus. It's similar to T5"
        "but trained with a different objective and slightly different scaling knobs.",
        # Does not support echo=True
        tags=[
            TEXT_MODEL_TAG,
            LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
            ABLATION_MODEL_TAG,
            NO_NEWLINES_TAG,
            NLG_PREFIX_TAG,
        ],
    ),
    Model(
        group="huggingface",
        creator_organization="OpenAI",
        name="huggingface/gpt2",
        display_name="GPT-2 (1.5B)",
        description="GPT-2 (1.5B parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    # OPT
    Model(
        group="together",
        creator_organization="Meta",
        name="together/opt-175b",
        display_name="OPT (175B)",
        # From https://arxiv.org/pdf/2205.01068.pdf
        description="Open Pre-trained Transformers (175B parameters) is a suite of decoder-only pre-trained "
        "transformers that are fully and responsibly shared with interested researchers.",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG, OPT_TOKENIZER_TAG],
    ),
    Model(
        group="together",
        creator_organization="Meta",
        name="together/opt-66b",
        display_name="OPT (66B)",
        # From https://arxiv.org/pdf/2205.01068.pdf
        description="Open Pre-trained Transformers (66B parameters) is a suite of decoder-only pre-trained "
        "transformers that are fully and responsibly shared with interested researchers.",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG, OPT_TOKENIZER_TAG],
    ),
    # Microsoft/NVIDIA
    Model(
        group="microsoft",
        creator_organization="Microsoft/NVIDIA",
        name="microsoft/TNLGv2_530B",
        display_name="TNLGv2 (530B)",
        description="TNLGv2 (530B parameters) autoregressive language model. "
        "It is trained on a filtered subset of the Pile and CommonCrawl.",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="microsoft",
        creator_organization="Microsoft/NVIDIA",
        name="microsoft/TNLGv2_7B",
        display_name="TNLGv2 (7B)",
        description="TNLGv2 (7B parameters) autoregressive language model. "
        "It is trained on a filtered subset of the Pile and CommonCrawl.",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    # OpenAI: https://beta.openai.com/docs/engines/gpt-3
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/davinci",
        display_name="GPT-3 (175B)",
        description="GPT-3 (175B parameters) autoregressive language model.",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/curie",
        display_name="GPT-3 (6.7B)",
        description="GPT-3 (6.7B parameters) autoregressive language model.",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/babbage",
        display_name="GPT-3 (1.3B)",
        description="GPT-3 (1.3B parameters) autoregressive language model.",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/ada",
        display_name="GPT-3 (350M)",
        description="GPT-3 (350M parameters) autoregressive language model.",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    # TODO: text-davinci-002 supports insertion. Support insertion in our framework.
    #       https://github.com/stanford-crfm/benchmarking/issues/359
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-davinci-002",
        display_name="Davinci Instruct v2",
        description="Davinci from Instruct series, 2nd generation - 4000 max tokens",
        tags=[TEXT_MODEL_TAG, WIDER_CONTEXT_WINDOW_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-davinci-001",
        display_name="Davinci Instruct v1",
        description="Davinci from Instruct series, 1st generation",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-curie-001",
        display_name="Curie Instruct",
        description="Curie from Instruct series",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-babbage-001",
        display_name="Babbage Instruct",
        description="Babbage from Instruct series",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-ada-001",
        display_name="Ada Instruct",
        description="Ada from Instruct series",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="codex",
        creator_organization="OpenAI",
        name="openai/code-davinci-002",
        display_name="Davinci Codex (4000 max tokens)",
        description="Davinci Codex 2nd Generation (for natural language to code) - 4000 max tokens",
        tags=[CODE_MODEL_TAG, WIDER_CONTEXT_WINDOW_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="codex",
        creator_organization="OpenAI",
        name="openai/code-davinci-001",
        display_name="Davinci Codex (2048 max tokens)",
        description="Davinci Codex (for natural language to code) - 2048 max tokens",
        tags=[CODE_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="codex",
        creator_organization="OpenAI",
        name="openai/code-cushman-001",
        display_name="Cushman Codex (2048 max tokens)",
        description="Cushman Codex (for natural language to code) - 2048 max tokens",
        tags=[CODE_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    # OpenAI similarity embedding models: https://beta.openai.com/docs/guides/embeddings
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-similarity-davinci-001",
        display_name="GPT-3 (12288-dimension embeddings)",
        description="GPT-3 (12288-dimension embeddings)",
        tags=[EMBEDDING_MODEL_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-similarity-curie-001",
        display_name="GPT-3 (4096-dimension embeddings)",
        description="GPT-3 (4096-dimension embeddings)",
        tags=[EMBEDDING_MODEL_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-similarity-babbage-001",
        display_name="GPT-3 (2048-dimension embeddings)",
        description="GPT-3 (2048-dimension embeddings)",
        tags=[EMBEDDING_MODEL_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-similarity-ada-001",
        display_name="GPT-3 (1024-dimension embeddings)",
        description="GPT-3 (1024-dimension embeddings)",
        tags=[EMBEDDING_MODEL_TAG],
    ),
    # Offline models
    Model(
        group="together",
        creator_organization="Tsinghua KEG",
        name="together/glm",
        display_name="GLM (130B)",
        # From https://github.com/THUDM/GLM-130B
        description="GLM-130B is an open bilingual (English & Chinese) bidirectional dense model with 130 billion "
        "parameters, pre-trained using the algorithm of General Language Model (GLM).",
        # Inference with echo=True is not feasible -- in the prompt encoding phase, they use
        # bidirectional attention and do not perform predictions on them.
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG, NO_NEWLINES_TAG],
    ),
    Model(
        group="together",
        creator_organization="Yandex",
        name="together/yalm",
        display_name="YaLM (100B)",
        # From https://github.com/yandex/YaLM-100B
        description="YaLM (100B parameters) is an autoregressive language model trained on English and Russian text.",
        # TODO: change to `FULL_FUNCTIONALITY_TEXT_MODEL_TAG` when we fix the infinite loop in L.M. adaptation
        #       https://github.com/stanford-crfm/benchmarking/issues/738
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG],
    ),
    # For debugging
    Model(
        group="simple",
        creator_organization="Simple",
        name="simple/model1",
        display_name="Simple Model 1",
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


def get_models_by_organization(organization: str) -> List[str]:
    """
    Gets models by organization e.g., ai21 => ai21/j1-jumbo, ai21/j1-grande, ai21-large.
    """
    return [model.name for model in ALL_MODELS if model.organization == organization]


def get_model_names_with_tag(tag: str) -> List[str]:
    """Get all the name of the models with tag `tag`."""
    return [model.name for model in ALL_MODELS if tag in model.tags]


def get_all_text_models() -> List[str]:
    """Get all text model names."""
    return get_model_names_with_tag(TEXT_MODEL_TAG)


def get_all_code_models() -> List[str]:
    """Get all code model names."""
    return get_model_names_with_tag(CODE_MODEL_TAG)
