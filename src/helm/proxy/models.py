from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Different modalities
TEXT_MODEL_TAG: str = "text"
IMAGE_MODEL_TAG: str = "image"
CODE_MODEL_TAG: str = "code"
EMBEDDING_MODEL_TAG: str = "embedding"

# Some model APIs have limited functionalities
FULL_FUNCTIONALITY_TEXT_MODEL_TAG: str = "full_functionality_text"
LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG: str = "limited_functionality_text"

# ChatML format
CHATML_MODEL_TAG: str = "chatml"

# OpenAI Chat format
OPENAI_CHATGPT_MODEL_TAG: str = "openai_chatgpt"

# For Anthropic models
ANTHROPIC_MODEL_TAG: str = "anthropic"

# For OpenAI models with wider context windows
# TODO(#1455): Simplify context window tags.
WIDER_CONTEXT_WINDOW_TAG: str = "openai_wider_context_window"  # 4000 tokens
GPT4_CONTEXT_WINDOW_TAG: str = "gpt4_context_window"  # 8192 tokens
GPT4_32K_CONTEXT_WINDOW_TAG: str = "gpt4_32k_context_window"  # 32768 tokens

# For AI21 Jurassic-2 models with wider context windows
AI21_WIDER_CONTEXT_WINDOW_TAG: str = "ai21_wider_context_window"

# For AI21 Jurassic-2 Jumbo
# AI21 has recommended using a sequence length of 6000 tokens to avoid OOMs.
AI21_JURASSIC_2_JUMBO_CONTEXT_WINDOW_TAG: str = "ai21_jurassic_2_jumbo_context_window"  # 6000

# To fetch models that use these tokenizers
GPT2_TOKENIZER_TAG: str = "gpt2_tokenizer"
AI21_TOKENIZER_TAG: str = "ai21_tokenizer"
COHERE_TOKENIZER_TAG: str = "cohere_tokenizer"
OPT_TOKENIZER_TAG: str = "opt_tokenizer"
GPTJ_TOKENIZER_TAG: str = "gptj_tokenizer"
GPT4_TOKENIZER_TAG: str = "gpt4_tokenizer"
GPTNEO_TOKENIZER_TAG: str = "gptneo_tokenizer"

# Models which emit garbage tokens when temperature=0.
BUGGY_TEMP_0_TAG: str = "buggy_temp_0"

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
        display_name="Jurassic-1 Jumbo (178B)",
        description="Jurassic-1 Jumbo (178B parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    ),
    # From AI21: "the new model is a mid-point in terms of size, cost and performance between Jumbo and Large.
    # We also implemented a few tweaks to its training process. Internal benchmarks suggest it can really
    # help the unit economics on your end compared to Jumbo, without compromising too much on quality."
    Model(
        group="jurassic",
        creator_organization="AI21 Labs",
        name="ai21/j1-grande",
        display_name="Jurassic-1 Grande (17B)",
        description="Jurassic-1 Grande (17B parameters) with a few tweaks to the training process.",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    ),
    Model(
        group="jurassic",
        creator_organization="AI21 Labs",
        name="ai21/j1-grande-v2-beta",
        display_name="Jurassic-1 Grande v2 beta (17B)",
        description="Jurassic-1 Grande v2 beta (17B parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    ),
    Model(
        group="jurassic",
        creator_organization="AI21 Labs",
        name="ai21/j1-large",
        display_name="Jurassic-1 Large (7.5B)",
        description="Jurassic-1 Large (7.5B parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    ),
    # AI21 Jurassic-2 Models: https://www.ai21.com/blog/introducing-j2
    Model(
        group="jurassic",
        creator_organization="AI21 Labs",
        name="ai21/j2-jumbo",
        display_name="Jurassic-2 Jumbo (178B)",
        description="Jurassic-2 Jumbo (178B parameters)",
        tags=[
            TEXT_MODEL_TAG,
            AI21_JURASSIC_2_JUMBO_CONTEXT_WINDOW_TAG,
            FULL_FUNCTIONALITY_TEXT_MODEL_TAG,
            AI21_TOKENIZER_TAG,
        ],
    ),
    Model(
        group="jurassic",
        creator_organization="AI21 Labs",
        name="ai21/j2-grande",
        display_name="Jurassic-2 Grande (17B)",
        description="Jurassic-2 Grande (17B parameters) with a few tweaks to the training process.",
        tags=[TEXT_MODEL_TAG, AI21_WIDER_CONTEXT_WINDOW_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    ),
    Model(
        group="jurassic",
        creator_organization="AI21 Labs",
        name="ai21/j2-large",
        display_name="Jurassic-2 Large (7.5B)",
        description="Jurassic-2 Large (7.5B parameters)",
        tags=[TEXT_MODEL_TAG, AI21_WIDER_CONTEXT_WINDOW_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    ),
    # Aleph Alpha's Luminous models: https://docs.aleph-alpha.com/docs/introduction/luminous
    Model(
        group="luminous",
        creator_organization="Aleph Alpha",
        name="AlephAlpha/luminous-base",
        display_name="Luminous Base (13B)",
        description="Luminous Base (13B parameters)",
        # Does not support echo
        tags=[TEXT_MODEL_TAG, IMAGE_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="luminous",
        creator_organization="Aleph Alpha",
        name="AlephAlpha/luminous-extended",
        display_name="Luminous Extended (30B)",
        description="Luminous Extended (30B parameters)",
        # Does not support echo
        tags=[TEXT_MODEL_TAG, IMAGE_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="luminous",
        creator_organization="Aleph Alpha",
        name="AlephAlpha/luminous-supreme",
        display_name="Luminous Supreme (70B)",
        description="Luminous Supreme (70B parameters)",
        # Does not support echo.
        # TODO: images will be supported in the near future. Add IMAGE_MODEL_TAG.
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    # TODO: coming soon. Uncomment out the following when Luminous World is released.
    # Model(
    #     group="luminous",
    #     creator_organization="Aleph Alpha",
    #     name="AlephAlpha/luminous-world",
    #     display_name="Luminous World",
    #     description="Luminous World",
    #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Anthropic
    Model(
        group="anthropic",
        creator_organization="Anthropic",
        name="anthropic/stanford-online-all-v4-s3",
        display_name="Anthropic-LM (52B)",
        description="Anthropic model (52B parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG, ABLATION_MODEL_TAG],
    ),
    Model(
        group="anthropic",
        creator_organization="Anthropic",
        name="anthropic/claude-v1.3",
        display_name="Anthropic Claude (52B)",
        description="Anthropic Claude V1.3 (52B parameters)",
        tags=[
            ANTHROPIC_MODEL_TAG,
            TEXT_MODEL_TAG,
            LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
            GPT2_TOKENIZER_TAG,
            ABLATION_MODEL_TAG,
        ],
    ),
    Model(
        group="anthropic",
        creator_organization="Anthropic",
        name="anthropic/claude-instant-v1",
        display_name="Anthropic Claude Instant (TBD)",
        description="Anthropic Claude Instant (TBD parameters)",
        tags=[
            ANTHROPIC_MODEL_TAG,
            TEXT_MODEL_TAG,
            LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
            GPT2_TOKENIZER_TAG,
            ABLATION_MODEL_TAG,
        ],
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
        display_name="Cohere xlarge v20220609 (52.4B)",
        description="Cohere xlarge v20220609 (52.4B parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    Model(
        group="cohere",
        creator_organization="Cohere",
        name="cohere/xlarge-20221108",
        display_name="Cohere xlarge v20221108 (52.4B)",
        description="Cohere xlarge v20221108 (52.4B parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    Model(
        group="cohere",
        creator_organization="Cohere",
        name="cohere/large-20220720",
        display_name="Cohere large v20220720 (13.1B)",
        description="Cohere large v20220720 (13.1B parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    Model(
        group="cohere",
        creator_organization="Cohere",
        name="cohere/medium-20220720",
        display_name="Cohere medium v20220720 (6.1B)",
        description="Cohere medium v20220720 (6.1B parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    Model(
        group="cohere",
        creator_organization="Cohere",
        name="cohere/medium-20221108",
        display_name="Cohere medium v20221108 (6.1B)",
        description="Cohere medium v20221108 (6.1B parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    Model(
        group="cohere",
        creator_organization="Cohere",
        name="cohere/small-20220720",
        display_name="Cohere small v20220720 (410M)",
        description="Cohere small v20220720 (410M parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    Model(
        group="cohere",
        creator_organization="Cohere",
        name="cohere/command-medium-beta",
        display_name="Cohere Command beta (6.1B)",
        description="Cohere Command beta (6.1B parameters) is fine-tuned from the medium model "
        "to respond well with instruction-like prompts",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    Model(
        group="cohere",
        creator_organization="Cohere",
        name="cohere/command-xlarge-beta",
        display_name="Cohere Command beta (52.4B)",
        description="Cohere Command beta (52.4B parameters) is fine-tuned from the XL model "
        "to respond well with instruction-like prompts",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    # EleutherAI
    Model(
        group="together",
        creator_organization="EleutherAI",
        name="together/gpt-j-6b",
        display_name="GPT-J (6B)",
        description="GPT-J (6B parameters) autoregressive language model trained on The Pile",
        tags=[
            TEXT_MODEL_TAG,
            FULL_FUNCTIONALITY_TEXT_MODEL_TAG,
            ABLATION_MODEL_TAG,
            GPTJ_TOKENIZER_TAG,
            BUGGY_TEMP_0_TAG,
        ],
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
        creator_organization="OpenAI",
        name="huggingface/gpt2",
        display_name="GPT-2 (1.5B)",
        description="GPT-2 (1.5B parameters)",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface",
        creator_organization="EleutherAI",
        name="huggingface/gpt-j-6b",
        display_name="GPT-J (6B, HuggingFace)",
        description="GPT-J (6B parameters) autoregressive language model trained on The Pile.",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPTJ_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface",
        creator_organization="BigCode",
        name="huggingface/santacoder",
        display_name="SantaCoder (1.1B)",
        description="SantaCoder (1.1B parameters) model trained on the Python, Java, and "
        "JavaScript subset of The Stack (v1.1).",
        tags=[CODE_MODEL_TAG],
    ),
    Model(
        group="huggingface",
        creator_organization="BigCode",
        name="huggingface/starcoder",
        display_name="StarCoder (15.5B)",
        description="StarCoder (15.5B parameter) model trained on 80+ programming languages from The Stack (v1.2)",
        tags=[CODE_MODEL_TAG],
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
        name="together/flan-t5-xxl",
        display_name="Flan-T5 (11B)",
        description="Flan-T5 (11B parameters) is T5 fine-tuned on 1.8K tasks.",
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
    # H3 model
    Model(
        group="together",
        creator_organization="HazyResearch",
        name="together/h3-2.7b",
        display_name="H3 (2.7B)",
        description="H3 (2.7B parameters) is a decoder-only language model based on state space models.",
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
    Model(
        group="together",
        creator_organization="Meta",
        name="together/opt-6.7b",
        display_name="OPT (6.7B)",
        # From https://arxiv.org/pdf/2205.01068.pdf
        description="Open Pre-trained Transformers (6.7B parameters) is a suite of decoder-only pre-trained "
        "transformers that are fully and responsibly shared with interested researchers.",
        tags=[
            TEXT_MODEL_TAG,
            FULL_FUNCTIONALITY_TEXT_MODEL_TAG,
            ABLATION_MODEL_TAG,
            OPT_TOKENIZER_TAG,
            BUGGY_TEMP_0_TAG,
        ],
    ),
    Model(
        group="together",
        creator_organization="Meta",
        name="together/opt-1.3b",
        display_name="OPT (1.3B)",
        # From https://arxiv.org/pdf/2205.01068.pdf
        description="Open Pre-trained Transformers (1.3B parameters) is a suite of decoder-only pre-trained "
        "transformers that are fully and responsibly shared with interested researchers.",
        tags=[
            TEXT_MODEL_TAG,
            FULL_FUNCTIONALITY_TEXT_MODEL_TAG,
            ABLATION_MODEL_TAG,
            OPT_TOKENIZER_TAG,
            BUGGY_TEMP_0_TAG,
        ],
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
        display_name="davinci (175B)",
        description="Original GPT-3 (175B parameters) autoregressive language model.",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/curie",
        display_name="curie (6.7B)",
        description="Original GPT-3 (6.7B parameters) autoregressive language model.",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/babbage",
        display_name="babbage (1.3B)",
        description="Original GPT-3 (1.3B parameters) autoregressive language model.",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/ada",
        display_name="ada (350M)",
        description="Original GPT-3 (350M parameters) autoregressive language model.",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    # TODO: text-davinci-002 supports insertion. Support insertion in our framework.
    #       https://github.com/stanford-crfm/benchmarking/issues/359
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-davinci-003",
        display_name="text-davinci-003",
        description="text-davinci-003 model that involves reinforcement learning (PPO) with reward models."
        "Derived from text-davinci-002.",
        tags=[TEXT_MODEL_TAG, WIDER_CONTEXT_WINDOW_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-davinci-002",
        display_name="text-davinci-002",
        description="text-davinci-002 model that involves supervised fine-tuning on human-written demonstrations."
        "Derived from code-davinci-002.",
        tags=[TEXT_MODEL_TAG, WIDER_CONTEXT_WINDOW_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-davinci-001",
        display_name="text-davinci-001",
        description="text-davinci-001 model that involves supervised fine-tuning on human-written demonstrations",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-curie-001",
        display_name="text-curie-001",
        description="text-curie-001 model that involves supervised fine-tuning on human-written demonstrations",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-babbage-001",
        display_name="text-babbage-001",
        description="text-babbage-001 model that involves supervised fine-tuning on human-written demonstrations",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/text-ada-001",
        display_name="text-ada-001",
        description="text-ada-001 model that involves supervised fine-tuning on human-written demonstrations",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="codex",
        creator_organization="OpenAI",
        name="openai/code-davinci-002",
        display_name="code-davinci-002",
        description="code-davinci-002 model that is designed for pure code-completion tasks",
        tags=[CODE_MODEL_TAG, WIDER_CONTEXT_WINDOW_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="codex",
        creator_organization="OpenAI",
        name="openai/code-davinci-001",
        display_name="code-davinci-001",
        description="code-davinci-001 model",
        tags=[CODE_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="codex",
        creator_organization="OpenAI",
        name="openai/code-cushman-001",
        display_name="code-cushman-001 (12B)",
        description="Code model that is a stronger, multilingual version of the Codex (12B) model in the paper.",
        tags=[CODE_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    # GPT-4
    Model(
        group="gpt4",
        creator_organization="OpenAI",
        name="openai/gpt-4-0314",
        display_name="gpt-4-0314",
        # https://platform.openai.com/docs/models/gpt-4
        description="GPT-4 is a large multimodal model (currently only accepting text inputs and emitting text "
        "outputs) that is optimized for chat but works well for traditional completions tasks. Snapshot of gpt-4 "
        "from March 14th 2023.",
        tags=[
            TEXT_MODEL_TAG,
            GPT4_CONTEXT_WINDOW_TAG,
            OPENAI_CHATGPT_MODEL_TAG,
            LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
            GPT2_TOKENIZER_TAG,
        ],
    ),
    Model(
        group="gpt4",
        creator_organization="OpenAI",
        name="openai/gpt-4-32k-0314",
        display_name="gpt-4-32k-0314",
        # https://platform.openai.com/docs/models/gpt-4
        description="GPT-4 is a large multimodal model (currently only accepting text inputs and emitting text "
        "outputs) that is optimized for chat but works well for traditional completions tasks. Snapshot of gpt-4 "
        "with a longer context length of 32,768 tokens from March 14th 2023.",
        tags=[
            TEXT_MODEL_TAG,
            GPT4_32K_CONTEXT_WINDOW_TAG,
            OPENAI_CHATGPT_MODEL_TAG,
            LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
            GPT2_TOKENIZER_TAG,
        ],
    ),
    # ChatGPT: https://openai.com/blog/chatgpt
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/gpt-3.5-turbo-0301",
        display_name="gpt-3.5-turbo-0301",
        # https://platform.openai.com/docs/models/gpt-3-5
        description="Sibling model of text-davinci-003 is optimized for chat but works well "
        "for traditional completions tasks as well. Snapshot from 2023-03-01.",
        # The claimed sequence length is 4096, but as of 2023-03-07, the empirical usable
        # sequence length is smaller at 4087 with one user input message and one assistant
        # output message because ChatGPT uses special tokens for message roles and boundaries.
        # We use a rounded-down sequence length of 4000 to account for these special tokens.
        tags=[
            TEXT_MODEL_TAG,
            WIDER_CONTEXT_WINDOW_TAG,
            GPT4_TOKENIZER_TAG,
            OPENAI_CHATGPT_MODEL_TAG,
            LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
        ],
    ),
    Model(
        group="gpt3",
        creator_organization="OpenAI",
        name="openai/chat-gpt",
        display_name="ChatGPT",
        description="Sibling model to InstructGPT which interacts in a conversational way",
        # TODO: The max context length is unknown. Assume it's the same length as Davinci Instruct for now.
        tags=[TEXT_MODEL_TAG, WIDER_CONTEXT_WINDOW_TAG, GPT2_TOKENIZER_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
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
    # Together
    Model(
        group="together",
        creator_organization="Together",
        name="together/gpt-jt-6b-v1",
        display_name="GPT-JT (6B)",
        description="GPT-JT (6B parameters) is a fork of GPT-J",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPTJ_TOKENIZER_TAG],
    ),
    Model(
        group="together",
        creator_organization="Together",
        name="together/gpt-neoxt-chat-base-20b",
        display_name="GPT-NeoXT-Chat-Base (20B)",
        description="GPT-NeoXT-Chat-Base (20B parameters) is a fork of GPT-NeoX",
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG, CHATML_MODEL_TAG, GPTNEO_TOKENIZER_TAG],
    ),
    # Tsinghua
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
    # Writer
    Model(
        group="palmyra",
        creator_organization="Writer",
        name="writer/palmyra-base",
        display_name="Palmyra Base (5B)",
        description="Autoregressive language model (5B parameters)",
        # Does not support echo
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="palmyra",
        creator_organization="Writer",
        name="writer/palmyra-large",
        display_name="Palmyra Large (20B)",
        description="Autoregressive language model (20B parameters)",
        # Does not support echo
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="palmyra",
        creator_organization="Writer",
        name="writer/palmyra-r",
        display_name="Palmyra R (30B)",
        description="Autoregressive language model with Retrieval-Augmented Generation (30B parameters)",
        # Does not support echo
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="palmyra",
        creator_organization="Writer",
        name="writer/camel",
        display_name="Camel (5B)",
        description="Training language models to follow instructions with human feedback (5B parameters)",
        # Does not support echo
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="palmyra",
        creator_organization="Writer",
        name="writer/palmyra-instruct-30",
        display_name="InstructPalmyra (30B)",
        description="Training language models to follow instructions with human feedback (30B parameters)",
        # Does not support echo
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="palmyra",
        creator_organization="Writer",
        name="writer/palmyra-e",
        display_name="Palmyra E (30B)",
        description="Training language models to follow instructions with human feedback (30B parameters)",
        # Does not support echo
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="palmyra",
        creator_organization="Writer",
        name="writer/silk-road",
        display_name="Silk Road (35B)",
        description="Autoregressive language mode with multi-scale Attention in parallel (35B parameters)",
        # Does not support echo
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    # Yandex
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
    # PaLM
    Model(
        group="google",
        creator_organization="Google",
        name="google/palm",
        display_name="PaLM (540B)",
        description="Pathways Language Model (540B parameters) is trained using 6144 TPU v4 chips "
        "([paper](https://arxiv.org/pdf/2204.02311.pdf)).",
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    # Nvidia
    Model(
        group="nvidia",
        creator_organization="Nvidia",
        name="nvidia/megatron-gpt2",
        display_name="Megatron GPT-2",
        description="GPT-2 implemented in Megatron-LM ([paper](https://arxiv.org/abs/1909.08053)).",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG, BUGGY_TEMP_0_TAG],
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
