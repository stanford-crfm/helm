from dataclasses import dataclass, field
from typing import List, Dict

# Different modalities
TEXT_MODEL_TAG: str = "text"
IMAGE_MODEL_TAG: str = "image"
CODE_MODEL_TAG: str = "code"
EMBEDDING_MODEL_TAG: str = "embedding"
TEXT_TO_IMAGE_MODEL_TAG: str = "text_to_image"

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
CLIP_TOKENIZER_TAG: str = "clip_tokenizer"

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
    """
    Represents a model that we can make requests to.  Conceptually, an instance
    of `Model` is tied more to the hosting implementation (where can we send
    requests) rather than the conceptual model.  These are the same for closed
    models, but different for open-source models.  Note: for all the metadata
    and documentation about the model itself, see `ModelField` in `schema.py`.
    """

    # Model group, used to determine quotas (e.g. "huggingface").
    # This group is only for user accounts, not benchmarking, and should probably
    # called something else.
    group: str

    # Name of the specific model (e.g. "huggingface/gpt-j-6b")
    # Currently, the name is <hosting_organization>/<model_name>,
    # There is also `creator_organization>` (see `ModelField`).
    name: str

    # Tags corresponding to the properties of the model.
    tags: List[str] = field(default_factory=list)

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
        name="ai21/j1-jumbo",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    ),
    # From AI21: "the new model is a mid-point in terms of size, cost and performance between Jumbo and Large.
    # We also implemented a few tweaks to its training process. Internal benchmarks suggest it can really
    # help the unit economics on your end compared to Jumbo, without compromising too much on quality."
    Model(
        group="jurassic",
        name="ai21/j1-grande",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    ),
    Model(
        group="jurassic",
        name="ai21/j1-grande-v2-beta",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    ),
    Model(
        group="jurassic",
        name="ai21/j1-large",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    ),
    # AI21 Jurassic-2 Models: https://www.ai21.com/blog/introducing-j2
    Model(
        group="jurassic",
        name="ai21/j2-jumbo",
        tags=[
            TEXT_MODEL_TAG,
            AI21_JURASSIC_2_JUMBO_CONTEXT_WINDOW_TAG,
            FULL_FUNCTIONALITY_TEXT_MODEL_TAG,
            AI21_TOKENIZER_TAG,
        ],
    ),
    Model(
        group="jurassic",
        name="ai21/j2-grande",
        tags=[TEXT_MODEL_TAG, AI21_WIDER_CONTEXT_WINDOW_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    ),
    Model(
        group="jurassic",
        name="ai21/j2-large",
        tags=[TEXT_MODEL_TAG, AI21_WIDER_CONTEXT_WINDOW_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    ),
    # Aleph Alpha's Luminous models: https://docs.aleph-alpha.com/docs/introduction/luminous
    Model(
        group="luminous",
        name="AlephAlpha/luminous-base",
        # Does not support echo
        tags=[TEXT_MODEL_TAG, IMAGE_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="luminous",
        name="AlephAlpha/luminous-extended",
        # Does not support echo
        tags=[TEXT_MODEL_TAG, IMAGE_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="luminous",
        name="AlephAlpha/luminous-supreme",
        # Does not support echo.
        # TODO: images will be supported in the near future. Add IMAGE_MODEL_TAG.
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    # TODO: coming soon. Uncomment out the following when Luminous World is released.
    # Model(
    #     group="luminous",
    #     name="AlephAlpha/luminous-world",
    #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Anthropic
    Model(
        group="anthropic",
        name="anthropic/stanford-online-all-v4-s3",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG, ABLATION_MODEL_TAG],
    ),
    Model(
        group="anthropic",
        name="anthropic/claude-v1.3",
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
        name="anthropic/claude-instant-v1",
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
        name="together/bloom",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG],
    ),
    Model(
        group="together",
        name="together/t0pp",
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
        name="cohere/xlarge-20220609",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    Model(
        group="cohere",
        name="cohere/xlarge-20221108",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    Model(
        group="cohere",
        name="cohere/large-20220720",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    Model(
        group="cohere",
        name="cohere/medium-20220720",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    Model(
        group="cohere",
        name="cohere/medium-20221108",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    Model(
        group="cohere",
        name="cohere/small-20220720",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    Model(
        group="cohere",
        name="cohere/command-medium-beta",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    Model(
        group="cohere",
        name="cohere/command-xlarge-beta",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    ),
    # EleutherAI
    Model(
        group="together",
        name="together/gpt-j-6b",
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
        name="together/gpt-neox-20b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG, GPTNEO_TOKENIZER_TAG],
    ),
    Model(
        group="together",
        name="together/pythia-7b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    # Meta
    Model(
        group="together",
        name="together/llama-7b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    # Stanford
    Model(
        group="together",
        name="together/alpaca-7b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    # LMSYS
    Model(
        group="together",
        name="together/vicuna-13b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    # MosaicML
    Model(
        group="togethger",
        name="together/mpt-7b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    # GooseAI supported models
    Model(
        group="gooseai",
        name="gooseai/gpt-neo-20b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPTNEO_TOKENIZER_TAG],
    ),
    Model(
        group="gooseai",
        name="gooseai/gpt-j-6b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPTJ_TOKENIZER_TAG],
    ),
    # HuggingFace
    Model(
        group="huggingface",
        name="huggingface/gpt2",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface",
        name="huggingface/gpt-j-6b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPTJ_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface",
        name="huggingface/santacoder",
        tags=[CODE_MODEL_TAG],
    ),
    Model(
        group="huggingface",
        name="huggingface/starcoder",
        tags=[CODE_MODEL_TAG],
    ),
    # Google
    Model(
        group="together",
        name="together/t5-11b",
        # Does not support echo=True
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG, NO_NEWLINES_TAG],
    ),
    Model(
        group="together",
        name="together/flan-t5-xxl",
        # Does not support echo=True
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG, NO_NEWLINES_TAG],
    ),
    Model(
        group="together",
        name="together/ul2",
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
        name="together/h3-2.7b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    # OPT
    Model(
        group="together",
        name="together/opt-175b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG, OPT_TOKENIZER_TAG],
    ),
    Model(
        group="together",
        name="together/opt-66b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG, OPT_TOKENIZER_TAG],
    ),
    Model(
        group="together",
        name="together/opt-6.7b",
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
        name="together/opt-1.3b",
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
        name="microsoft/TNLGv2_530B",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="microsoft",
        name="microsoft/TNLGv2_7B",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    # OpenAI: https://beta.openai.com/docs/engines/gpt-3
    Model(
        group="gpt3",
        name="openai/davinci",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        name="openai/curie",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        name="openai/babbage",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        name="openai/ada",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    # TODO: text-davinci-002 supports insertion. Support insertion in our framework.
    #       https://github.com/stanford-crfm/benchmarking/issues/359
    Model(
        group="gpt3",
        name="openai/text-davinci-003",
        tags=[TEXT_MODEL_TAG, WIDER_CONTEXT_WINDOW_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        name="openai/text-davinci-002",
        tags=[TEXT_MODEL_TAG, WIDER_CONTEXT_WINDOW_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        name="openai/text-davinci-001",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        name="openai/text-curie-001",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        name="openai/text-babbage-001",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="gpt3",
        name="openai/text-ada-001",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="codex",
        name="openai/code-davinci-002",
        tags=[CODE_MODEL_TAG, WIDER_CONTEXT_WINDOW_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="codex",
        name="openai/code-davinci-001",
        tags=[CODE_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    Model(
        group="codex",
        name="openai/code-cushman-001",
        tags=[CODE_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
    # GPT-4
    Model(
        group="gpt4",
        name="openai/gpt-4-0314",
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
        name="openai/gpt-4-32k-0314",
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
        name="openai/gpt-3.5-turbo-0301",
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
        name="openai/chat-gpt",
        # TODO: The max context length is unknown. Assume it's the same length as Davinci Instruct for now.
        tags=[TEXT_MODEL_TAG, WIDER_CONTEXT_WINDOW_TAG, GPT2_TOKENIZER_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    # OpenAI similarity embedding models: https://beta.openai.com/docs/guides/embeddings
    Model(
        group="gpt3",
        name="openai/text-similarity-davinci-001",
        tags=[EMBEDDING_MODEL_TAG],
    ),
    Model(
        group="gpt3",
        name="openai/text-similarity-curie-001",
        tags=[EMBEDDING_MODEL_TAG],
    ),
    Model(
        group="gpt3",
        name="openai/text-similarity-babbage-001",
        tags=[EMBEDDING_MODEL_TAG],
    ),
    Model(
        group="gpt3",
        name="openai/text-similarity-ada-001",
        tags=[EMBEDDING_MODEL_TAG],
    ),
    # Together
    Model(
        group="together",
        name="together/gpt-jt-6b-v1",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPTJ_TOKENIZER_TAG],
    ),
    Model(
        group="together",
        name="together/gpt-neoxt-chat-base-20b",
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG, CHATML_MODEL_TAG, GPTNEO_TOKENIZER_TAG],
    ),
    Model(
        group="together",
        name="together/redpajama-incite-base-3b-v1",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    # Tsinghua
    Model(
        group="together",
        name="together/glm",
        # Inference with echo=True is not feasible -- in the prompt encoding phase, they use
        # bidirectional attention and do not perform predictions on them.
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG, NO_NEWLINES_TAG],
    ),
    # Writer
    Model(
        group="palmyra",
        name="writer/palmyra-base",
        # Does not support echo
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="palmyra",
        name="writer/palmyra-large",
        # Does not support echo
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="palmyra",
        name="writer/palmyra-r",
        # Does not support echo
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="palmyra",
        name="writer/camel",
        # Does not support echo
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="palmyra",
        name="writer/palmyra-instruct-30",
        # Does not support echo
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="palmyra",
        name="writer/palmyra-e",
        # Does not support echo
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="palmyra",
        name="writer/silk-road",
        # Does not support echo
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    # Yandex
    Model(
        group="together",
        name="together/yalm",
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG],
    ),
    # Google
    Model(
        group="google",
        name="google/palm",
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    # NVIDIA
    Model(
        group="nvidia",
        name="nvidia/megatron-gpt2",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG, BUGGY_TEMP_0_TAG],
    ),
    # For debugging
    Model(
        group="simple",
        name="simple/model1",
    ),
    # Text-to-image models
    Model(
        group="magma",
        creator_organization="Aleph Alpha",
        name="AlephAlpha/m-vader",
        display_name="M-VADER",
        description="M-VADER is similar to Stable Diffusion, but it supports multimodal inputs and the text encoder "
        "is a Luminous model.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="giga_gan",
        creator_organization="Adobe",
        name="adobe/giga-gan",
        display_name="GigaGAN (1B)",
        description="GigaGAN (1B parameters)",
        # TODO: add TEXT_TO_IMAGE_MODEL_TAG later after the first batch of results
        tags=[CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="firefly",
        creator_organization="Adobe",
        name="adobe/firefly",
        display_name="Firefly",
        description="Adobe Firefly was trained on the Adobe Stock dataset, along with openly licensed work and "
        "public domain content where copyright has expired.",
        # TODO: add TEXT_TO_IMAGE_MODEL_TAG later after the first batch of results
        tags=[CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="dall_e2",
        creator_organization="OpenAI",
        name="openai/dalle-2",
        display_name="DALL-E 2 (3.5B)",
        description="DALL-E 2 (3.5B parameters) is a model that can create realistic images and art "
        "from a description in natural language.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG],
    ),
    Model(
        group="lexica",
        creator_organization="Lexica",
        name="lexica/search-stable-diffusion-1.5",
        display_name="Lexica Search with Stable Diffusion v1.5",
        description="Searches Stable Diffusion v1.5 images Lexica users generated.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG],
    ),
    Model(
        group="deepfloyd_if",
        creator_organization="DeepFloyd",
        name="DeepFloyd/IF-I-M-v1.0",
        display_name="DeepFloyd IF (medium)",
        description="DeepFloyd IF is a novel state-of-the-art open-source text-to-image model "
        "with a high degree of photorealism and language understanding.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="deepfloyd_if",
        creator_organization="DeepFloyd",
        name="DeepFloyd/IF-I-L-v1.0",
        display_name="DeepFloyd IF (large)",
        description="DeepFloyd IF is a novel state-of-the-art open-source text-to-image model "
        "with a high degree of photorealism and language understanding.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="deepfloyd_if",
        creator_organization="DeepFloyd",
        name="DeepFloyd/IF-I-XL-v1.0",
        display_name="DeepFloyd IF (XL)",
        description="DeepFloyd IF is a novel state-of-the-art open-source text-to-image model "
        "with a high degree of photorealism and language understanding.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="mindall-e",
        creator_organization="kakaobrain",
        name="kakaobrain/mindall-e",
        display_name="minDALL-E (1.3B)",
        description="minDALL-E, named after minGPT, is a 1.3B text-to-image generation model trained "
        "on 14 million image-text pairs for non-commercial purposes.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="dall_e",
        creator_organization="craiyon",
        name="craiyon/dalle-mini",
        display_name="DALL-E mini (0.4B)",
        description="DALL-E mini (0.4B parameters) is an open-source text-to-image model that "
        "attempt to reproduce OpenAI DALL-E 1.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="dall_e",
        creator_organization="craiyon",
        name="craiyon/dalle-mega",
        display_name="DALL-E mega (2.6B)",
        description="DALL-E mega (2.6B parameters) is an open-source text-to-image model that "
        "attempt to reproduce OpenAI DALL-E 1.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="cogview",
        creator_organization="THUDM",
        name="thudm/cogview2",
        display_name="CogView2",
        description="CogView2 is a hierarchical transformer (6B-9B-9B parameters) for text-to-"
        "image generation in general domain.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        creator_organization="dreamlike.art",
        name="huggingface/dreamlike-photoreal-v2-0",
        display_name="Dreamlike Photoreal v2.0",
        description="Dreamlike Photoreal v2.0 is a photorealistic model based on Stable Diffusion v1.5.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        creator_organization="dreamlike.art",
        name="huggingface/dreamlike-diffusion-v1-0",
        display_name="Dreamlike Diffusion v1.0",
        description="Dreamlike Diffusion v1.0 is Stable Diffusion v1.5 fine tuned on high quality art.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        creator_organization="PromptHero",
        name="huggingface/openjourney-v1-0",
        display_name="Openjourney v1.0",
        description="Openjourney is an open source Stable Diffusion fine tuned model on Midjourney images",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        creator_organization="PromptHero",
        name="huggingface/openjourney-v2-0",
        display_name="Openjourney v2.0",
        description="Openjourney v2 is an open source Stable Diffusion fine tuned model on +60k Midjourney images",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        creator_organization="nitrosocke",
        name="huggingface/redshift-diffusion",
        display_name="Redshift Diffusion",
        description="Redshift Diffusion is an open source Stable Diffusion fine tuned on high resolution 3D artworks.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        creator_organization="Microsoft",
        name="huggingface/promptist-stable-diffusion-v1-4",
        display_name="Promptist + Stable Diffusion v1.4",
        description="Promptist optimizes user input into model-preferred prompts for Stable Diffusion v1.4.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        creator_organization="Ludwig Maximilian University of Munich CompVis",
        name="huggingface/stable-diffusion-v1-4",
        display_name="Stable Diffusion v1.4",
        description="Stable Diffusion v1.4 is a latent text-to-image diffusion model capable of generating "
        "photo-realistic images given any text input.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        creator_organization="Runway",
        name="huggingface/stable-diffusion-v1-5",
        display_name="Stable Diffusion v1.5",
        description="The Stable-Diffusion-v1-5 checkpoint was initialized with the weights of the "
        "Stable-Diffusion-v1-2 checkpoint and subsequently fine-tuned on 595k steps at "
        "resolution 512x512 on laion-aesthetics v2 5+ and 10% dropping of the text-conditioning "
        "to improve classifier-free guidance sampling.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        creator_organization="Stability AI",
        name="huggingface/stable-diffusion-v2-base",
        display_name="Stable Diffusion v2 base",
        description="The model is trained from scratch 550k steps at resolution 256x256 on a subset of LAION-5B "
        "filtered for explicit pornographic material, using the LAION-NSFW classifier with punsafe=0.1 "
        "and an aesthetic score >= 4.5. Then it is further trained for 850k steps at resolution 512x512 "
        "on the same dataset on images with resolution >= 512x512.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        creator_organization="Stability AI",
        name="huggingface/stable-diffusion-v2-1-base",
        display_name="Stable Diffusion v2.1 base",
        description="This stable-diffusion-2-1-base model fine-tunes stable-diffusion-2-base "
        "with 220k extra steps taken, with punsafe=0.98 on the same dataset.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        creator_organization="TU Darmstadt",
        name="huggingface/stable-diffusion-safe-weak",
        display_name="Safe Stable Diffusion weak",
        description="Safe Stable Diffusion is an extension to the Stable Diffusion that drastically reduces "
        "inappropriate content.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        creator_organization="TU Darmstadt",
        name="huggingface/stable-diffusion-safe-medium",
        display_name="Safe Stable Diffusion medium",
        description="Safe Stable Diffusion is an extension to the Stable Diffusion that drastically reduces "
        "inappropriate content.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        creator_organization="TU Darmstadt",
        name="huggingface/stable-diffusion-safe-strong",
        display_name="Safe Stable Diffusion strong",
        description="Safe Stable Diffusion is an extension to the Stable Diffusion that drastically reduces "
        "inappropriate content.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        creator_organization="TU Darmstadt",
        name="huggingface/stable-diffusion-safe-max",
        display_name="Safe Stable Diffusion max",
        description="Safe Stable Diffusion is an extension to the Stable Diffusion that drastically reduces "
        "inappropriate content.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        creator_organization="22 Hours",
        name="huggingface/vintedois-diffusion-v0-1",
        display_name="Vintedois (22h) Diffusion model v0.1",
        description="Vintedois (22h) Diffusion model v0.1 is Stable Diffusion v1.5 that was finetuned on a "
        "large amount of high quality images with simple prompts to generate beautiful images "
        "without a lot of prompt engineering.",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
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


def get_models_with_tag(tag: str) -> List[Model]:
    """Get all models with tag `tag`."""
    return [model for model in ALL_MODELS if tag in model.tags]


def get_all_text_models() -> List[str]:
    """Get all text model names."""
    return get_model_names_with_tag(TEXT_MODEL_TAG)


def get_all_code_models() -> List[str]:
    """Get all code model names."""
    return get_model_names_with_tag(CODE_MODEL_TAG)


def is_text_to_image_model(model_name: str) -> bool:
    model: Model = get_model(model_name)
    return TEXT_TO_IMAGE_MODEL_TAG in model.tags
