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
WIDER_CONTEXT_WINDOW_TAG: str = "openai_wider_context_window"  # huggingface/gpt2 tokenizer, 4000 tokens
GPT_TURBO_CONTEXT_WINDOW_TAG: str = "gpt_turbo_context_window"  # cl100k_base tokenizer, 4000 tokens
GPT_TURBO_16K_CONTEXT_WINDOW_TAG: str = "gpt_turbo_16k_context_window"  # cl100k_base tokenizer, 8000 tokens
GPT4_CONTEXT_WINDOW_TAG: str = "gpt4_context_window"  # cl100k_base tokenizer, 8192 tokens
GPT4_32K_CONTEXT_WINDOW_TAG: str = "gpt4_32k_context_window"  # cl100k_base tokenizer, 32768 tokens

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

# Whether the HuggingFace model needs to be loaded locally
LOCAL_HUGGINGFACE_MODEL_TAG: str = "local_huggingface_model"

# Some models can follow instructions.
INSTRUCTION_FOLLOWING_MODEL_TAG: str = "instruction_following"


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
    # The name is <hosting_organization>/<model_name> or
    # <creator_organization>/<model_name>
    # There is also `<creator_organization>` (see `ModelField`).
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
            INSTRUCTION_FOLLOWING_MODEL_TAG,
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
            INSTRUCTION_FOLLOWING_MODEL_TAG,
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
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG, INSTRUCTION_FOLLOWING_MODEL_TAG],
    ),
    Model(
        group="cohere",
        name="cohere/command-xlarge-beta",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG, INSTRUCTION_FOLLOWING_MODEL_TAG],
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
        name="eleutherai/pythia-1b-v0",
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="together",
        name="eleutherai/pythia-2.8b-v0",
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="together",
        name="eleutherai/pythia-6.9b",
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="together",
        name="eleutherai/pythia-12b-v0",
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    # Meta
    Model(
        group="together",
        name="meta/llama-7b",
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="together",
        name="meta/llama-13b",
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="together",
        name="meta/llama-30b",
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="together",
        name="meta/llama-65b",
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="together",
        name="meta/llama-2-7b",
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="together",
        name="meta/llama-2-13b",
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="together",
        name="meta/llama-2-70b",
        tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    # Stanford
    Model(
        group="together",
        name="together/alpaca-7b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, INSTRUCTION_FOLLOWING_MODEL_TAG],
    ),
    # LMSYS
    Model(
        group="together",
        name="together/vicuna-13b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, INSTRUCTION_FOLLOWING_MODEL_TAG],
    ),
    # MosaicML
    Model(
        group="huggingface",
        name="mosaicml/mpt-7b",
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
    # Local HuggingFace models
    # Copy these models (in HuggingFace format) to ./huggingface_models
    Model(
        group="huggingface",
        name="huggingface/llama-7b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, LOCAL_HUGGINGFACE_MODEL_TAG],
    ),
    Model(
        group="huggingface",
        name="huggingface/alpaca-7b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, LOCAL_HUGGINGFACE_MODEL_TAG],
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
        tags=[
            TEXT_MODEL_TAG,
            LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
            ABLATION_MODEL_TAG,
            NO_NEWLINES_TAG,
            INSTRUCTION_FOLLOWING_MODEL_TAG,
        ],
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
        tags=[
            TEXT_MODEL_TAG,
            WIDER_CONTEXT_WINDOW_TAG,
            FULL_FUNCTIONALITY_TEXT_MODEL_TAG,
            GPT2_TOKENIZER_TAG,
            INSTRUCTION_FOLLOWING_MODEL_TAG,
        ],
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
            GPT4_TOKENIZER_TAG,
            OPENAI_CHATGPT_MODEL_TAG,
            LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
            INSTRUCTION_FOLLOWING_MODEL_TAG,
        ],
    ),
    Model(
        group="gpt4",
        name="openai/gpt-4-32k-0314",
        tags=[
            TEXT_MODEL_TAG,
            GPT4_32K_CONTEXT_WINDOW_TAG,
            GPT4_TOKENIZER_TAG,
            OPENAI_CHATGPT_MODEL_TAG,
            LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
            INSTRUCTION_FOLLOWING_MODEL_TAG,
        ],
    ),
    Model(
        group="gpt4",
        name="openai/gpt-4-0613",
        tags=[
            TEXT_MODEL_TAG,
            GPT4_CONTEXT_WINDOW_TAG,
            GPT4_TOKENIZER_TAG,
            OPENAI_CHATGPT_MODEL_TAG,
            LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
            INSTRUCTION_FOLLOWING_MODEL_TAG,
        ],
    ),
    Model(
        group="gpt4",
        name="openai/gpt-4-32k-0613",
        tags=[
            TEXT_MODEL_TAG,
            GPT4_32K_CONTEXT_WINDOW_TAG,
            GPT4_TOKENIZER_TAG,
            OPENAI_CHATGPT_MODEL_TAG,
            LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
            INSTRUCTION_FOLLOWING_MODEL_TAG,
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
            GPT_TURBO_CONTEXT_WINDOW_TAG,
            GPT4_TOKENIZER_TAG,
            OPENAI_CHATGPT_MODEL_TAG,
            LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
            INSTRUCTION_FOLLOWING_MODEL_TAG,
        ],
    ),
    Model(
        group="gpt3",
        name="openai/gpt-3.5-turbo-0613",
        # The claimed sequence length is 4096, but as of 2023-03-07, the empirical usable
        # sequence length is smaller at 4087 with one user input message and one assistant
        # output message because ChatGPT uses special tokens for message roles and boundaries.
        # We use a rounded-down sequence length of 4000 to account for these special tokens.
        tags=[
            TEXT_MODEL_TAG,
            GPT_TURBO_CONTEXT_WINDOW_TAG,
            GPT4_TOKENIZER_TAG,
            OPENAI_CHATGPT_MODEL_TAG,
            LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
            INSTRUCTION_FOLLOWING_MODEL_TAG,
        ],
    ),
    Model(
        group="gpt3",
        name="openai/gpt-3.5-turbo-16k-0613",
        # Claimed length is 16,384; we round down to 16,000 for the same reasons as explained
        # in the openai/gpt-3.5-turbo-0613 comment
        tags=[
            TEXT_MODEL_TAG,
            GPT_TURBO_16K_CONTEXT_WINDOW_TAG,
            GPT4_TOKENIZER_TAG,
            OPENAI_CHATGPT_MODEL_TAG,
            LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
            INSTRUCTION_FOLLOWING_MODEL_TAG,
        ],
    ),
    Model(
        group="gpt3",
        name="openai/chat-gpt",
        # TODO: The max context length is unknown. Assume it's the same length as Davinci Instruct for now.
        tags=[
            TEXT_MODEL_TAG,
            WIDER_CONTEXT_WINDOW_TAG,
            GPT2_TOKENIZER_TAG,
            LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
            INSTRUCTION_FOLLOWING_MODEL_TAG,
        ],
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
    Model(
        group="gpt3",
        name="openai/text-embedding-ada-002",
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
    Model(
        group="together",
        name="together/redpajama-incite-instruct-3b-v1",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="together",
        name="together/redpajama-incite-base-7b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="together",
        name="together/redpajama-incite-instruct-7b",
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
    Model(
        group="palmyra",
        name="writer/palmyra-x",
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
    # Databricks
    Model(
        group="together",
        name="databricks/dolly-v2-3b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="together",
        name="databricks/dolly-v2-7b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="together",
        name="databricks/dolly-v2-12b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    # Stability AI
    Model(
        group="together",
        name="stabilityai/stablelm-base-alpha-3b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    Model(
        group="together",
        name="stabilityai/stablelm-base-alpha-7b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),
    # For debugging
    Model(
        group="simple",
        name="simple/model1",
    ),
    # Text-to-image models
    Model(
        group="magma",
        name="AlephAlpha/m-vader",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="giga_gan",
        name="adobe/giga-gan",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="firefly",
        name="adobe/firefly",
        # TODO: add TEXT_TO_IMAGE_MODEL_TAG later after the first batch of results
        tags=[CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="dall_e2",
        name="openai/dalle-2",
        tags=[TEXT_TO_IMAGE_MODEL_TAG],
    ),
    Model(
        group="lexica",
        name="lexica/search-stable-diffusion-1.5",
        tags=[TEXT_TO_IMAGE_MODEL_TAG],
    ),
    Model(
        group="deepfloyd_if",
        name="DeepFloyd/IF-I-M-v1.0",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="deepfloyd_if",
        name="DeepFloyd/IF-I-L-v1.0",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="deepfloyd_if",
        name="DeepFloyd/IF-I-XL-v1.0",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="mindall-e",
        name="kakaobrain/mindall-e",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="dall_e",
        name="craiyon/dalle-mini",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="dall_e",
        name="craiyon/dalle-mega",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="cogview",
        name="thudm/cogview2",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/dreamlike-photoreal-v2-0",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/dreamlike-diffusion-v1-0",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/openjourney-v1-0",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/openjourney-v2-0",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/redshift-diffusion",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/promptist-stable-diffusion-v1-4",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/stable-diffusion-v1-4",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/stable-diffusion-v1-5",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/stable-diffusion-v2-base",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/stable-diffusion-v2-1-base",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/stable-diffusion-safe-weak",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/stable-diffusion-safe-medium",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/stable-diffusion-safe-strong",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/stable-diffusion-safe-max",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/vintedois-diffusion-v0-1",
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


def get_all_instruction_following_models() -> List[str]:
    """Get all instruction-following model names."""
    return get_model_names_with_tag(INSTRUCTION_FOLLOWING_MODEL_TAG)
