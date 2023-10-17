# TODO(PR): Delete this file.

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from helm.benchmark.model_deployment_registry import ModelDeployment


@dataclass
class Model:
    """
    Represents a model that we can make requests to.  Conceptually, an instance
    of `Model` is tied more to the hosting implementation (where can we send
    requests) rather than the conceptual model.  These are the same for closed
    models, but different for open-source models.  Note: for all the metadata
    and documentation about the model itself, see `ModelField` in `schema.py`.
    """

    # Name of the model group (e.g. "openai/davinci").
    # This is the name of the model, not the name of the deployment.
    # Usually formatted as "<creator_organization>/<engine_name>".
    # Example: "ai21/j1-jumbo"
    name: str

    # Name that is going to be displayed to the user (on the website, etc.)
    display_name: str

    # Description of the model, to be displayed on the website.
    description: str

    # Description of the access level of the model.
    # Should be one of the following:
    # - "open": the model is open-source and can be downloaded from the internet.
    # - "closed": TODO
    # - "limited": TODO
    access: str

    # Number of parameters in the model.
    num_parameters: int

    # Release date of the model.
    release_date: str

    # Tags corresponding to the properties of the model.
    tags: List[str] = field(default_factory=list)

    # List of the model deployments for this model.
    # Should at least contain one model deployment.
    # Refers to the field "name" in the ModelDeployment class.
    # Defaults to a single model deployment with the same name as the model.
    deployment_names: Optional[List[str]] = None

    @property
    def creator_organization(self) -> str:
        """
        Extracts the creator organization from the model name.
        Example: 'ai21/j1-jumbo' => 'ai21'
        This can be different from the hosting organization.
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

# TODO(PR): Port all these models to the new format.

ALL_MODELS = [
    # # Local Model
    # Model(
    #     group="neurips",
    #     name="neurips/local",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    # ),
    # # AI21: https://studio.ai21.com/pricing
    # Model(
    #     group="jurassic",
    #     name="ai21/j1-jumbo",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    # ),
    # # From AI21: "the new model is a mid-point in terms of size, cost and performance between Jumbo and Large.
    # # We also implemented a few tweaks to its training process. Internal benchmarks suggest it can really
    # # help the unit economics on your end compared to Jumbo, without compromising too much on quality."
    # Model(
    #     group="jurassic",
    #     name="ai21/j1-grande",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="jurassic",
    #     name="ai21/j1-grande-v2-beta",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="jurassic",
    #     name="ai21/j1-large",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    # ),
    # # AI21 Jurassic-2 Models: https://www.ai21.com/blog/introducing-j2
    # Model(
    #     group="jurassic",
    #     name="ai21/j2-jumbo",
    #     tags=[
    #         TEXT_MODEL_TAG,
    #         AI21_JURASSIC_2_JUMBO_CONTEXT_WINDOW_TAG,
    #         FULL_FUNCTIONALITY_TEXT_MODEL_TAG,
    #         AI21_TOKENIZER_TAG,
    #     ],
    # ),
    # Model(
    #     group="jurassic",
    #     name="ai21/j2-grande",
    #     tags=[TEXT_MODEL_TAG, AI21_WIDER_CONTEXT_WINDOW_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="jurassic",
    #     name="ai21/j2-large",
    #     tags=[TEXT_MODEL_TAG, AI21_WIDER_CONTEXT_WINDOW_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    # ),
    # # Aleph Alpha's Luminous models: https://docs.aleph-alpha.com/docs/introduction/luminous
    # Model(
    #     group="luminous",
    #     name="AlephAlpha/luminous-base",
    #     # Does not support echo
    #     tags=[TEXT_MODEL_TAG, IMAGE_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="luminous",
    #     name="AlephAlpha/luminous-extended",
    #     # Does not support echo
    #     tags=[TEXT_MODEL_TAG, IMAGE_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="luminous",
    #     name="AlephAlpha/luminous-supreme",
    #     # Does not support echo.
    #     # TODO: images will be supported in the near future. Add IMAGE_MODEL_TAG.
    #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # # TODO: coming soon. Uncomment out the following when Luminous World is released.
    # # Model(
    # #     group="luminous",
    # #     name="AlephAlpha/luminous-world",
    # #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    # # ),
    # # Anthropic
    # Model(
    #     group="anthropic",
    #     name="anthropic/stanford-online-all-v4-s3",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG, ABLATION_MODEL_TAG],
    # ),
    Model(
        display_name="Anthropic Claude v1.3",
        description="A 52B parameter language model, trained using reinforcement learning from human feedback [paper](https://arxiv.org/pdf/2204.05862.pdf).",
        name="anthropic/claude-v1.3",
        access="limited",
        num_parameters=52000000000,
        release_date="2023-03-17",
        tags=[],
        # deployments=[
        #     ModelDeployment(
        #         name="anthropic/claude-v1.3",
        #         host_group="anthropic",
        #         tokenizer_name="anthropic/claude",
        #         max_sequence_length=8000,
        #         max_request_length=8000,
        #         # TODO: Missing max_sequence_and_generated_tokens_length
        #         client_spec=ClientSpec(
        #             class_name="helm.proxy.clients.anthropic_client.AnthropicClient",
        #             args=None,  # TODO: Figure this out. Should API key be defined here?
        #         ),
        #     )
        # ],
    ),
    # Model(
    #     group="anthropic",
    #     name="anthropic/claude-instant-v1",
    #     tags=[
    #         ANTHROPIC_MODEL_TAG,
    #         TEXT_MODEL_TAG,
    #         LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
    #         GPT2_TOKENIZER_TAG,
    #         ABLATION_MODEL_TAG,
    #         INSTRUCTION_FOLLOWING_MODEL_TAG,
    #     ],
    # ),
    # # BigScience
    # Model(
    #     group="together",
    #     name="together/bloom",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="together/t0pp",
    #     # Does not support echo=True
    #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG, NO_NEWLINES_TAG],
    # ),
    # # Cohere models
    # # Model versioning and the possible versions are not documented here:
    # # https://docs.cohere.ai/generate-reference#model-optional.
    # # So, instead, we got the names of the models from the Cohere Playground.
    # #
    # # Note that their tokenizer and model were trained on English text and
    # # they do not have a dedicated decode API endpoint, so the adaptation
    # # step for language modeling fails for certain Scenarios:
    # # the_pile:subset=ArXiv
    # # the_pile:subset=Github
    # # the_pile:subset=PubMed Central
    # Model(
    #     group="cohere",
    #     name="cohere/xlarge-20220609",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="cohere",
    #     name="cohere/xlarge-20221108",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="cohere",
    #     name="cohere/large-20220720",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="cohere",
    #     name="cohere/medium-20220720",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="cohere",
    #     name="cohere/medium-20221108",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="cohere",
    #     name="cohere/small-20220720",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="cohere",
    #     name="cohere/command-medium-beta",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG, INSTRUCTION_FOLLOWING_MODEL_TAG],
    # ),
    # Model(
    #     group="cohere",
    #     name="cohere/command-xlarge-beta",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, COHERE_TOKENIZER_TAG, INSTRUCTION_FOLLOWING_MODEL_TAG],
    # ),
    # # EleutherAI
    # Model(
    #     group="together",
    #     name="together/gpt-j-6b",
    #     tags=[
    #         TEXT_MODEL_TAG,
    #         FULL_FUNCTIONALITY_TEXT_MODEL_TAG,
    #         ABLATION_MODEL_TAG,
    #         GPTJ_TOKENIZER_TAG,
    #         BUGGY_TEMP_0_TAG,
    #     ],
    # ),
    # Model(
    #     group="together",
    #     name="together/gpt-neox-20b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG, GPTNEO_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="eleutherai/pythia-1b-v0",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="eleutherai/pythia-2.8b-v0",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="eleutherai/pythia-6.9b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="eleutherai/pythia-12b-v0",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # # Meta
    # Model(
    #     group="together",
    #     name="meta/llama-7b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="meta/llama-13b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="meta/llama-30b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="meta/llama-65b",
    #     # TODO(#1828): Upgrade to FULL_FUNCTIONALITY_TEXT_MODEL_TAG
    #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="meta/llama-2-7b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="meta/llama-2-13b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="meta/llama-2-70b",
    #     # TODO(#1828): Upgrade to FULL_FUNCTIONALITY_TEXT_MODEL_TAG
    #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # # Stanford
    # Model(
    #     group="together",
    #     name="stanford/alpaca-7b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, INSTRUCTION_FOLLOWING_MODEL_TAG],
    # ),
    # # LMSYS
    # Model(
    #     group="together",
    #     name="lmsys/vicuna-7b-v1.3",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, INSTRUCTION_FOLLOWING_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="lmsys/vicuna-13b-v1.3",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, INSTRUCTION_FOLLOWING_MODEL_TAG],
    # ),
    # # MosaicML
    # Model(
    #     group="together",
    #     name="mosaicml/mpt-7b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="mosaicml/mpt-instruct-7b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="mosaicml/mpt-30b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="mosaicml/mpt-instruct-30b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # # TII UAE
    # Model(
    #     group="together",
    #     name="tiiuae/falcon-7b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="tiiuae/falcon-7b-instruct",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="tiiuae/falcon-40b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="tiiuae/falcon-40b-instruct",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # # GooseAI supported models
    # Model(
    #     group="gooseai",
    #     name="gooseai/gpt-neo-20b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPTNEO_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="gooseai",
    #     name="gooseai/gpt-j-6b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPTJ_TOKENIZER_TAG],
    # ),
    # # HuggingFace
    # Model(
    #     group="huggingface",
    #     name="huggingface/gpt2",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="huggingface",
    #     name="huggingface/gpt-j-6b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPTJ_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="huggingface",
    #     name="huggingface/santacoder",
    #     tags=[CODE_MODEL_TAG],
    # ),
    # Model(
    #     group="huggingface",
    #     name="huggingface/starcoder",
    #     tags=[CODE_MODEL_TAG],
    # ),
    # # Google
    # Model(
    #     group="together",
    #     name="together/t5-11b",
    #     # Does not support echo=True
    #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG, NO_NEWLINES_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="together/flan-t5-xxl",
    #     # Does not support echo=True
    #     tags=[
    #         TEXT_MODEL_TAG,
    #         LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
    #         ABLATION_MODEL_TAG,
    #         NO_NEWLINES_TAG,
    #         INSTRUCTION_FOLLOWING_MODEL_TAG,
    #     ],
    # ),
    # Model(
    #     group="together",
    #     name="together/ul2",
    #     # Does not support echo=True
    #     tags=[
    #         TEXT_MODEL_TAG,
    #         LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
    #         ABLATION_MODEL_TAG,
    #         NO_NEWLINES_TAG,
    #         NLG_PREFIX_TAG,
    #     ],
    # ),
    # # H3 model
    # Model(
    #     group="together",
    #     name="together/h3-2.7b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    # ),
    # # OPT
    # Model(
    #     group="together",
    #     name="together/opt-175b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG, OPT_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="together/opt-66b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG, OPT_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="together/opt-6.7b",
    #     tags=[
    #         TEXT_MODEL_TAG,
    #         FULL_FUNCTIONALITY_TEXT_MODEL_TAG,
    #         ABLATION_MODEL_TAG,
    #         OPT_TOKENIZER_TAG,
    #         BUGGY_TEMP_0_TAG,
    #     ],
    # ),
    # Model(
    #     group="together",
    #     name="together/opt-1.3b",
    #     tags=[
    #         TEXT_MODEL_TAG,
    #         FULL_FUNCTIONALITY_TEXT_MODEL_TAG,
    #         ABLATION_MODEL_TAG,
    #         OPT_TOKENIZER_TAG,
    #         BUGGY_TEMP_0_TAG,
    #     ],
    # ),
    # # Microsoft/NVIDIA
    # Model(
    #     group="microsoft",
    #     name="microsoft/TNLGv2_530B",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="microsoft",
    #     name="microsoft/TNLGv2_7B",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    # ),
    # # OpenAI: https://beta.openai.com/docs/engines/gpt-3
    # Model(
    #     group="gpt3",
    #     name="openai/davinci",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="gpt3",
    #     name="openai/curie",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="gpt3",
    #     name="openai/babbage",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="gpt3",
    #     name="openai/ada",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    # ),
    # # TODO: text-davinci-002 supports insertion. Support insertion in our framework.
    # #       https://github.com/stanford-crfm/benchmarking/issues/359
    # Model(
    #     group="gpt3",
    #     name="openai/text-davinci-003",
    #     tags=[
    #         TEXT_MODEL_TAG,
    #         WIDER_CONTEXT_WINDOW_TAG,
    #         FULL_FUNCTIONALITY_TEXT_MODEL_TAG,
    #         GPT2_TOKENIZER_TAG,
    #         INSTRUCTION_FOLLOWING_MODEL_TAG,
    #     ],
    # ),
    # Model(
    #     group="gpt3",
    #     name="openai/text-davinci-002",
    #     tags=[TEXT_MODEL_TAG, WIDER_CONTEXT_WINDOW_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="gpt3",
    #     name="openai/text-davinci-001",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="gpt3",
    #     name="openai/text-curie-001",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="gpt3",
    #     name="openai/text-babbage-001",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="gpt3",
    #     name="openai/text-ada-001",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="codex",
    #     name="openai/code-davinci-002",
    #     tags=[CODE_MODEL_TAG, WIDER_CONTEXT_WINDOW_TAG, GPT2_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="codex",
    #     name="openai/code-davinci-001",
    #     tags=[CODE_MODEL_TAG, GPT2_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="codex",
    #     name="openai/code-cushman-001",
    #     tags=[CODE_MODEL_TAG, GPT2_TOKENIZER_TAG],
    # ),
    # # GPT-4
    # Model(
    #     group="gpt4",
    #     name="openai/gpt-4-0314",
    #     tags=[
    #         TEXT_MODEL_TAG,
    #         GPT4_CONTEXT_WINDOW_TAG,
    #         GPT4_TOKENIZER_TAG,
    #         OPENAI_CHATGPT_MODEL_TAG,
    #         LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
    #         INSTRUCTION_FOLLOWING_MODEL_TAG,
    #     ],
    # ),
    # Model(
    #     group="gpt4",
    #     name="openai/gpt-4-32k-0314",
    #     tags=[
    #         TEXT_MODEL_TAG,
    #         GPT4_32K_CONTEXT_WINDOW_TAG,
    #         GPT4_TOKENIZER_TAG,
    #         OPENAI_CHATGPT_MODEL_TAG,
    #         LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
    #         INSTRUCTION_FOLLOWING_MODEL_TAG,
    #     ],
    # ),
    # Model(
    #     group="gpt4",
    #     name="openai/gpt-4-0613",
    #     tags=[
    #         TEXT_MODEL_TAG,
    #         GPT4_CONTEXT_WINDOW_TAG,
    #         GPT4_TOKENIZER_TAG,
    #         OPENAI_CHATGPT_MODEL_TAG,
    #         LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
    #         INSTRUCTION_FOLLOWING_MODEL_TAG,
    #     ],
    # ),
    # Model(
    #     group="gpt4",
    #     name="openai/gpt-4-32k-0613",
    #     tags=[
    #         TEXT_MODEL_TAG,
    #         GPT4_32K_CONTEXT_WINDOW_TAG,
    #         GPT4_TOKENIZER_TAG,
    #         OPENAI_CHATGPT_MODEL_TAG,
    #         LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
    #         INSTRUCTION_FOLLOWING_MODEL_TAG,
    #     ],
    # ),
    # # ChatGPT: https://openai.com/blog/chatgpt
    # Model(
    #     group="gpt3",
    #     name="openai/gpt-3.5-turbo-0301",
    #     # The claimed sequence length is 4096, but as of 2023-03-07, the empirical usable
    #     # sequence length is smaller at 4087 with one user input message and one assistant
    #     # output message because ChatGPT uses special tokens for message roles and boundaries.
    #     # We use a rounded-down sequence length of 4000 to account for these special tokens.
    #     tags=[
    #         TEXT_MODEL_TAG,
    #         GPT_TURBO_CONTEXT_WINDOW_TAG,
    #         GPT4_TOKENIZER_TAG,
    #         OPENAI_CHATGPT_MODEL_TAG,
    #         LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
    #         INSTRUCTION_FOLLOWING_MODEL_TAG,
    #     ],
    # ),
    # Model(
    #     group="gpt3",
    #     name="openai/gpt-3.5-turbo-0613",
    #     # The claimed sequence length is 4096, but as of 2023-03-07, the empirical usable
    #     # sequence length is smaller at 4087 with one user input message and one assistant
    #     # output message because ChatGPT uses special tokens for message roles and boundaries.
    #     # We use a rounded-down sequence length of 4000 to account for these special tokens.
    #     tags=[
    #         TEXT_MODEL_TAG,
    #         GPT_TURBO_CONTEXT_WINDOW_TAG,
    #         GPT4_TOKENIZER_TAG,
    #         OPENAI_CHATGPT_MODEL_TAG,
    #         LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
    #         INSTRUCTION_FOLLOWING_MODEL_TAG,
    #     ],
    # ),
    # Model(
    #     group="gpt3",
    #     name="openai/gpt-3.5-turbo-16k-0613",
    #     # Claimed length is 16,384; we round down to 16,000 for the same reasons as explained
    #     # in the openai/gpt-3.5-turbo-0613 comment
    #     tags=[
    #         TEXT_MODEL_TAG,
    #         GPT_TURBO_16K_CONTEXT_WINDOW_TAG,
    #         GPT4_TOKENIZER_TAG,
    #         OPENAI_CHATGPT_MODEL_TAG,
    #         LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
    #         INSTRUCTION_FOLLOWING_MODEL_TAG,
    #     ],
    # ),
    # # OpenAI similarity embedding models: https://beta.openai.com/docs/guides/embeddings
    # Model(
    #     group="gpt3",
    #     name="openai/text-similarity-davinci-001",
    #     tags=[EMBEDDING_MODEL_TAG],
    # ),
    # Model(
    #     group="gpt3",
    #     name="openai/text-similarity-curie-001",
    #     tags=[EMBEDDING_MODEL_TAG],
    # ),
    # Model(
    #     group="gpt3",
    #     name="openai/text-similarity-babbage-001",
    #     tags=[EMBEDDING_MODEL_TAG],
    # ),
    # Model(
    #     group="gpt3",
    #     name="openai/text-similarity-ada-001",
    #     tags=[EMBEDDING_MODEL_TAG],
    # ),
    # Model(
    #     group="gpt3",
    #     name="openai/text-embedding-ada-002",
    #     tags=[EMBEDDING_MODEL_TAG],
    # ),
    # # Together
    # Model(
    #     group="together",
    #     name="together/gpt-jt-6b-v1",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPTJ_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="together/gpt-neoxt-chat-base-20b",
    #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG, CHATML_MODEL_TAG, GPTNEO_TOKENIZER_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="together/redpajama-incite-base-3b-v1",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="together/redpajama-incite-instruct-3b-v1",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="together/redpajama-incite-base-7b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="together/redpajama-incite-instruct-7b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # # Tsinghua
    # Model(
    #     group="together",
    #     name="together/glm",
    #     # Inference with echo=True is not feasible -- in the prompt encoding phase, they use
    #     # bidirectional attention and do not perform predictions on them.
    #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG, NO_NEWLINES_TAG],
    # ),
    # # Writer
    # Model(
    #     group="palmyra",
    #     name="writer/palmyra-base",
    #     # Does not support echo
    #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="palmyra",
    #     name="writer/palmyra-large",
    #     # Does not support echo
    #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="palmyra",
    #     name="writer/palmyra-r",
    #     # Does not support echo
    #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="palmyra",
    #     name="writer/camel",
    #     # Does not support echo
    #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="palmyra",
    #     name="writer/palmyra-instruct-30",
    #     # Does not support echo
    #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="palmyra",
    #     name="writer/palmyra-e",
    #     # Does not support echo
    #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="palmyra",
    #     name="writer/silk-road",
    #     # Does not support echo
    #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="palmyra",
    #     name="writer/palmyra-x",
    #     # Does not support echo
    #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # # Yandex
    # Model(
    #     group="together",
    #     name="together/yalm",
    #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG, ABLATION_MODEL_TAG],
    # ),
    # # Google
    # Model(
    #     group="google",
    #     name="google/palm",
    #     tags=[TEXT_MODEL_TAG, LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # # NVIDIA
    # Model(
    #     group="nvidia",
    #     name="nvidia/megatron-gpt2",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG, BUGGY_TEMP_0_TAG],
    # ),
    # # Databricks
    # Model(
    #     group="together",
    #     name="databricks/dolly-v2-3b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="databricks/dolly-v2-7b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="databricks/dolly-v2-12b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # # Stability AI
    # Model(
    #     group="together",
    #     name="stabilityai/stablelm-base-alpha-3b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="together",
    #     name="stabilityai/stablelm-base-alpha-7b",
    #     tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    # ),
    # Model(
    #     group="lightningai",
    #     name="lightningai/lit-gpt",
    #     tags=[
    #         TEXT_MODEL_TAG,
    #         INSTRUCTION_FOLLOWING_MODEL_TAG,
    #         LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG,
    #         GPT2_TOKENIZER_TAG,
    #     ],
    # ),
    # # For debugging
    # Model(
    #     group="simple",
    #     name="simple/model1",
    # ),
]
