from dataclasses import dataclass


@dataclass
class Model:
    """Represents a model that we can make requests too."""

    group: str
    name: str
    description: str


# For the list of available models, see the following docs:
#
# OpenAI: https://help.openai.com/en/articles/5832130-what-s-changed-with-engine-names-and-best-practices
# AI21: https://studio.ai21.com/docs/jurassic1-language-models/
ALL_MODELS = [
    # AI21
    Model(group="jurassic", name="ai21/j1-jumbo", description="Jurassic J1-Jumbo (178B parameters)"),
    Model(group="jurassic", name="ai21/j1-large", description="Jurassic J1-Large (7.5B parameters)"),
    # OpenAI
    Model(group="gpt3", name="openai/davinci", description="GPT-3 (175B parameters)"),
    Model(group="gpt3", name="openai/curie", description="GPT-3 (6.7B parameters)"),
    Model(group="gpt3", name="openai/babbage", description="GPT-3 (1.3B parameters)"),
    Model(group="gpt3", name="openai/ada", description="GPT-3 (350M parameters)"),
    Model(group="gpt3", name="openai/text-davinci-001", description="GPT-3 from Instruct series (175B parameters)"),
    Model(group="gpt3", name="openai/text-curie-001", description="GPT-3 from Instruct series (6.7B parameters)"),
    Model(group="gpt3", name="openai/text-babbage-001", description="GPT-3 from Instruct series (1.3B parameters)"),
    Model(group="gpt3", name="openai/text-ada-001", description="GPT-3 from Instruct series (350M parameters)"),
    Model(
        group="codex",
        name="openai/code-davinci-001",
        description="Codex (for natural language to code) - 4096 max tokens",
    ),
    Model(
        group="codex",
        name="openai/code-cushman-001",
        description="Codex (for natural language to code) - 2048 max tokens.",
    ),
    # HuggingFace
    Model(group="eleuther", name="huggingface/gptj_6b", description="GPT-J (6B parameters)"),
    Model(group="gpt2", name="huggingface/gpt2", description="GPT-2 (1.5B parameters)"),
    # For debugging
    Model(group="simple", name="simple/model1", description="Copy last tokens (for debugging)"),
]


def get_model_group(name: str):
    for model in ALL_MODELS:
        if model.name == name:
            return model.group
    raise Exception(f"No model with name {name}")
