from dataclasses import dataclass


@dataclass
class Model:
    """Represents a model that we can make requests too."""

    group: str
    name: str
    description: str


all_models = [
    Model(group="gpt3", name="openai/davinci", description="GPT-3 (175B parameters)"),
    Model(group="gpt3", name="openai/curie", description="GPT-3 (6.7B parameters)"),
    Model(group="gpt3", name="openai/babbage", description="GPT-3 (1.3B parameters)"),
    Model(group="gpt3", name="openai/ada", description="GPT-3 (350M parameters)"),
    Model(group="gpt3", name="openai/instruct-davinci", description="GPT-3 Instruct series"),
    Model(
        group="codex",
        name="openai/davinci-codex",
        description="Codex (for natural language to code) - 4096 max tokens",
    ),
    Model(
        group="codex",
        name="openai/cushman-codex",
        description="Codex (for natural language to code) - 2048 max tokens",
    ),
    Model(group="jurassic", name="ai21/j1-jumbo", description="Jurassic J1-Jumbo (178B parameters)"),
    Model(group="jurassic", name="ai21/j1-large", description="Jurassic J1-Large (7.5B parameters)"),
    Model(group="simple", name="simple/model1", description="Copy last tokens (for debugging)"),
]


def get_model_group(name: str):
    for model in all_models:
        if model.name == name:
            return model.group
    raise Exception(f"No model with name {name}")
