from helm.benchmark.model_deployment_registry import ClientSpec, ModelDeployment, model_deployment_generator
from helm.common.hierarchical_logger import hwarn


@model_deployment_generator("huggingface")
def get_huggingface_model_deployment(name: str):
    from helm.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer

    name_parts = name.split("/")
    model_name = "/".join(name_parts[-2:])
    pretrained_model_name_or_path = "/".join(name_parts[1:])
    with HuggingFaceTokenizer.create_tokenizer(pretrained_model_name_or_path) as tokenizer:
        max_sequence_length = tokenizer.model_max_length
        if max_sequence_length > 1_000_000_000:
            hwarn(
                f"Hugging Face model {pretrained_model_name_or_path} does not have a configured model_max_length; "
                "input truncation may not work correctly; errors may result from exceeding the model's max length"
            )
    return ModelDeployment(
        name=name,
        model_name=model_name,
        client_spec=ClientSpec(
            "helm.clients.huggingface_client.HuggingFaceClient",
            args={"pretrained_model_name_or_path": pretrained_model_name_or_path},
        ),
        tokenizer_name=name,
        max_sequence_length=max_sequence_length,
    )
