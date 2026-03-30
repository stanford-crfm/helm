from helm.benchmark.model_deployment_registry import ClientSpec, ModelDeployment, model_deployment_generator


@model_deployment_generator("litellm")
def get_litellm_model_deployment(name: str):
    name_parts = name.split("/")
    model_name = "/".join(name_parts[-2:])
    return ModelDeployment(
        name=name,
        model_name=model_name,
        client_spec=ClientSpec(
            "helm.clients.litellm_client.LiteLLMCompletionClient", args={"litellm_model": "/".join(name_parts[1:])}
        ),
    )
