from helm.benchmark.model_deployment_registry import ClientSpec, ModelDeployment, model_deployment_generator


@model_deployment_generator("openrouter")
def get_openrouter_model_deployment(name: str):
    name_parts = name.split("/")
    model_name = "/".join(name_parts[-2:])
    return ModelDeployment(
        name=name,
        model_name=model_name,
        client_spec=ClientSpec(
            "helm.clients.openrouter_client.OpenRouterClient",
            args={
                "openrouter_model_name": model_name,
            },
        ),
    )
