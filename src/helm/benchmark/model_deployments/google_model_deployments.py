from helm.benchmark.model_deployment_registry import ClientSpec, ModelDeployment, model_deployment_generator


@model_deployment_generator("google")
def get_google_model_deployment(name: str):
    name_parts = name.split("/")
    assert len(name_parts) >= 2
    model_name = "/".join(name_parts[-2:])
    genai_model_name = name_parts[-1] if name_parts[-2] == "google" else model_name

    return ModelDeployment(
        name=name,
        model_name=model_name,
        client_spec=ClientSpec("helm.clients.google_genai_client.GoogleGenAIClient", {"genai_model": genai_model_name}),
    )


@model_deployment_generator("gemini")
def get_gemini_model_deployment(name: str):
    name_parts = name.split("/")
    model_name = "/".join(name_parts[-2:])
    assert len(name_parts) >= 2
    if len(name_parts) < 3 or name_parts[-2] != "google":
        raise ValueError("Model name must be in the format gemini/google/model")
    genai_model_name = name_parts[-1]

    return ModelDeployment(
        name=name,
        model_name=model_name,
        client_spec=ClientSpec(
            "helm.clients.google_genai_client.GoogleGenAIClient",
            {
                "genai_model": genai_model_name,
                "genai_use_vertexai": False,
            },
        ),
    )


@model_deployment_generator("vertexai")
def get_vertexai_model_deployment(name: str):
    name_parts = name.split("/")
    model_name = "/".join(name_parts[-2:])
    assert len(name_parts) >= 2
    if len(name_parts) < 3:
        raise ValueError("Model name must be in the format vertexai/publisher/model")
    genai_model_name = name_parts[-1] if name_parts[-2] == "google" else "/".join(name_parts[-2:])
    return ModelDeployment(
        name=name,
        model_name=model_name,
        client_spec=ClientSpec(
            "helm.clients.google_genai_client.GoogleGenAIClient",
            {
                "genai_model": genai_model_name,
                "genai_use_vertexai": True,
            },
        ),
    )
