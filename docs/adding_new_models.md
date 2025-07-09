# Adding New Models

HELM comes with more than a hundred built-in models. If you want to run a HELM evaluation on a model that is not built-in, you can configure HELM to add your own model. This also allows you to evaluate private models that are not publicly accessible, such as a model checkpoint on local disk, or a model server on a private network

HELM comes with many built-in `Client` classes (i.e. model API clients) and `Tokenizer` clients. If there is already an existing `Client` and `Tokenizer` class for your use case, you can simply add it to your local configuration. You would only need to implement a new class if you are adding a model with a API format or inference platform that is currently not supported by HELM.

If you wish to evaluate a model not covered by an existing `Client` and `Tokenizer`, you can implement your own `Client` and `Tokenizer` subclasses. Instructions for adding custom `Client` and `Tokenizer` subclasses will be added to the documentation in the future.

## Adding a Model Locally

### Model Metadata

Create a local model metadata configuration file if it does not already exist. The file should be a `prod_env/model_metadata.yaml` by default, or at `$LOCAL_PATH/model_metadata.yaml` if `--local-path` is set where `$LOCAL_FOLDER` is the value of the flag.

This file should contain a YAML-formatted `ModelMetadataList` object. For an example of this format, refer to `model_metadata.yaml` in the GitHub repository, or follow the example below:

```yaml
models:
  - name: eleutherai/pythia-70m
    display_name: Pythia (70M)
    description: Pythia (70M parameters). The Pythia project combines interpretability analysis and scaling laws to understand how knowledge develops and evolves during training in autoregressive transformers.
    creator_organization_name: EleutherAI
    access: open
    num_parameters: 95600000
    release_date: 2023-02-13
    tags: [TEXT_MODEL_TAG, PARTIAL_FUNCTIONALITY_TEXT_MODEL_TAG]
```

### Model Deployment

A model deployment defines the actual implementation of the model. The model deployment configuration tells HELM how to generate outputs from the model model by running local inference or or sending requests to an API. Every model should have at least one model deployment. However, since there are sometimes multiple implementations or inference platform providers for the same model, a model can have more than one model deployment. For instance, the model `google/gemma-2-9b-it` has the model deployments `together/gemma-2-9b-it` (remote inference using Together AI's API) and `google/gemma-2-9b-it` (local inference with Hugging Face).

Create a local model deployments configuration file if it does not already exist. The file should be a `prod_env/model_metadata.yaml` by default, or at `$LOCAL_PATH/model_metadata.yaml` if `--local-path` is set where `$LOCAL_FOLDER` is the value of the flag.

This file should contain a YAML-formatted `ModelDeployments` object. For an example of this format, refer to `model_deployments.yaml` in the GitHub repository, or follow an example below for your preferred model platform.

Note that the model deployment name will frequently differ from the model name. The model deployment name should be `$HOST_ORGANIZATON/$MODEL_NAME`, while the model name should be `$CREATOR_ORGANIZATON/$MODEL_NAME`.

### Hugging Face

Example:

```yaml
model_deployments:
  - name: huggingface/pythia-70m
    model_name: eleutherai/pythia-70m
    tokenizer_name: EleutherAI/gpt-neox-20b
    max_sequence_length: 2048
    client_spec:
      class_name: "helm.clients.huggingface_client.HuggingFaceClient"
      args:
        pretrained_model_name_or_path: EleutherAI/pythia-70m
```

Note: If `pretrained_model_name_or_path` is omitted, the model will be loaded from Hugging Face Hub using `model_name` (_not_ `name`) by default.

Examples of common arguments within `args`:

- Loading from local disk: `pretrained_model_name_or_path: /path/to/my/model`
- Revision: `revision: my_revision`
- Quantization: `load_in_8bit: true`
- Model precision: `torch_dtype: torch.float16`
- Model device: `device: cpu` or `device: cuda:0`
- Allow running remote code: `trust_remote_code: true`
- Multi-GPU: `device_map: auto`


Notes:

- This uses local inference with Hugging Face. It will attempt to use GPU inference if available, and use CPU inference otherwise.
- Multi-GPU inference can be enabled by setting `device_map: auto` in the `args`.
- GPU models loaded by `helm-run` will remain loaded on the GPU for the lifespan of `helm-run`.
- If evaluating multiple models, it is prudent to evaluate each model with a separate `helm-run` invocation.
- If you are attempting to access models that are private, restricted, or require signing an agreement (e.g. Llama 3), you need to be authenticated to Hugging Face through the CLI. As the user that will be running `helm-run`, run `huggingface-cli login` in your shell. Refer to [Hugging Face's documentation](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command) for more information.

### vLLM

```yaml
model_deployments:
  - name: vllm/pythia-70m
    model_name: eleutherai/pythia-70m
    tokenizer_name: EleutherAI/gpt-neox-20b
    max_sequence_length: 2048
    client_spec:
      class_name: "helm.clients.vllm_client.VLLMClient"
      args:
        base_url:  http://mymodelserver:8000/v1/
```

For non-chat models, set `class_name` in `client_spec` to `helm.clients.vllm_client.VLLMClient`. For chat models, set `class_name` in `client_spec` to `helm.clients.vllm_client.VLLMChatClient`.

Set `base_url` to the URL of your inference server. On your inference server, run vLLM's OpenAI compatible server with:

```
python -m vllm.entrypoints.openai.api_server --model EleutherAI/pythia-70m
```

### Together AI

```yaml
model_deployments:
  - name: together/gemma-2-9b-it
    model_name: google/gemma-2-9b-it
    tokenizer_name: google/gemma-2-9b
    max_sequence_length: 8191
    client_spec:
      class_name: "helm.clients.together_client.TogetherClient"
      args:
        together_model: google/gemma-2-9b-it
```

Notes:

- You will need to add Together AI credentials to your credentials file e.g. add `togetherApiKey: your-api-key` to `./prod_env/credentials.conf`.
- If `together_model` is omitted, the Together model with `model_name` (_not_ `name`) will be used by default.
- This above model may not be currently available on Together AI. Consult [Together AI's Inference Models documentation](https://docs.together.ai/docs/inference-models) for a list of currently available models and corresponding model strings.

## Testing New Models

After you've added your model, you can run your model with `helm-run` using a run entry such as `mmlu:subject=anatomy,model=your-org/your-model`. It is also recommended to use the `--disable-cache` flag so that in the event that you made a mistake, the incorrect requests are not written to the request cache. Example:

```sh
helm-run --run-entry mmlu:subject=anatomy,model=your-org/your-model --suite my-suite --max-eval-instances 10 --disable-cache

helm-summarize --suite my-suite

helm-server
```

## Adding New Models to HELM

If your model is publicly accessible, you may want to add it to the HELM itself so that all HELM users may use the model. This should only be done only if the model may be easily accessible by other users.

To do so, simply add your new model metadata and model deployments to the respective configuration files in the HELM repository at `src/helm/config/`, rather than the local config files, and then open a pull request on GitHub. If you already added your model to your local configuration files at `prod_env/`, you should move those changes to the corresponding configuration files in `src/helm/config/` - do not add the model to both `src/helm/config/` and `prod_env/` simulatenously.

Test the changes using the same procedure above, and then open a pull request on HELM GitHub repository.
