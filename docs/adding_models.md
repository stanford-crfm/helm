# Adding Models

HELM comes with more than a hundred built-in models. If you want to run a HELM evaluation on a model that is not built-in, you can configure HELM to add your own model. This also allows you to evaluate private models that not publicly accessible, such as a model checkpoint on local disk, or a model server on a private network

HELM comes with many built-in `Client` classes (i.e. model API clients) and `Tokenizer` clients. If there is already an existing `Client` and `Tokenizer` class for your use case, you can simply add it to your local configuration.

If you wish to evaluate a model not covered by an existing `Client` and `Tokenizer`, you can implement your own `Client` and `Tokenizer` subclasses. Instructions for adding custom `Client` and `Tokenizer` subclasses will be added to the documentation in the future.

## Creating a model deployments configuration

To create a local model deployments configuration file, determine the location of your local configuration folder. If you are using the `--local-path` flag with `helm-run`, the specified folder of the flag is the local configuration folder. Otherwise, the local configuration folder is `./prod_env/` under your current working directory by default.

Create a file called `model_deployments.yaml` underneath that directory (e.g. `./prod_env/model_deployments.yaml`).

This file should contain a YAML-formatted `ModelDeployments` object. For an example of this format, refer to the built-in `model_deployments.yaml` in the GitHub repository, or follow the example below for your preferred model platform.

## Model platform integrations

### Hugging Face

Example:

```yaml
model_deployments:

  - name: vllm/pythia-70m
    model_name: eleutherai/pythia-70m
    tokenizer_name: EleutherAI/gpt-neox-20b
    max_sequence_length: 2048
    client_spec:
      class_name: "helm.clients.huggingface_client.HuggingFaceClient"
      args:
        pretrained_model_name_or_path: EleutherAI/pythia-70m
```

Note: If `pretrained_model_name_or_path` is omitted, the model will be loaded from Hugging Face Hub using `model_name` (_not_ `name`) by default.

Examples of common arguments:

- Loading from local disk: `pretrained_model_name_or_path: /path/to/my/model`
- Revision: `revision: my_revision`
- Quantization: `load_in_8bit: true`
- Model precision: `torch_dtype: torch.float16`

Then run:

```
helm-run --run-entries mmlu:subject=anatomy,model=eleutherai/pythia-70m --suite my_suite --max-eval-instances 5
```

Note: This uses Hugging Face local inference. It will attempt to use GPU inference if available, and use CPU inference otherwise. It is only able to use the first GPU. Multi-GPU inference is not supported. Every model needed by `helm-run` will be loaded on the same GPU - if evaluating multiple models, it is prudent to evaluate each model with a separate `helm-run` invocation.

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

Set `base_url` to the URL of your inference server. On your inference server, run vLLM's OpenAI compatible server with:

```
python -m vllm.entrypoints.openai.api_server --model EleutherAI/pythia-70m
```

Then run:

```
helm-run --run-entries mmlu:subject=anatomy,model=eleutherai/pythia-70m --suite my_suite --max-eval-instances 5
```

### Together AI

```yaml
model_deployments:
  - name: together/llama-2-7b
    model_name: meta/llama-2-7b
    tokenizer_name: meta-llama/Llama-2-7b-hf
    max_sequence_length: 4094
    client_spec:
      class_name: "helm.clients.together_client.TogetherClient"
      args:
        together_model: togethercomputer/llama-2-7b
```

Then run:

```
helm-run --run-entries mmlu:subject=anatomy,model=meta/llama-2-7b --suite my_suite --max-eval-instances 5
```

Note: If `together_model` is omitted, the Together model with `model_name` (_not_ `name`) will be used by default.

Note: This model may not be currently available on Together AI. Consult [Together AI's Inference Models documentation](https://docs.together.ai/docs/inference-models) for a list of currently available models and corresponding model strings.
