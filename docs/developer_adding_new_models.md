# Adding New Clients

**Warning** &mdash; The document is stale. The information below may be outdated and incorrect. Please proceed with caution!

## Overview of the process
To add a new model you need to define 3 objects:
* a `ModelMetadata` objects that defines properties of your model (name, metadata, capabilities, ...).
* one or several `ModelDeployment` which defines how to query a model (mainly by providing a `Client`, a `WindowService` and a `Tokenizer`). You can define several deployments for a single model (`local/your-model`, `huggingface/your-model`, `together/your-model`, ...).
* a `TokenizerConfig` which defines how to build the `Tokenizer` (mainly by providing a `TokenizerSpec`).

In some cases you might have to define additionally:
* a `Client` if your query method differ from any clients we have implemented. This will be then referenced in the `ModelDeployment`. We recommend checking `HTTPModelClient` and `HuggingFaceClient` which can be used in a lot of cases. If you identify the need for a new client, a good starting to point is to have a look at `SimpleClient`.
* a `WindowService`. First have a look at `DefaultWindowService` to check if this is not enough for your use case. If you need you own `truncate_from_right` function, then you might need to create your own `WindowService`. In that case, a good starting point is to have a look at `YaLMWindowService`.


## Where to create the objects
There are two cases: private models that should only be accessible to you and models not yet supported by HELM but that would benefit everyone if added.

In the first case, you should create the files `model_deployments.yaml`, `model_metadata.yaml` and `tokenizer_configs.yaml` in `prod_env/` (A folder that you should create at the root of the repo if not already done). HELM will automatically registed any model defined in these files without any change in the code while ignoring them on Github which can be convenient for you. Then you can simply duplicate the corresponding files from `src/helm/config`, delete the models and add yours. Follow the next section for an example.

In the second case, if you want to add a model to HELM, you can directly do it in `src/helm/config`. You can then open a Pull Request on Github to share the model. When you do, make sure to:
* Include any link justifying the metadata used in `ModelMetadata` such as the release data, number of parameters, capabilities and so on (you should not infer anything).
* Check that you are respecting the format used in those files (`ModelMetadata` should be named as `<CREATOR-ORGANIZATION>/<MODEL-NAME>` and the `ModelDeployment` should be named as `<HOST-ORGANIZATION>/<MODEL-NAME>`, for example `ModelMetadata`: `openai/gpt2` and `ModelDeployment`: `huggingface/gpt2`). Add the appropriate comments and so on.
* Run `helm-run --run-entries "mmlu:subject=anatomy,model_deployment=<YOUR-DEPLOYMENT>" --suite v1 --max-eval-instances 10` and make sure that everything works. Include the logs from the terminal in your PR.
* Not create unnecessary objects (`Client` `TokenizerCOnfig`, `WindowService`) and if you have to create one of these objects, document in your PR why you had to. Make them general enough so that they could be re-used by other models (especially the `Client`).


## Example

In `src/helm/config/model_metadata.yaml`:
```yaml
# [...]

models:

  - name: simple/model1
    [...]

  # NEW MODEL STARTS HERE
  - name: simple/tutorial
    display_name: Tutorial Model
    description: This is a simple model used in the tutorial.
    creator_organization_name: Helm
    access: open
    release_date: 2023-01-01
    tags: [TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG]

  [...]
```

In `src/helm/config/model_deployments.yaml`:
```yaml
# [...]

model_deployments:

  - name: simple/model1
    [...]

  - name: simple/tutorial
    model_name: simple/tutorial
    tokenizer_name: simple/model1
    max_sequence_length: 2048
    client_spec:
      class_name: "helm.clients.simple_client.SimpleClient"
      args: {}
    window_service_spec:
      class_name: "helm.benchmark.window_services.openai_window_service.OpenAIWindowService"
      args: {}

  [...]
```

We won't be adding any `TokenizerConfig` here as we are reusing `simple/model1`. This shows a good practice when adding a new model, always check if the correct tokenizer does not already exists.

You should now be able to run `helm-run --run-entries "mmlu:subject=anatomy,model_deployment=simple/tutorial" --suite v1 --max-eval-instances 10` without any error.


