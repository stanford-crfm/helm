# Hugging Face Model Hub Integration

HELM can be used to evaluate `AutoModelForCausalLM` models (e.g. [`BioMedLM`](https://huggingface.co/stanford-crfm/BioMedLM)) on [Hugging Face Model Hub](https://huggingface.co/models) or local disk. Note that only `AutoModelForCausalLM` models are supported; other classes such as `AutoModelForSeq2SeqLM` may be supported in the future.

## Using `model_deployments.yaml`

You can add Hugging Face models using the method discussed in [Adding New Models](adding_new_models.md). This can be used for both models from Hugging Face Hub and from local disk. Please refer to the page for instructions for how to do so.

## Using command-line flags

In some cases, you can use command-line flags with `helm-run` to evaluating Hugging Face models. This provides a more convenient way to use Hugging Face models that does not require configuration files.

To use `AutoModelForCausalLM` models from Hugging Face Model Hub, add the Hugging Face model IDs to the `--enable-huggingface-models` flags to `helm-run`. This will make the corresponding Hugging Face models available to use in your run spec descriptions. In the run spec description, use the Hugging Face model ID as the model name.

To use a revision of a model other than the default main revision, append a `@` followed by the revision name to the model ID passed to the `--enable-huggingface-models` flag.

Current restrictions with command-line flags:

- Models without a namespace are not supported (e.g. `bert-base-uncased`).
- The model must have `model_max_length` set in the tokenizer configuration.

Example model from Hugging Face Hub:

```bash
# Run boolq on stanford-crfm/BioMedLM at the default main revision
helm-run \
    --run-entries boolq:model=stanford-crfm/BioMedLM \
    --enable-huggingface-models stanford-crfm/BioMedLM \
    --suite v1 \
    --max-eval-instances 10

# Run boolq on stanford-crfm/BioMedLM at revision main
helm-run \
    --run-entries boolq:model=stanford-crfm/BioMedLM@main \
    --enable-huggingface-models stanford-crfm/BioMedLM@main \
    --suite v1 \
    --max-eval-instances 10
```

Example model from local disk:

```bash
# Run boolq on stanford-crfm/BioMedLM at the default main revision
helm-run \
    --run-entries boolq:model=your-org/your-model \
    --enable-local-huggingface-models path/to/your-org/your-model \
    --suite v1 \
    --max-eval-instances 10
```

To use Optimum Intel, add `--openvino` flag to `helm-run`. Optimum Intel provides a simple interface to optimize Transformer models and convert them to OpenVINO™ Intermediate Representation format to accelerate end-to-end pipelines on Intel® architectures using OpenVINO™ runtime. It runs the model on the CPU.

Examples:

```bash
# Run boolq on stanford-crfm/BioMedLM optimized by Optimum Intel OpenNIVO
helm-run \
    --run-entries boolq:model=stanford-crfm/BioMedLM \
    --enable-huggingface-models stanford-crfm/BioMedLM \
    --suite v1 \
    --max-eval-instances 10 \
    --openvino 
```
