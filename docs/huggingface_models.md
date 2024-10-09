# Hugging Face Model Hub Integration

HELM can be used to evaluate `AutoModelForCausalLM` models (e.g. [`BioMedLM`](https://huggingface.co/stanford-crfm/BioMedLM)) on [Hugging Face Model Hub](https://huggingface.co/models).

To use `AutoModelForCausalLM` models from Hugging Face Model Hub, add the Hugging Face model IDs to the `--enable-huggingface-models` flags to `helm-run`. This will make the corresponding Hugging Face models available to use in your run spec descriptions. In the run spec description, use the Hugging Face model ID as the model name.

To use a revision of a model other than the default main revision, append a `@` followed by the revision name to the model ID passed to the `--enable-huggingface-models` flag.

Current restrictions:

- Only `AutoModelForCausalLM` is supported; other classes such as `AutoModelForSeq2SeqLM` may be supported in the future.
- Models without a namespace are not supported (e.g. `bert-base-uncased`).
- Models at local file paths are not supported.

Examples:

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
