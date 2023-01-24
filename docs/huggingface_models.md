# Using HuggingFace Models

HELM can be used to evaluate models on [Hugging Face Model Hub](https://huggingface.co/models).

To use models from Hugging Face Model Hub, add the Hugging Face model IDs to the `--register-huggingface-models` flags to `helm-run`. This will make the corresponding Hugging Face models available to use in your run spec descriptions. In the run spec description, use the Hugging Face model ID as the model name.

To use revision of a model other than the default main revision, append a `#` followed by the revision name to the model ID passed to the `--register-huggingface-models` flag. Note that in the run spec description, the model name should not include `#` or the revision name.

Restrictions: `--register-huggingface-models` currently does not support models without a namespace (e.g. `bert-base-uncased`).

Examples:

```bash
# Run boolq on stanford-crfm/alias-gpt2-small-x21 at the default main revision
helm-run \
    --run-specs boolq:model=stanford-crfm/alias-gpt2-small-x21 \
    --register-huggingface-models stanford-crfm/alias-gpt2-small-x21 \
    --local \
    --suite v1 \
    --max-eval-instances 10

# Run boolq on stanford-crfm/alias-gpt2-small-x21 at revision checkpoint-400000
helm-run \
    --run-specs boolq:model=stanford-crfm/alias-gpt2-small-x21 \
    --register-huggingface-models stanford-crfm/alias-gpt2-small-x21#checkpoint-400000 \
    --local \
    --suite v1 \
    --max-eval-instances 10
```
