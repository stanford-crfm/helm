# Custom Tokenizers

HELM comes with many built-in tokenizers, but in some cases, you may need to add your own custom tokenizer for your custom model.

## Creating a tokenizer configuration file

Create a file called `tokenizer_configs.yaml` in your local configuration folder (e.g. `./prod_env/tokenizer_configs.yaml`).

This file should contain a YAML-formatted `TokenizerConfigs` object. For an example of this format, refer to the built-in `tokenizer_configs.yaml` in the GitHub repository, or follow the example below for your preferred model platform.

After adding a tokenizer configuration, you can then use the tokenizer in your custom model deployments by setting the specifying the tokenizer name in the `tokenizer` field of the model deployment.

### Hugging Face tokenizers

To add a Hugging Face tokenizer, follow the format below, setting `name` to Hugging Face hub model ID.

```yaml
tokenizer_configs:
  - name: bigscience/bloom
    tokenizer_spec:
      class_name: "helm.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer"
      args:
        pretrained_model_name_or_path: bigscience/bloom
    end_of_text_token: "<s>"
    prefix_token: "</s>"
```

Note that `pretrained_model_name_or_path` can also be set to a path to load a Hugging Face tokenizer from local disk.

If `pretrained_model_name_or_path` (or `args`) is omitted, the model will be loaded from Hugging Face Hub using `name` as the model ID by default. For example:

```yaml
tokenizer_configs:
  - name: bigscience/bloom
    tokenizer_spec:
      class_name: "helm.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer"
    end_of_text_token: "<s>"
    prefix_token: "</s>"
```

To find the values for `end_of_text_token` and `prefix_token`, you can run the following Python code snippet below (replacing `bigscience/bloom` with the Hugging Face Hub model ID). If any special token is unknown, it should be set to the empty string `""`.

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
print(f'end_of_text_token: "{tokenizer.eos_token}"\nprefix_token: "{tokenizer.bos_token}"')
```

HELM does not auto-infer special token information because some tokenizers on Hugging Face Model Hub may have incorrect or missing special token values. Therefore, you must manually set these values and verify that they are correct.
