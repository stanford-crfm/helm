from helm.benchmark.tokenizer_config_registry import TokenizerConfig, TokenizerSpec, tokenizer_config_generator


@tokenizer_config_generator("huggingface")
def get_huggingface_tokenizer_config(name: str):
    from helm.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer

    name_parts = name.split("/")
    pretrained_model_name_or_path = "/".join(name_parts[1:])
    with HuggingFaceTokenizer.create_tokenizer(pretrained_model_name_or_path) as tokenizer:
        end_of_text_token = tokenizer.eos_token or ""
        prefix_token = tokenizer.bos_token or ""
    return TokenizerConfig(
        name=name,
        tokenizer_spec=TokenizerSpec(
            "helm.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer",
            args={"pretrained_model_name_or_path": pretrained_model_name_or_path},
        ),
        end_of_text_token=end_of_text_token,
        prefix_token=prefix_token,
    )
