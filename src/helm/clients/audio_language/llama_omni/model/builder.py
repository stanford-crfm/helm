import os

from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch
from helm.clients.audio_language.llama_omni.model.language_model.omni_speech_llama import OmniSpeechLlamaForCausalLM
from helm.clients.audio_language.llama_omni.model.language_model.omni_speech2s_llama import OmniSpeech2SLlamaForCausalLM
from helm.clients.audio_language.llama_omni.model.speech_encoder.builder import build_speech_encoder


def load_pretrained_model(
    model_path,
    model_base,
    is_lora=False,
    s2s=False,
    load_8bit=False,
    load_4bit=False,
    device="cuda",
    use_flash_attn=False,
    **kwargs
):
    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    model_cls = OmniSpeech2SLlamaForCausalLM if s2s else OmniSpeechLlamaForCausalLM

    # Load OmniSpeech model
    if is_lora:
        assert model_base is not None, "model_base is required for LoRA models."
        from language_model.omni_speech_llama import OmniSpeechConfig

        lora_cfg_pretrained = OmniSpeechConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        print("Loading OmniSpeech from base model...")
        model = model_cls.from_pretrained(model_base, low_cpu_mem_usage=False, config=lora_cfg_pretrained, **kwargs)
        print("Loading additional OmniSpeech weights...")
        if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
            non_lora_trainables = torch.load(os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu")
        non_lora_trainables = {
            (k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()
        }
        if any(k.startswith("model.model.") for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel

        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(model, model_path)
        print("Merging LoRA weights...")
        model = model.merge_and_unload()
        print("Model is loaded...")
    elif model_base is not None:
        print("Loading OmniSpeech from base model...")
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        model = model_cls.from_pretrained(model_base, low_cpu_mem_usage=False, config=cfg_pretrained, **kwargs)

        speech_projector_weights = torch.load(os.path.join(model_path, "speech_projector.bin"), map_location="cpu")
        speech_projector_weights = {k: v.to(torch.float16) for k, v in speech_projector_weights.items()}
        model.load_state_dict(speech_projector_weights, strict=False)
        model = model.to(device=device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = model_cls.from_pretrained(model_path, low_cpu_mem_usage=False, **kwargs)
        model = model.to(device=device)

    model.get_model().speech_encoder = build_speech_encoder(model.config)
    model.get_model().speech_encoder.to(device=device, dtype=torch.float16)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
