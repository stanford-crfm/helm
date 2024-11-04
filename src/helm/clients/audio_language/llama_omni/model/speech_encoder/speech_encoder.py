# Adopted from https://github.com/ddlBoJack/SLAM-LLM/blob/main/src/slam_llm/models/encoder.py
import torch.nn as nn
import whisper


class WhisperWrappedEncoder:

    @classmethod
    def load(cls, model_config):

        def replace_layer_norm(module):
            from whisper.model import LayerNorm

            for name, child in module.named_children():
                if isinstance(child, LayerNorm):
                    old_params = child.state_dict()
                    new_layer_norm = nn.LayerNorm(
                        child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine
                    )
                    new_layer_norm.load_state_dict(old_params)
                    setattr(module, name, new_layer_norm)
                else:
                    replace_layer_norm(child)

        encoder = whisper.load_model(name="large-v3", device="cpu").encoder
        replace_layer_norm(encoder)
        return encoder
