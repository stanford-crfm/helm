from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from helm.clients.audio_language.llama_omni.model.omni_speech_arch import OmniSpeechMetaModel, OmniSpeechMetaForCausalLM


class OmniSpeechConfig(LlamaConfig):
    model_type = "omni_speech_llama"


class OmniSpeechLlamaModel(OmniSpeechMetaModel, LlamaModel):
    config_class = OmniSpeechConfig

    def __init__(self, config: LlamaConfig):
        super(OmniSpeechLlamaModel, self).__init__(config)


class OmniSpeechLlamaForCausalLM(LlamaForCausalLM, OmniSpeechMetaForCausalLM):
    config_class = OmniSpeechConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = OmniSpeechLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        tgt_units: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = (
                self.prepare_inputs_labels_for_speech_and_text(
                    input_ids, position_ids, attention_mask, past_key_values, labels, speech, speech_lengths
                )
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if speech is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = (
                self.prepare_inputs_labels_for_speech_and_text(
                    inputs, position_ids, attention_mask, None, None, speech, speech_lengths
                )
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        speech = kwargs.pop("speech", None)
        speech_lengths = kwargs.pop("speech_lengths", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if speech is not None:
            inputs["speech"] = speech
            inputs["speech_lengths"] = speech_lengths
        return inputs


AutoConfig.register("omni_speech_llama", OmniSpeechConfig)
AutoModelForCausalLM.register(OmniSpeechConfig, OmniSpeechLlamaForCausalLM)
