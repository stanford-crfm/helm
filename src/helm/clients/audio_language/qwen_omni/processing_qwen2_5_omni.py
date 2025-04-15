# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for Qwen2.5Omni.
"""

import logging
from typing import List, Optional, Union

import numpy as np
import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput


class Qwen2_5OmniProcessorKwargs(ProcessingKwargs):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
    }


class Qwen2_5OmniProcessor(ProcessorMixin):

    attributes = ["omni_processor", "feature_extractor", "tokenizer"]
    omni_processor_class = "Qwen2VLImageProcessor"
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    valid_kwargs = ["chat_template"]

    def __init__(self, omni_processor=None, feature_extractor=None, tokenizer=None, chat_template=None):
        self.image_token = "<|IMAGE|>"
        self.audio_token = "<|AUDIO|>"
        self.video_token = "<|VIDEO|>"
        self.vision_bos_token = "<|vision_bos|>"
        self.vision_eos_token = "<|vision_eos|>"
        self.audio_bos_token = "<|audio_bos|>"
        self.audio_eos_token = "<|audio_eos|>"
        super().__init__(omni_processor, feature_extractor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        audios: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        sampling_rate: Optional[int] = 16000,
        fps: Optional[List[float]] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        use_audio_in_video: Optional[bool] = False,
        position_id_per_seconds: int = 25,
        seconds_per_chunk: float = 2.0,
        **kwargs: Unpack[Qwen2_5OmniProcessorKwargs],
    ) -> BatchFeature:

        output_kwargs = self._merge_kwargs(
            Qwen2_5OmniProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            padding=padding,
            **kwargs,
        )

        if audios is not None:
            audios_inputs = self.feature_extractor(
                audios, sampling_rate=sampling_rate, return_attention_mask=True, padding="max_length", **kwargs
            )
            audios_inputs["feature_attention_mask"] = audios_inputs.pop(
                "attention_mask"
            )  # rename feature_attention_mask to prevent conflicts later on
            audios_inputs["input_features"] = audios_inputs.pop(
                "input_features"
            )  # rename input_features to prevent conflicts later on
            input_lengths = (audios_inputs["feature_attention_mask"].sum(-1).numpy() - 1) // 2 + 1
            audio_lengths = (input_lengths - 2) // 2 + 1
        else:
            audios_inputs = {}
            audio_lengths = None

        if images is not None:
            images_inputs = self.omni_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = images_inputs["image_grid_thw"]
        else:
            images_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.omni_processor(images=None, videos=videos, **output_kwargs["videos_kwargs"])
            if fps is None:
                fps = [2.0] * len(videos)
            videos_inputs["video_second_per_grid"] = [
                self.omni_processor.temporal_patch_size / fps[i] for i in range(len(fps))
            ]
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        if not isinstance(text, list):
            text = [text]

        merge_length = self.omni_processor.merge_size**2
        audio_index = 0
        image_index = 0
        video_index = 0
        for i in range(len(text)):
            positions = []
            for special_token in [self.audio_token, self.image_token, self.video_token]:
                start = 0
                while True:
                    pos = text[i].find(special_token, start)
                    if pos == -1:
                        break
                    positions.append((pos, special_token))
                    start = pos + len(special_token)
            positions.sort(key=lambda x: x[0])
            for _, special_token in positions:
                if audios is not None and special_token == self.audio_token:
                    text[i] = text[i].replace(
                        self.audio_token,
                        "<|audio_placeholder|>" * audio_lengths[audio_index],
                        1,
                    )
                    audio_index += 1
                elif images is not None and special_token == self.image_token:
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|image_placeholder|>" * (image_grid_thw[image_index].prod() // merge_length),
                        1,
                    )
                    image_index += 1
                elif videos is not None and special_token == self.video_token:
                    if use_audio_in_video:
                        audio_t_index = torch.arange(audio_lengths[audio_index])
                        video_t_index = (
                            torch.arange(video_grid_thw[video_index][0])
                            .view(-1, 1, 1)
                            .expand(
                                -1,
                                video_grid_thw[video_index][1] // self.omni_processor.merge_size,
                                video_grid_thw[video_index][2] // self.omni_processor.merge_size,
                            )
                            .flatten()
                            * videos_inputs["video_second_per_grid"][video_index]
                            * position_id_per_seconds
                        ).long()
                        t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                        video_chunk_indexes = self.get_chunked_index(video_t_index, t_ntoken_per_chunk)
                        audio_chunk_indexes = self.get_chunked_index(audio_t_index, t_ntoken_per_chunk)
                        placeholder_string = str()
                        placeholder_string += self.vision_bos_token + self.audio_bos_token
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            video_chunk_index = video_chunk_indexes[j] if j < len(video_chunk_indexes) else None
                            audio_chunk_index = audio_chunk_indexes[j] if j < len(audio_chunk_indexes) else None
                            if video_chunk_index is not None:
                                placeholder_string += "<|video_placeholder|>" * (
                                    video_chunk_index[1] - video_chunk_index[0]
                                )
                            if audio_chunk_index is not None:
                                placeholder_string += "<|audio_placeholder|>" * (
                                    audio_chunk_index[1] - audio_chunk_index[0]
                                )
                        placeholder_string += self.audio_eos_token + self.vision_eos_token
                        text[i] = text[i].replace(
                            self.vision_bos_token + self.video_token + self.vision_eos_token,
                            placeholder_string,
                            1,
                        )
                        audio_index += 1
                        video_index += 1
                    else:
                        text[i] = text[i].replace(
                            self.video_token,
                            "<|video_placeholder|>" * (video_grid_thw[video_index].prod() // merge_length),
                            1,
                        )
                        video_index += 1

            text[i] = text[i].replace("<|audio_placeholder|>", self.audio_token)
            text[i] = text[i].replace("<|image_placeholder|>", self.image_token)
            text[i] = text[i].replace("<|video_placeholder|>", self.video_token)

        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**texts_inputs, **images_inputs, **videos_inputs, **audios_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def get_chunked_index(self, t_index, t_ntoken_per_chunk):
        def _iter():
            i, start_idx = 0, 0  # skip bos token
            current_chunk = 1
            while i < len(t_index):  # skip eos token
                if t_index[i] >= current_chunk * t_ntoken_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(t_index))

        return list(_iter())

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        for conversation in conversations:
            if conversation[0]["role"] != "system" or conversation[0]["content"] != (
                "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable"
                " of perceiving auditory and visual inputs, as well as generating text and speech."
            ):
                logging.warning(
                    "System prompt modified, audio output may not work as expected. "
                    "Audio output mode only works when using default system prompt 'You are Qwen,"
                    " a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving"
                    " auditory and visual inputs, as well as generating text and speech.'"
                )
        return super().apply_chat_template(conversations, chat_template, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        omni_processor_input_names = self.omni_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + omni_processor_input_names
                + ["feature_attention_mask"]
                + ["video_second_per_grid"]
            )
        )


__all__ = ["Qwen2_5OmniProcessor"]
