from typing import List
import os
import json

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)
from tqdm import tqdm
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_file_downloaded


class COREBenchScenario(Scenario):
    """COREBench

    COREBench is a new audio benchmark incorporating multi-speaker conversations. It consists of conversational
    audio, transcript, question, and answer. There are two challenging features of this benchmark: (1) the questions
    are designed to require reasoning over multiple turns of conversation, and (2) the average audio length is
    longer than 1 minute, which is significantly longer than existing benchmarks.
    """

    ANNOT_URL = (
        "https://huggingface.co/datasets/stanford-crfm/COnversationalREasoningBench_v0.1/resolve/"
        "main/test/instances.jsonl"
    )
    HF_AUDIO_FOLDER = (
        "https://huggingface.co/datasets/stanford-crfm/COnversationalREasoningBench_v0.1/resolve/main/test/audio"
    )

    COREBENCH_INSTRUCTION = (
        "\n\n Answer the question by just giving the final answer and nothing else. "
        "Answer 'unanswerable' if the question is irrelevant to the audio or cannot be inferred."
    )

    name = "corebench"
    description = "Exploring multi-speaker conversational audio reasoning task."
    tags: List[str] = ["audio", "reasoning"]

    def load_jsonl(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f]

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        annot_save_path = os.path.join(output_path, "instances.jsonl")
        ensure_file_downloaded(source_url=COREBenchScenario.ANNOT_URL, target_path=annot_save_path)
        annotations = self.load_jsonl(annot_save_path)
        audio_save_dir = os.path.join(output_path, "audio")
        # Download audio files first
        for row in tqdm(annotations):
            audio_path = row["audio_path"]
            local_audio_path = os.path.join(audio_save_dir, audio_path)
            ensure_file_downloaded(
                source_url=os.path.join(COREBenchScenario.HF_AUDIO_FOLDER, audio_path), target_path=local_audio_path
            )
        for row in tqdm(annotations):
            local_audio_path = os.path.join(audio_save_dir, row["audio_path"])
            answer = row["answer"].lower()
            question = row["question"]

            input = Input(
                multimedia_content=MultimediaObject(
                    [
                        MediaObject(content_type="audio/mpeg", location=local_audio_path),
                        MediaObject(content_type="text/plain", text=question + self.COREBENCH_INSTRUCTION),
                    ]
                )
            )
            references = [Reference(Output(text=answer), tags=[CORRECT_TAG])]
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))
        return instances
