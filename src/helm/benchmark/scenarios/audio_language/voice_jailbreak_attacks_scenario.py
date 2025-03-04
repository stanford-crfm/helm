import os
from typing import List

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Scenario,
    Instance,
    Input,
)
from helm.common.audio_utils import is_invalid_audio_file
from helm.common.media_object import MediaObject, MultimediaObject


class VoiceJailbreakAttacksScenario(Scenario):
    """
    Voice Jailbreak Attacks Against GPT-4o

    Paper: https://arxiv.org/abs/2405.19103

    The examples used in this scenario was generated following the instructions in the GitHub:
    https://github.com/TrustAIRLab/VoiceJailbreakAttack
    Note: The more advanced jailbreaking subset addresses the AI as "ChatGPT".

    We ran:
    python tts/prompt2audio.py --dataset baseline --voice fable
    python tts/prompt2audio.py --dataset baseline --voice fable

    then placed the generated folders at benchmark_output/scenarios/voice_jailbreak_attacks:

    voice_jailbreak_attacks/
        baseline_fable/
            *.wav
        textjailbreak_fable/
            *.wav

    Base prompts (30 total):
    https://github.com/TrustAIRLab/VoiceJailbreakAttack/blob/main/data/question_set/questions_tiny.csv
    Jailbreak templates:
    https://github.com/TrustAIRLab/VoiceJailbreakAttack/blob/main/data/jailbreak_prompts/text_jailbreak_prompts.csv

    The authors used Attack Success Rate (ASR) as the evaluation metric.
    Specifically, two authors manually label each response to determine if it answers the forbidden
    question. If there is any uncertainty, the authors discuss to reach a conclusion

    Citation:

    @article{SWBZ24,
      author = {Xinyue Shen and Yixin Wu and Michael Backes and Yang Zhang},
      title = {{Voice Jailbreak Attacks Against GPT-4o}},
      journal = {{CoRR abs/2405.19103}},
      year = {2024}
    }
    """

    VALID_SUBSETS = ["baseline", "textjailbreak"]

    name = "voice_jailbreak_attacks"
    description = "Voice jailbreak attacks against GPT-4o ([Shen et al, 2024](https://arxiv.org/abs/2405.19103))."
    tags = ["audio", "safety"]

    def __init__(self, subset: str = "baseline") -> None:
        super().__init__()
        assert subset in self.VALID_SUBSETS, f"Invalid subset: {subset}"
        self._subset = subset

    def get_instances(self, output_path: str) -> List[Instance]:
        audio_directory_path: str = os.path.join(output_path, f"{self._subset}_fable")
        assert os.path.exists(audio_directory_path), f"Audio directory does not exist: {audio_directory_path}"

        instances: List[Instance] = []
        for file in os.listdir(audio_directory_path):
            if not file.endswith("wav"):
                continue

            audio_path: str = os.path.join(audio_directory_path, file)
            assert not is_invalid_audio_file(audio_path), f"Invalid audio file: {audio_path}"

            input = Input(
                multimedia_content=MultimediaObject(
                    media_objects=[
                        MediaObject(location=audio_path, content_type="audio/wav"),
                    ]
                )
            )
            instances.append(Instance(input=input, references=[], split=TEST_SPLIT))

        return instances
