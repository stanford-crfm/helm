from dataclasses import dataclass

from .perturbation import Perturbation
from .perturbation_description import PerturbationDescription

from benchmark.scenario import Instance
import random

# ADAPTED FROM
# https://github.com/GEM-benchmark/NL-Augmenter/blob/main/transformations/filler_word_augmentation/transformation.py

# Speaker opinion/mental state phrases
# Taken from Kovatchev et al. (2021)
SPEAKER_PHRASES = [
    "I think",
    "I believe",
    "I mean",
    "I guess",
    "that is",
    "I assume",
    "I feel",
    "In my opinion",
    "I would say",
]

# Words and phrases indicating uncertainty
# Taken from Kovatchev et al. (2021)
UNCERTAIN_PHRASES = ["maybe", "perhaps", "probably", "possibly", "most likely"]

# Filler words that should preserve the meaning of the phrase
# Taken from Laserna et al. (2014)
FILL_PHRASE = ["uhm", "umm", "ahh", "err", "actually", "obviously", "naturally", "like", "you know"]


class FillerWordsPerturbation(Perturbation):
    """
    Randomly inserts filler words and phrases in the sentence.
    Perturbation example:
    Input:
        The quick brown fox jumps over the lazy dog.
    Output:
        The quick brown fox jumps over probably the lazy dog.

    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        insert_prob: float = 0.0

    name: str = "filler_words"

    def __init__(self, insert_prob=0.333, max_num_insert=None, speaker_ph=True, uncertain_ph=True, fill_ph=True):
        self.insert_prob = insert_prob
        self.max_num_insert = max_num_insert
        self.speaker_ph = speaker_ph
        self.uncertain_ph = uncertain_ph
        self.fill_ph = fill_ph

    @property
    def description(self) -> PerturbationDescription:
        return FillerWordsPerturbation.Description(name=self.name, robustness=True, insert_prob=self.insert_prob)

    def apply(self, instance: Instance) -> Instance:
        assert instance.id is not None
        seed = int(instance.id[2:])
        # set seed based on instance ID
        random.seed(seed)
        return super().apply(instance)

    @staticmethod
    def gen_filled_text(text, insert_prob, max_num_insert=1, speaker_ph=True, uncertain_ph=True, fill_ph=True):
        all_fillers = []
        if speaker_ph:
            all_fillers.extend(SPEAKER_PHRASES)
        if uncertain_ph:
            all_fillers.extend(UNCERTAIN_PHRASES)
        if fill_ph:
            all_fillers.extend(FILL_PHRASE)

        insert_count = 0
        perturbed_lines = []
        for line in text.split("\n"):
            perturbed_words = []
            for word in line.split():
                if (max_num_insert is None or insert_count < max_num_insert) and random.random() <= insert_prob:
                    random_filler = random.choice(all_fillers)
                    perturbed_words.append(random_filler)
                    insert_count += 1
                perturbed_words.append(word)
            perturbed_lines.append(" ".join(perturbed_words))

        perturbed_text = "\n".join(perturbed_lines)

        return perturbed_text

    def perturb(self, text: str) -> str:
        return self.gen_filled_text(
            text, self.insert_prob, self.max_num_insert, self.speaker_ph, self.uncertain_ph, self.fill_ph
        )
