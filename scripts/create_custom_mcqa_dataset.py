from helm.benchmark.scenarios.scenario import (
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)
from helm.common.general import write
from typing import List
import os
from helm.common.codec import to_jsonl


# Creates a custom MCQA dataset about Stanford University
dataset: List[Instance] = [
    Instance(
        split=TRAIN_SPLIT,
        input=Input(
            text="What is the mascot of Stanford University?",
        ),
        references=[
            Reference(
                output=Output(text="Lion"),
                tags=[],
            ),
            Reference(
                output=Output(text="Tree"),
                tags=[CORRECT_TAG],
            ),
            Reference(
                output=Output(text="Bear"),
                tags=[],
            ),
            Reference(
                output=Output(text="Tiger"),
                tags=[],
            ),
        ],
    ),
    Instance(
        split=TRAIN_SPLIT,
        input=Input(
            text="What is the name of the Stanford University football team?",
        ),
        references=[
            Reference(
                output=Output(text="Cardinal"),
                tags=[CORRECT_TAG],
            ),
            Reference(
                output=Output(text="Buffaloes"),
                tags=[],
            ),
            Reference(
                output=Output(text="Bears"),
                tags=[],
            ),
            Reference(
                output=Output(text="Tigers"),
                tags=[],
            ),
        ],
    ),
    Instance(
        split=VALID_SPLIT,
        input=Input(
            text="What is the name of the Stanford University basketball team?",
        ),
        references=[
            Reference(
                output=Output(text="Cardinal"),
                tags=[CORRECT_TAG],
            ),
            Reference(
                output=Output(text="Buffaloes"),
                tags=[],
            ),
            Reference(
                output=Output(text="Bears"),
                tags=[],
            ),
            Reference(
                output=Output(text="Tigers"),
                tags=[],
            ),
        ],
    ),
    Instance(
        split=VALID_SPLIT,
        input=Input(
            text="Who was the founder of Stanford University?",
        ),
        references=[
            Reference(
                output=Output(text="Leland Stanford"),
                tags=[CORRECT_TAG],
            ),
            Reference(
                output=Output(text="John Harvard"),
                tags=[],
            ),
            Reference(
                output=Output(text="John Stanford"),
                tags=[],
            ),
            Reference(
                output=Output(text="Leland Harvard"),
                tags=[],
            ),
        ],
    ),
    Instance(
        split=VALID_SPLIT,
        input=Input(
            text="When was Stanford University founded?",
        ),
        references=[
            Reference(
                output=Output(text="1885"),
                tags=[],
            ),
            Reference(
                output=Output(text="1891"),
                tags=[CORRECT_TAG],
            ),
            Reference(
                output=Output(text="1893"),
                tags=[],
            ),
            Reference(
                output=Output(text="1895"),
                tags=[],
            ),
        ],
    ),
]


jsonl_path = "benchmark_output/scenarios/stanford_mcqa/data/training.jsonl"
os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)  # Make dir recursively if it doesn't exist
write(
    jsonl_path,
    to_jsonl(list(dataset)),
)
