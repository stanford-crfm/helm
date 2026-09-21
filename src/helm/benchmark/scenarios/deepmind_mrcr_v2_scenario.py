import os
import re
from typing import List

import pandas as pd

from helm.common.general import ensure_directory_exists, ensure_file_downloaded
from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
    ScenarioMetadata,
    TEST_SPLIT,
)


class DeepMindMRCRV2Scenario(Scenario):
    """MRCR stands for "multi-round coreference resolution" and is a minimally simple long-context reasoning evaluation testing the length generalization capabilities of the model to follow a simple reasoning task with a fixed complexity: count instances of a body of text and reproduce the correct instance. The model is presented with a sequence of user-assistant turns where the user requests a piece of writing satisfying a format/style/topic tuple, and the assistant responds with a piece of writing. At the end of this sequence, the model is asked to reproduce the ith instance of the assistant output for one of the user queries (all responses to the same query are distinct). The model is also asked to certify that it will produce that output by first outputting a specialized and unique random string beforehand.

    Code: https://github.com/google-deepmind/eval_hub/tree/master/eval_hub/mrcr_v2
    Paper: https://arxiv.org/abs/2409.12640v2"""

    name = "deepmind_mrcr_v2"
    description = "MRCR (multi-round coreference resolution) is a minimally simple long-context reasoning evaluation testing the length generalization capabilities of the model to follow a simple reasoning task with a fixed complexity: count instances of a body of text and reproduce the correct instance. ([paper](https://arxiv.org/abs/2409.12640v2))"  # noqa: E501
    tags = ["long_context", "mrcr"]

    def __init__(self, needles: int, tokens: str):
        super().__init__()
        self.needles = needles
        self.tokens = tokens

    def get_instances(self, output_path: str) -> List[Instance]:
        # Convert "in_4096_8192" to strings like "in_(4096,8192)"
        match = re.match(r"in_(\d+)_(\d+)", self.tokens)
        if match:
            tokens_fragment = f"in_({match.group(1)},{match.group(2)})"
        else:
            tokens_fragment = self.tokens

        filename = f"mrcr_v2p1_{self.needles}needle_{tokens_fragment}_dynamic_fewshot_text_style_fast.csv"
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        source_url = f"https://storage.googleapis.com/mrcr_v2/{filename}"
        file_path = os.path.join(output_path, filename)
        ensure_file_downloaded(source_url, file_path)

        df = pd.read_csv(file_path)
        instances: List[Instance] = []
        for index, row in df.iterrows():
            input = Input(text=row["queries"])
            references = [Reference(output=Output(text=row["answer"]), tags=[CORRECT_TAG])]
            instance = Instance(
                id=f"{self.needles}needle_{self.tokens}_{index}",
                input=input,
                references=references,
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="deepmind_mrcr_v2",
            display_name="DeepMind MRCR v2",
            description=self.description,  # noqa: E501
            taxonomy=TaxonomyInfo(task="MRCR", what="Synthetic data", when="2025", who="None", language="English"),
            main_metric="deepmind_mrcr_v2_score",
            main_split="test",
        )
