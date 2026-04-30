# The following code includes templates and evaluation logic reproduced with minor modifications from:
# https://github.com/SCT-Bench/sctpublic/blob/c2843137b36218eac3f47e76eaa559c9fe7973e9/
#
# MIT License
#
# Copyright (c) 2025 liamgmccoy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import csv
import os
from typing import List

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
    ScenarioMetadata,
)

# Copied verbatim from:
# https://github.com/SCT-Bench/sctpublic/blob/c2843137b36218eac3f47e76eaa559c9fe7973e9/data/templates/guideline.md
GUIDELINE_TEMPLATE = """# Script Concordance Testing

You are taking a Script Concordance Test, which evaluates your understanding of medical knowledge.

In this test, you will evaluate how new information impacts a specific hypothesis. Use the following scale to rate the impact:

-2: Strongly decreases the likelihood of the hypothesis
-1: Slightly decreases the likelihood of the hypothesis
0: No effect on the likelihood of the hypothesis
+1: Slightly increases the likelihood of the hypothesis
+2: Strongly increases the likelihood of the hypothesis

## Response format

Respond with "Rating: " followed by a rating (-2, -1, 0, +1, +2) and a brief explanation for your choice.

This is an exam, and you are required to provide a valid answer.
"""

# Copied verbatim from:
# https://github.com/SCT-Bench/sctpublic/blob/c2843137b36218eac3f47e76eaa559c9fe7973e9/data/templates/testcase.md
TESTCASE_TEMPLATE = """## Scenario

{{ scenario }}

## Hypothesis

{{ hypothesis }}

## Additional Information

{{ additional information }}

## Response"""

# Copied verbatim from:
# https://github.com/SCT-Bench/sctpublic/blob/c2843137b36218eac3f47e76eaa559c9fe7973e9/data/templates/example_-2.md
EXAMPLE_NEG2 = """### Scenario

A 2-year-old female has a 2-day history of left ear pain. Her mother has been giving her acetaminophen with temporary improvement in pain. You elicit further history from the mother.  The mother reports that the patient has a history of multiple previous episodes of acute otitis media that have required oral antibiotics. You complete a physical exam on the patient. The patient's mother inquiries about your management plans.

### Hypothesis

If you were thinking of:   Prescribing oral amoxicillin for 10 days

### Additional Information

And then you find: An erythematous, swollen external ear canal with white discharge

### Response

Rating: -2
"""

# Copied verbatim from:
# https://github.com/SCT-Bench/sctpublic/blob/c2843137b36218eac3f47e76eaa559c9fe7973e9/data/templates/example_-1.md
EXAMPLE_NEG1 = """### Scenario

A 12-month-old male child presents to the pediatric emergency department with a three-day history of diarrhea and intermittent abdominal pain.

### Hypothesis

If you were thinking of a diagnosis of:    Lactose intolerance

### Additional Information

And then you find: A temperature of 38.1°C rectally

### Response

Rating: -1
"""

# Copied verbatim from:
# https://github.com/SCT-Bench/sctpublic/blob/c2843137b36218eac3f47e76eaa559c9fe7973e9/data/templates/example_0.md
EXAMPLE_0 = """### Scenario

A sixteen-year-old female presents to her primary care provider with her parents for concerns of vomiting following meals for the past month. You elicit further history from the parents and the patient.  The patient has recently been promoted to the varsity, competitive cheer squard. Due to her short stature and petite size, she has been selected to be a flyer and top to one of their stunt pyramids.

### Hypothesis

If you were considering the following laboratory study:     A rapid urine HCG test

### Additional Information

And then you find: The patient denies sexual activity

### Response

Rating: 0
"""

# Copied verbatim from:
# https://github.com/SCT-Bench/sctpublic/blob/c2843137b36218eac3f47e76eaa559c9fe7973e9/data/templates/example_+1.md
EXAMPLE_POS1 = """### Scenario

A five-year child presents to his primary care provider with a rash on his upper and lower extremities for 2 weeks. You elicit further history from the parents.  The patient has had intermittent dry rough skin since birth. In infancy, the rash primarily presented on his cheeks and trunk, but for the last few years, he has had intermittent outbreaks of rough skin on his trunk and popliteal and antecubital fossae.

### Hypothesis

If you were considering the following laboratory test:     A rapid streptococcal antigen test

### Additional Information

And then you find: A rough popular truncal rash and axillary temperature of 99.6F

### Response

Rating: +1
"""

# Copied verbatim from:
# https://github.com/SCT-Bench/sctpublic/blob/c2843137b36218eac3f47e76eaa559c9fe7973e9/data/templates/example_+2.md
EXAMPLE_POS2 = """### Scenario

You are called to the newborn nursery by the labor and delivery nurse to examine a newborn infant with abnormal tone, dysmorphic facial features and a heart murmur. You elicit further prenatal history from the mother's chart, perform a complete physical examination on the infant and consider your further evaluation for this infant. A karyotype is obtained and confirms a diagnosis of Trisomy 21. The parents inquire about possible future complications from this diagnosis. In children with Trisomy 21, how would you handle the following associated issues in the future?

### Hypothesis

If you were thinking of:  Ordering a complete blood count (CBC) with differential and blood smear

### Additional Information

And then you find: A scattered petechial rash

### Response

Rating: +2
"""

FEW_SHOT_EXAMPLES = {
    "-2": EXAMPLE_NEG2,
    "-1": EXAMPLE_NEG1,
    "0": EXAMPLE_0,
    "+1": EXAMPLE_POS1,
    "+2": EXAMPLE_POS2,
}

RATING_INDEX_MAP = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
EXPERT_DISTRIBUTION_COLUMNS = ["-2", "-1", "0", "1", "2"]


class SCTBenchScenario(Scenario):
    """
    SCT-Bench evaluates clinical reasoning under uncertainty using Script Concordance Tests (SCTs).
    SCTs are validated medical assessment tools that measure how new information alters diagnostic
    and treatment hypotheses. Models rate the impact on a -2 to +2 scale and are scored against
    an expert clinician panel distribution.

    The public dataset contains 177 questions from the Adelaide SCT and Open Medical SCT datasets.

    Dataset: https://github.com/SCT-Bench/sctpublic
    """

    DATASET_DOWNLOAD_URL = (
        "https://raw.githubusercontent.com/SCT-Bench/sctpublic/"
        "c2843137b36218eac3f47e76eaa559c9fe7973e9/data/sct_cleaned_full.csv"
    )
    FILENAME = "sct_cleaned_full.csv"

    name = "sct_bench"
    description = (
        "SCT-Bench evaluates clinical reasoning under uncertainty using Script Concordance Tests. "
        "Models rate the impact of new clinical information on diagnostic/treatment hypotheses "
        "on a -2 to +2 scale, scored against an expert clinician panel distribution."
    )
    tags = ["knowledge", "reasoning", "biomedical", "clinical"]

    def __init__(self, reason: bool = False, few_shot: bool = False):
        super().__init__()
        self.reason = reason
        self.few_shot = few_shot

    def _build_prompt_template(self) -> str:
        # Adapted from:
        # https://github.com/SCT-Bench/sctpublic/blob/c2843137b36218eac3f47e76eaa559c9fe7973e9/modeling.py#L48-L73
        guideline = GUIDELINE_TEMPLATE
        if not self.reason:
            guideline = guideline.replace(" and a brief explanation for your choice", "")

        prompt = guideline
        if self.few_shot:
            prompt += "## Examples with Response Labels"
            for rating in ["-2", "-1", "0", "+1", "+2"]:
                prompt += FEW_SHOT_EXAMPLES[rating]
        prompt += TESTCASE_TEMPLATE

        return prompt

    def _find_best_rating(self, expert_dist: List[float]) -> str:
        ratings = ["-2", "-1", "0", "+1", "+2"]
        best_idx = expert_dist.index(max(expert_dist))
        return ratings[best_idx]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = os.path.join(output_path, self.FILENAME)
        ensure_file_downloaded(
            source_url=self.DATASET_DOWNLOAD_URL,
            target_path=data_path,
            unpack=False,
        )

        prompt_template = self._build_prompt_template()
        instances: List[Instance] = []

        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row["sct_stem"] or not row["question"] or not row["additional_info"]:
                    continue

                prompt = (
                    prompt_template.replace("{{ scenario }}", row["sct_stem"])
                    .replace("{{ hypothesis }}", row["question"])
                    .replace("{{ additional information }}", row["additional_info"])
                )

                expert_dist = [float(row[col]) for col in EXPERT_DISTRIBUTION_COLUMNS]
                best_rating = self._find_best_rating(expert_dist)

                instance = Instance(
                    input=Input(text=prompt),
                    references=[Reference(Output(text=best_rating), tags=[CORRECT_TAG])],
                    extra_data={"expert_distribution": expert_dist},
                    split=TEST_SPLIT,
                    id=str(row["question_id"]),
                )
                instances.append(instance)

        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="sct_bench",
            display_name="SCT-Bench",
            description=(
                "SCT-Bench evaluates clinical reasoning under uncertainty using Script Concordance "
                "Tests (SCTs). Models rate the impact of new clinical information on diagnostic/"
                "treatment hypotheses on a -2 to +2 scale, scored against an expert clinician "
                "panel distribution "
                "([SCT-Bench](https://github.com/SCT-Bench/sctpublic))."
            ),
            taxonomy=TaxonomyInfo(
                task="clinical reasoning",
                what="Rate impact of new clinical information on diagnostic/treatment hypotheses",
                when="Any",
                who="Clinician",
                language="English",
            ),
            main_metric="sct_score",
            main_split="test",
        )
