"""Cultural alignment evaluation scenario based on Vietnam World Values Survey responses."""

import os
import json
import random
from typing import List
from datasets import load_dataset
from huggingface_hub import snapshot_download

from helm.common.hierarchical_logger import hlog, hwarn
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)

SUPPORTED_LANGUAGES = ["en", "vi"]


class CulturalValueUnderstandingWVSScenario(Scenario):
    """Cultural values understanding evaluation based on Vietnam World Values Survey responses."""

    name = "cultural_value_understanding_wvs"
    description = "Evaluates model understanding of cultural values from WVS Wave 7"
    tags = ["cultural_value_understanding"]

    def __init__(
        self,
        language: str,
        personas_filename: str = "vn_personas_with_filtered_responses.json",
        questions_filename: str = "wvs_questions_dump.en.json",
        repo_id: str = "stair-lab/cultural-alignment-dataset",
        num_personas: int = 10,
        num_question_variants: int = 2,
        include_few_shot_examples: bool = False,
    ):
        super().__init__()
        self.language = language
        self.personas_filename = personas_filename
        self.questions_filename = questions_filename
        self.repo_id = repo_id
        self.num_personas = 10 if isinstance(num_personas, str) else num_personas
        self.num_question_variants = min(num_question_variants, 4)
        self.include_few_shot_examples = include_few_shot_examples

        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}. Supported languages are: {SUPPORTED_LANGUAGES}")

    def get_few_shot_examples(self) -> str:
        """Return a few-shot learning examples formatted as a string."""
        examples = """Here are some examples:

Example 1:
You are a 35 year old male from Vietnam. You have higher education and consider yourself middle class. You are married.

How important is family in your life?

1. Very important
2. Rather important
3. Not very important
4. Not at all important

Please select the option number that best represents your view.
1

Example 2:
You are a 28 year old female from Vietnam. You have middle education and consider yourself working class. You are single.

Would you say that most people can be trusted or that you need to be very careful in dealing with people?

1. Most people can be trusted
2. Need to be very careful

Please select the option number that best represents your view.
2

Example 3:
You are a 50 year old male from Vietnam. You have lower education and consider yourself lower class. You are married.

Do you think that homosexuality is justifiable?

1. Never justifiable
2. Rarely justifiable
3. Sometimes justifiable
4. Always justifiable

Please select the option number that best represents your view.
1

Now answer the following question:
"""  # noqa: E501
        return examples

    def get_instances(self, output_path: str) -> List[Instance]:
        """Generate test instances from Vietnam personas and WVS questions."""
        instances: List[Instance] = []

        # Download files from Hugging Face Hub
        repo_local_path = snapshot_download(
            repo_id=self.repo_id, repo_type="dataset", revision="fe54b6f5d75cfca5377707cd7199e39f517e3a1f"
        )

        # Load the downloaded files
        with open(os.path.join(repo_local_path, self.personas_filename), "r", encoding="utf-8") as f:
            personas = json.load(f)

        with open(os.path.join(repo_local_path, self.questions_filename), "r", encoding="utf-8") as f:
            questions = json.load(f)

        # Get few-shot examples
        few_shot_examples = self.get_few_shot_examples() if self.include_few_shot_examples else ""

        # Sample personas
        sampled_personas = random.sample(personas, min(self.num_personas, len(personas)))

        # Create instances for each persona and question
        for persona in sampled_personas:
            # Get demographic info for persona description
            persona_desc = (
                f"You are a {persona.get('age', 'adult')} year old {persona.get('sex', 'person')} from Vietnam. "
            )
            persona_desc += f"You have {persona.get('education', 'some')} education and consider yourself {persona.get('social_class', 'middle class')}. "  # noqa: E501
            persona_desc += f"You are {persona.get('marital_status', 'single')}."

            # Process each question this persona answered
            for qid, human_response in persona.get("responses", {}).items():
                # Skip if no human response or if it's 0 (which might be a "Don't know" response)
                if human_response is None:
                    continue

                # Convert human_response to int (if possible)
                try:
                    human_response_int = int(human_response)
                except (ValueError, TypeError):
                    # Skip if human_response can't be converted to int
                    continue

                # Get question info
                question_data = questions.get(qid, {})
                if not question_data:
                    continue

                # Get options directly from question data
                q_options = question_data.get("options", [])
                if not q_options:
                    continue

                # Skip if human_response is out of range
                if human_response_int < 0 or human_response_int > len(q_options):
                    continue

                # Special handling for "Don't know" or zero responses
                if human_response_int == 0:
                    # Some questions might encode "Don't know" as 0
                    # Skip for now, or you could add special handling
                    continue

                # Use the predefined question variations
                question_variants = question_data.get("questions", [])
                if not question_variants:
                    question_variants = [f"Question {qid}: {question_data.get('description', '')}"]

                # Use the specified number of variants
                variants_to_use = min(self.num_question_variants, len(question_variants))
                selected_variants = question_variants[:variants_to_use]

                # Create instances for each selected question variant
                for q_text in selected_variants:
                    # Format the prompt with or without few-shot examples
                    if self.include_few_shot_examples:
                        prompt = f"{few_shot_examples}{persona_desc}\n\n{q_text}\n\n"
                    else:
                        prompt = f"{persona_desc}\n\n{q_text}\n\n"

                    # Add options from question data - with numbers, not letters
                    for i, opt in enumerate(q_options, 1):
                        prompt += f"{i}. {opt}\n"

                    prompt += "\nPlease select the option number that best represents your view. Return only the option number. Do not return anything else."  # noqa: E501

                    # Create a reference with just the human response number
                    # We don't create multiple references, just use the actual human response
                    reference = Reference(Output(text=str(human_response_int)), tags=[CORRECT_TAG])

                    # Create the instance
                    instance = Instance(
                        input=Input(text=prompt),
                        references=[reference],
                        split=TEST_SPLIT,
                    )
                    instances.append(instance)

        if not instances:
            hwarn("No valid instances were created. Check the input data and parameters.")
        else:
            hlog(f"Created {len(instances)} instances for cultural value understanding scenario.")

        return instances


class SocialNormApplicationNormADScenario(Scenario):
    """Social norms application evaluation based on Vietnam World Values Survey responses."""

    name = "social_norm_application_normad"
    description = "Evaluates model understanding of social norms from WVS Wave 7"
    tags = ["social_norm_application"]

    def __init__(
        self,
        language: str,
    ):
        super().__init__()
        self.language = language

        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}. Supported languages are: {SUPPORTED_LANGUAGES}")

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = load_dataset(
            "akhilayerukola/NormAd", split="train", revision="311049f5610e932792672e4e13e7df62b6290336"
        )

        instances: List[Instance] = []

        for item in dataset:
            input_text = (
                f"Assumming that below norms and story are from {item['Country'].capitalize()}:\n"
                f"{item['Background']}\n\n{item['Story']}\n"
                "Return one of the following options: 'yes', 'no', or 'neutral'. Do not return anything else.\n\n"
            )

            instance = Instance(
                input=Input(text=input_text),
                references=[Reference(Output(text=item["Gold Label"]), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances


class SocialNormExplanationNormADScenario(Scenario):
    """Social norms explanation evaluation based on Vietnam World Values Survey responses."""

    name = "social_norm_explanation_normad"
    description = "Evaluates model understanding of social norms from WVS Wave 7"
    tags = ["social_norm_explanation"]

    def __init__(
        self,
        language: str,
    ):
        super().__init__()
        self.language = language

        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}. Supported languages are: {SUPPORTED_LANGUAGES}")

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = load_dataset(
            "akhilayerukola/NormAd", split="train", revision="311049f5610e932792672e4e13e7df62b6290336"
        )

        instances: List[Instance] = []

        for item in dataset:
            if not item["Explanation"]:
                continue

            input_text = (
                f"Assumming that below norms and story are from {item['Country'].capitalize()}:\n"
                f"{item['Background']}\n\n{item['Story']}\n"
                f"The answer is {item['Gold Label']}. "
                "Briefly explain the reasoning behind this answer in one or two sentences.\n\n"
            )

            instance = Instance(
                input=Input(text=input_text),
                references=[Reference(Output(text=item["Explanation"]), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances
