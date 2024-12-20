from typing import List, Dict, Any, DefaultDict
from datasets import load_dataset, Dataset
from collections import defaultdict

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    Reference,
    Scenario,
    Instance,
    Input,
    TRAIN_SPLIT,
    VALID_SPLIT,
    Output,
)


class OpenAssistantScenario(Scenario):
    """
    This scenario is based on the OpenAssistant Conversations Dataset (OASST1) released by LAION.
    The dataset includes 66,497 human-generated, human-annotated assistant-style conversation trees
    in 35 different languages. Each conversation tree has an initial prompt message as the root node, and
    every node can have multiple child messages. In total, there are 161,443 messages in the dataset.

    https://arxiv.org/pdf/2304.07327.pdf

    Note that we are only using the initial prompt messages and their direct responses in this scenario.
    We are not including the subsequent turns of the chat.
    """

    name = "open_assistant"
    description = "The conversation dataset released by LAION's Open Assistant project."
    tags = ["instructions"]

    def __init__(self, language: str):
        super().__init__()
        self.language: str = language

    def get_instances(self, output_path: str) -> List[Instance]:
        """An example of a message in the dataset:
        {
            'message_id': '6ab24d72-0181-4594-a9cd-deaf170242fb',
            'parent_id': None,
            'user_id': 'c3fe8c76-fc30-4fa7-b7f8-c492f5967d18',
            'created_date': '2023-02-05T14:23:50.983374+00:00',
            'text': 'Can you write a short introduction about the relevance of the term "monopsony" in economics?'
                    'Please use examples related to potential monopsonies in the labour market and'
                    'cite relevant research.',
            'role': 'prompter',
            'lang': 'en',
            'review_count': 3,
            'review_result': True,
            'deleted': False,
            'rank': None,
            'synthetic': False,
            'model_name': None,
            'detoxify': {
                'toxicity': 0.00044308538781479,
                'severe_toxicity': 3.252684837207198e-05,
                'obscene': 0.00023475120542570949,
                'identity_attack': 0.0001416115992469713,
                'insult': 0.00039489680784754455,
                'threat': 4.075629112776369e-05,
                'sexual_explicit': 2.712695459194947e-05
            },
            'message_tree_id': '6ab24d72-0181-4594-a9cd-deaf170242fb',
            'tree_state': 'ready_for_export',
            'emojis': {
                'name': ['+1', '_skip_reply', '_skip_ranking'],
                'count': [10, 1, 4]
            },
            'labels': {
                'name': ['spam', 'lang_mismatch', 'pii', 'not_appropriate', 'hate_speech', 'sexual_content',
                        'quality', 'toxicity', 'humor', 'creativity','violence'],
                'value': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9166666666666666, 0.16666666666666666, 0.3333333333333333,
                        0.6666666666666666, 0.0],
                'count': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            }
        }
        """

        def matches_target_language(msg):
            """Check if the message is in the target language"""
            if self.language == "all":
                return True
            else:
                return msg["lang"] == self.language

        def get_split_instances(split: Dataset, split_tag: str):
            """Get instances from a split (e.g. train) of the dataset"""
            # Step 1: get all initial prompts (roots of the conversation trees)
            initial_prompts: Dict[str, str] = {}

            for msg in split:
                assert msg["model_name"] is None
                if msg["parent_id"] is None and matches_target_language(msg):
                    assert msg["role"] == "prompter"
                    assert msg["message_id"] not in initial_prompts
                    initial_prompts[msg["message_id"]] = msg["text"]

            # Step 2: for each initial prompt, find the corresponding assistant messages.
            # Each initial prompt can have multiple assistant responses.
            prompt_responses: DefaultDict[str, List[str]] = defaultdict(list)
            for msg in split:
                if msg["role"] == "assistant" and msg["parent_id"] in initial_prompts and matches_target_language(msg):
                    prompt_responses[msg["parent_id"]].append(msg["text"])

            # Step 3: for each initial prompt, create an instance
            instances: List[Instance] = []
            for msg_id, prompt in initial_prompts.items():
                reference_texts: List[str] = []
                if msg_id in prompt_responses:
                    reference_texts = prompt_responses[msg_id]
                instance = Instance(
                    input=Input(text=prompt),
                    references=[Reference(Output(text=ref), tags=[CORRECT_TAG]) for ref in reference_texts],
                    split=split_tag,
                )
                instances.append(instance)
            return instances

        # Download the raw data from Huggingface
        dataset: Any = load_dataset("OpenAssistant/oasst1", revision="fdf72ae0827c1cda404aff25b6603abec9e3399b")

        # Get the instances for each split
        train_instances = get_split_instances(dataset["train"], TRAIN_SPLIT)
        valid_instances = get_split_instances(dataset["validation"], VALID_SPLIT)

        return train_instances + valid_instances
