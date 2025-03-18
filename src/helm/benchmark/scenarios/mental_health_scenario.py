import pandas as pd
from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    PassageQuestionInput,
    Output,
)


class MentalHealthScenario(Scenario):
    """
    This scenario evaluates language models' ability to generate appropriate counseling responses
    in mental health conversations. The dataset contains counseling dialogues covering
    various topics including workplace issues, anxiety, suicidal thoughts, relationship
    problems, and more.

    Each dialogue consists of interactions between a counselor and a client, where the counselor
    demonstrates expert mental health counseling techniques. The dialogues were selected based on high
    quality scores from multiple evaluators.

    Example dialogue structure:
    ```
    counselor: Hi there, to start can you tell me your name and a little bit about what's been going on?
    client: I sleep too much... I'm 23, female and work as IT professional. I feel like I'm not fitting in...
    counselor: I can see you have been facing challenges with feeling like you don't fit in...
    ```

    The task is to generate the next counselor response given the conversation history. Models
    are evaluated on their ability to:
    1. Provide empathetic and supportive responses
    2. Follow proper mental health counseling protocols
    3. Generate contextually appropriate interventions

    The dataset includes:
    - 7 complete dialogues covering different mental health topics
    - Metadata about dialogue topic and type
    - Gold-standard counselor responses as references
    - Full conversation history for context

    Each instance includes:
    - input: Previous conversation turns formatted with speaker labels
    - reference: The actual counselor's response (gold standard)
    - metadata: Topic and type of mental health conversation
    """

    name = "mental_health"
    description = "A dataset containing a counselor and mental health patient conversation, where the objective is to \
                    generate an empathetic counselor response."
    tags = ["dialogue", "counseling", "mental_health", "empathy", "healthcare"]

    def process_dialogue_data(self, data: pd.DataFrame) -> List[Instance]:
        """
        Process the dialogue data into evaluation instances.
        Each instance represents a point in the conversation where the model needs to generate
        a counselor response.

        Args:
            data (pd.DataFrame): DataFrame containing processed dialogues with columns:
                - context: Previous conversation history
                - gold_counselor_response: Reference counselor response
                - topic: Type of conversation (e.g., "Anxiety", "Workplace")
                - dialogue_type: Numerical type identifier

        Returns:
            List[Instance]: List of processed instances ready for evaluation
        """
        instances: List[Instance] = []

        for _, row in data.iterrows():
            # Format input with clear section breaks and instructions
            input_text = (
                f"Topic: {row['topic']}\n"
                f"Type: {row['dialogue_type']}\n"
                f"\nPrevious conversation:\n{row['context']}\n"
                f"\nGenerate an empathetic and appropriate counselor response:"
            )

            # Create input with empty passage since all context is in question
            prompt = PassageQuestionInput(passage="", question=input_text)

            # Create instance with gold standard response
            instance = Instance(
                input=prompt,
                references=[Reference(Output(text=row["gold_counselor_response"]), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load and process the mental health dialogue dataset.

        Args:
            output_path (str): Path for any cached or intermediate files

        Returns:
            List[Instance]: List of processed instances for evaluation
        """
        # Load the processed dialogue data
        data_path = "/share/pi/nigam/data/medhelm/mental_health/processed_dialogues.csv"
        dialogue_data = pd.read_csv(data_path)

        # Process into instances
        instances = self.process_dialogue_data(dialogue_data)

        return instances
