import ast
import random
from pathlib import Path
from typing import List, Any

import datasets
from datasets import load_dataset

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    Output,
    Input,
)


class TaskType:
    SLTC = "SingleLabelTextClassification"
    MLTC = "MultiLabelTextClassification"
    NER = "NamedEntityRecognition"
    QA = "QuestionAnswering"


BRAZILIAN_COURT_DECISIONS_JUDGMENT = "brazilian_court_decisions_judgment"
BRAZILIAN_COURT_DECISIONS_UNANIMITY = "brazilian_court_decisions_unanimity"
GERMAN_ARGUMENT_MINING = "german_argument_mining"
GREEK_LEGAL_CODE_CHAPTER = "greek_legal_code_chapter"
GREEK_LEGAL_CODE_SUBJECT = "greek_legal_code_subject"
GREEK_LEGAL_CODE_VOLUME = "greek_legal_code_volume"
SWISS_JUDGMENT_PREDICTION = "swiss_judgment_prediction"
ONLINE_TERMS_OF_SERVICE_UNFAIRNESS_LEVELS = "online_terms_of_service_unfairness_levels"
ONLINE_TERMS_OF_SERVICE_CLAUSE_TOPICS = "online_terms_of_service_clause_topics"
COVID19_EMERGENCY_EVENT = "covid19_emergency_event"
MULTI_EURLEX_LEVEL_1 = "multi_eurlex_level_1"
MULTI_EURLEX_LEVEL_2 = "multi_eurlex_level_2"
MULTI_EURLEX_LEVEL_3 = "multi_eurlex_level_3"
GREEK_LEGAL_NER = "greek_legal_ner"
LEGALNERO = "legalnero"
LENER_BR = "lener_br"
MAPA_COARSE = "mapa_coarse"
MAPA_FINE = "mapa_fine"
TASK_CODE_MAPPING = {
    BRAZILIAN_COURT_DECISIONS_JUDGMENT: TaskType.SLTC,
    BRAZILIAN_COURT_DECISIONS_UNANIMITY: TaskType.SLTC,
    GERMAN_ARGUMENT_MINING: TaskType.SLTC,
    GREEK_LEGAL_CODE_CHAPTER: TaskType.SLTC,
    GREEK_LEGAL_CODE_SUBJECT: TaskType.SLTC,
    GREEK_LEGAL_CODE_VOLUME: TaskType.SLTC,
    SWISS_JUDGMENT_PREDICTION: TaskType.SLTC,
    ONLINE_TERMS_OF_SERVICE_UNFAIRNESS_LEVELS: TaskType.SLTC,
    ONLINE_TERMS_OF_SERVICE_CLAUSE_TOPICS: TaskType.MLTC,
    COVID19_EMERGENCY_EVENT: TaskType.MLTC,
    MULTI_EURLEX_LEVEL_1: TaskType.MLTC,
    MULTI_EURLEX_LEVEL_2: TaskType.MLTC,
    MULTI_EURLEX_LEVEL_3: TaskType.MLTC,
    GREEK_LEGAL_NER: TaskType.NER,
    LEGALNERO: TaskType.NER,
    LENER_BR: TaskType.NER,
    MAPA_COARSE: TaskType.NER,
    MAPA_FINE: TaskType.NER,
}


def get_lextreme_task_type(subset):
    return TASK_CODE_MAPPING[subset]


TASK_MAX_TRAIN_INSTANCES_MAPPING = {
    BRAZILIAN_COURT_DECISIONS_JUDGMENT: 4,  # ~ max 1024 tokens
    BRAZILIAN_COURT_DECISIONS_UNANIMITY: 4,  # ~ max 1024 tokens
    GERMAN_ARGUMENT_MINING: 5,  # ~ max 256 tokens
    GREEK_LEGAL_CODE_CHAPTER: 1,  # ~ max 4096 tokens
    GREEK_LEGAL_CODE_SUBJECT: 1,  # ~ max 4096 tokens
    GREEK_LEGAL_CODE_VOLUME: 1,  # ~ max 4096 tokens
    SWISS_JUDGMENT_PREDICTION: 2,  # ~ max 2048 tokens
    ONLINE_TERMS_OF_SERVICE_UNFAIRNESS_LEVELS: 5,  # ~ max 256 tokens
    ONLINE_TERMS_OF_SERVICE_CLAUSE_TOPICS: 5,  # ~ max 256 tokens
    COVID19_EMERGENCY_EVENT: 5,  # ~ max 256 tokens
    MULTI_EURLEX_LEVEL_1: 1,  # ~ max 4096 tokens
    MULTI_EURLEX_LEVEL_2: 1,  # ~ max 4096 tokens
    MULTI_EURLEX_LEVEL_3: 1,  # ~ max 4096 tokens
    GREEK_LEGAL_NER: 5,  # ~ max 512 tokens
    LEGALNERO: 5,  # ~ max 512 tokens
    LENER_BR: 5,  # ~ max 512 tokens
    MAPA_COARSE: 5,  # ~ max 512 tokens
    MAPA_FINE: 5,  # ~ max 512 tokens
}


def get_lextreme_max_train_instances(subset):
    return TASK_MAX_TRAIN_INSTANCES_MAPPING[subset]


TASK_MAX_TOKENS_MAPPING = {
    BRAZILIAN_COURT_DECISIONS_JUDGMENT: 5,  # one word
    BRAZILIAN_COURT_DECISIONS_UNANIMITY: 5,  # one word
    GERMAN_ARGUMENT_MINING: 5,  # one word
    GREEK_LEGAL_CODE_CHAPTER: 20,  # few non-ASCII words
    GREEK_LEGAL_CODE_SUBJECT: 20,  # few non-ASCII words
    GREEK_LEGAL_CODE_VOLUME: 20,  # few non-ASCII words
    SWISS_JUDGMENT_PREDICTION: 5,  # one word
    ONLINE_TERMS_OF_SERVICE_UNFAIRNESS_LEVELS: 10,  # two words
    ONLINE_TERMS_OF_SERVICE_CLAUSE_TOPICS: 10,  # max two words
    COVID19_EMERGENCY_EVENT: 10,  # max two words
    MULTI_EURLEX_LEVEL_1: 10,  # max two words
    MULTI_EURLEX_LEVEL_2: 10,  # max two words
    MULTI_EURLEX_LEVEL_3: 10,  # max two words
    GREEK_LEGAL_NER: 430,  # num NER labels: max 2593, 99% 215, 95% 101 ==> 215 * 2 = 430
    LEGALNERO: 788,  # num NER labels: max 737, 99% 394, 95% 103 ==> 394 * 2 = 788
    LENER_BR: 338,  # num NER labels: max 654, 99% 169, 95% 100 ==> 169 * 2 = 338
    MAPA_COARSE: 274,  # num NER labels: max 367, 99% 137, 95% 83 ==> 137 * 2 = 274
    MAPA_FINE: 274,  # num NER labels: max 367, 99% 137, 95% 83 ==> 137 * 2 = 274
}


def get_lextreme_max_tokens(subset):
    return TASK_MAX_TOKENS_MAPPING[subset]


INSTRUCTIONS = {
    BRAZILIAN_COURT_DECISIONS_JUDGMENT: "In this task, you are given the case description "
    "from a decision heard at the State Supreme Court of Alagoas (Brazil). "
    "Predict the judgment of the case "
    "(no: The appeal was denied, "
    "partial: For partially favourable decisions, "
    "yes: For fully favourable decisions)",
    BRAZILIAN_COURT_DECISIONS_UNANIMITY: "In this task, you are given the case description "
    "from a decision heard at the State Supreme Court of Alagoas (Brazil). "
    "Predict the unanimity of the case (unanimity, not-unanimity, not_determined)",
    GERMAN_ARGUMENT_MINING: "In this task, you are given sentences from German court decisions. "
    "Predict the major component of German Urteilsstil "
    "(conclusion: Overall result, "
    "definition: Abstract legal facts and consequences, "
    "subsumption: Determination sentence / Concrete facts, "
    "other: Anything else)",
    GREEK_LEGAL_CODE_CHAPTER: "In this task, you are given a Greek legislative document. "
    "Predict the chapter level category of the "
    "'Permanent Greek Legislation Code - Raptarchis (Ραπτάρχης)' the document belongs to.",
    GREEK_LEGAL_CODE_SUBJECT: "In this task, you are given a Greek legislative document. "
    "Predict the subject level category of the "
    "'Permanent Greek Legislation Code - Raptarchis (Ραπτάρχης)' the document belongs to.",
    GREEK_LEGAL_CODE_VOLUME: "In this task, you are given a Greek legislative document. "
    "Predict the volume level category of the "
    "'Permanent Greek Legislation Code - Raptarchis (Ραπτάρχης)' the document belongs to.",
    SWISS_JUDGMENT_PREDICTION: "In this task, you are given the facts description "
    "from a decision heard at the Swiss Federal Supreme Court. "
    "Predict the judgment of the case (approval: The appeal was approved, or dismissal: The appeal was denied)",
    ONLINE_TERMS_OF_SERVICE_UNFAIRNESS_LEVELS: "In this task, you are given a sentence "
    "from a Terms of Service (ToS) document. "
    "Predict the unfairness level of the sentence (potentially_unfair, clearly_unfair, clearly_fair, untagged)",
    ONLINE_TERMS_OF_SERVICE_CLAUSE_TOPICS: "In this task, you are given a sentence "
    "from a Terms of Service (ToS) document. "
    "Predict the clause topics of the sentence out of the following: "
    "0: Arbitration, "
    "1: Unilateral change, "
    "2: Content removal, "
    "3: Jurisdiction, "
    "4: Choice of law, "
    "5: Limitation of liability, "
    "6: Unilateral termination, "
    "7: Contract by using, "
    "8: Privacy included. "
    "If there is no label reply n/a, if there are multiple labels specify all of them separated by a comma.",
    COVID19_EMERGENCY_EVENT: "In this task, you are given a sentence from a European legislative document. "
    "Predict the applicable measurements against COVID-19 out of the following: "
    "0: State of Emergency, "
    "1: Restrictions of fundamental rights and civil liberties, "
    "2: Restrictions of daily liberties, "
    "3: Closures / lockdown, "
    "4: Suspension of international cooperation and commitments, "
    "5: Police mobilization, "
    "6: Army mobilization, "
    "7: Government oversight. "
    "If there is no label reply n/a, if there are multiple labels specify all of them separated by a comma.",
    MULTI_EURLEX_LEVEL_1: "In this task, you are given a document from an EU law. "
    "Predict the level 1 concept in the EUROVOC taxonomy. "
    "If there is no label reply n/a, if there are multiple labels specify all of them separated by a comma.",
    MULTI_EURLEX_LEVEL_2: "In this task, you are given a document from an EU law. "
    "Predict the level 2 concept in the EUROVOC taxonomy. "
    "If there is no label reply n/a, if there are multiple labels specify all of them separated by a comma.",
    MULTI_EURLEX_LEVEL_3: "In this task, you are given a document from an EU law. "
    "Predict the level 3 concept in the EUROVOC taxonomy. "
    "If there is no label reply n/a, if there are multiple labels specify all of them separated by a comma.",
    GREEK_LEGAL_NER: "In this task, you are given a sentence from Greek legislation. "
    "Predict the named entity type for each token out of the following: "
    "O, B-ORG, I-ORG, B-GPE, I-GPE, B-LEG-REFS, I-LEG-REFS, B-PUBLIC-DOCS, I-PUBLIC-DOCS, B-PERSON, I-PERSON, "
    "B-FACILITY, I-FACILITY, B-LOCATION-UNK, I-LOCATION-UNK, B-LOCATION-NAT, I-LOCATION-NAT",
    LEGALNERO: "In this task, you are given a sentence from Romanian legislation. "
    "Predict the named entity type for each token out of the following: "
    "O, B-TIME, I-TIME, B-LEGAL, I-LEGAL, B-ORG, I-ORG, B-LOC, I-LOC, B-PER, I-PER",
    LENER_BR: "In this task, you are given a sentence "
    "from Brazilian legal documents (court decisions and legislation). "
    "Predict the named entity type for each token out of the following: "
    "O, B-ORGANIZACAO, I-ORGANIZACAO, B-PESSOA, I-PESSOA, B-TEMPO, I-TEMPO, B-LOCAL, I-LOCAL, "
    "B-LEGISLACAO, I-LEGISLACAO, B-JURISPRUDENCIA, I-JURISPRUDENCIA",
    MAPA_COARSE: "In this task, you are given a sentence from the EUR-Lex database. "
    "Predict the coarse grained named entity type for each token out of the following: "
    "O, B-ORGANISATION, I-ORGANISATION, B-ADDRESS, I-ADDRESS, B-DATE, I-DATE, "
    "B-PERSON, I-PERSON, B-AMOUNT, I-AMOUNT, B-TIME, I-TIME",
    MAPA_FINE: "In this task, you are given a sentence from the EUR-Lex database. "
    "Predict the fine grained named entity type for each token out of the following: "
    "O, B-BUILDING, I-BUILDING, B-CITY, I-CITY, B-COUNTRY, I-COUNTRY, B-PLACE, I-PLACE, B-TERRITORY, I-TERRITORY, "
    "I-UNIT, B-UNIT, B-VALUE, I-VALUE, B-YEAR, I-YEAR, B-STANDARD ABBREVIATION, I-STANDARD ABBREVIATION, "
    "B-MONTH, I-MONTH, B-DAY, I-DAY, B-AGE, I-AGE, B-ETHNIC CATEGORY, I-ETHNIC CATEGORY, B-FAMILY NAME, I-FAMILY NAME, "
    "B-INITIAL NAME, I-INITIAL NAME, B-MARITAL STATUS, I-MARITAL STATUS, B-PROFESSION, I-PROFESSION, B-ROLE, I-ROLE, "
    "B-NATIONALITY, I-NATIONALITY, B-TITLE, I-TITLE, B-URL, I-URL, B-TYPE, I-TYPE",
}


def get_lextreme_instructions(subset):
    return INSTRUCTIONS[subset]


class LEXTREMEScenario(Scenario):
    """
    The dataset consists of 11 diverse multilingual legal NLU tasks.
    6 tasks have one single configuration and 5 tasks have two or three configurations.
    This leads to a total of 18 tasks (8 single-label text classification tasks,
    5 multi-label text classification tasks and 5 token-classification tasks).
    Find more information on the dataset here: https://huggingface.co/datasets/joelito/lextreme

    We prompt models using the following format (example for german_argument_mining)

        <sentence>
        Urteilsstil:

        Target completion:
            <sentence> (<sentence>:conclusion, subsumption, definition or other)

    Using an example from the training dataset, we have

    ```
    Die Klage ist hinsichtlich der begehrten „Umzugkosten“ und hinsichtlich der begehrten
    „Übernahme der durch den Rechtsstreit gegen das Jobcenter verursachten tatsächlichen Kosten“ insgesamt unzulässig.

    Urteilsstil:
    Target completion:
        conclusion
    ```

    """

    name = "lextreme"
    description = "Multilingual Legal Text Classification and Named Entity Recognition dataset."
    tags = ["single_label_text_classification", "multi_label_text_classification", "named_entity_recognition"]

    # Mapping from HELM splits to HF splits
    splits_mapping = {
        TRAIN_SPLIT: datasets.Split.TRAIN,
        VALID_SPLIT: datasets.Split.VALIDATION,
        TEST_SPLIT: datasets.Split.TEST,
    }

    dataset_name = "joelito/lextreme"
    max_number_of_wrong_answers = 30
    delimiter = '" "'  # we choose quotes and whitespace as a delimiter because this is what worked for gpt3

    ner_class_mapping = {
        LENER_BR: [
            "O",
            "B-ORGANIZACAO",
            "I-ORGANIZACAO",
            "B-PESSOA",
            "I-PESSOA",
            "B-TEMPO",
            "I-TEMPO",
            "B-LOCAL",
            "I-LOCAL",
            "B-LEGISLACAO",
            "I-LEGISLACAO",
            "B-JURISPRUDENCIA",
            "I-JURISPRUDENCIA",
        ],
        LEGALNERO: [
            "O",
            "B-TIME",
            "I-TIME",
            "B-LEGAL",
            "I-LEGAL",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
            "B-PER",
            "I-PER",
        ],
        GREEK_LEGAL_NER: [
            "O",
            "B-ORG",
            "I-ORG",
            "B-GPE",
            "I-GPE",
            "B-LEG-REFS",
            "I-LEG-REFS",
            "B-PUBLIC-DOCS",
            "I-PUBLIC-DOCS",
            "B-PERSON",
            "I-PERSON",
            "B-FACILITY",
            "I-FACILITY",
            "B-LOCATION-UNK",
            "I-LOCATION-UNK",
            "B-LOCATION-NAT",
            "I-LOCATION-NAT",
        ],
        MAPA_COARSE: [
            "O",
            "B-ORGANISATION",
            "I-ORGANISATION",
            "B-ADDRESS",
            "I-ADDRESS",
            "B-DATE",
            "I-DATE",
            "B-PERSON",
            "I-PERSON",
            "B-AMOUNT",
            "I-AMOUNT",
            "B-TIME",
            "I-TIME",
        ],
        MAPA_FINE: [
            "O",
            "B-BUILDING",
            "I-BUILDING",
            "B-CITY",
            "I-CITY",
            "B-COUNTRY",
            "I-COUNTRY",
            "B-PLACE",
            "I-PLACE",
            "B-TERRITORY",
            "I-TERRITORY",
            "I-UNIT",
            "B-UNIT",
            "B-VALUE",
            "I-VALUE",
            "B-YEAR",
            "I-YEAR",
            "B-STANDARD ABBREVIATION",
            "I-STANDARD ABBREVIATION",
            "B-MONTH",
            "I-MONTH",
            "B-DAY",
            "I-DAY",
            "B-AGE",
            "I-AGE",
            "B-ETHNIC CATEGORY",
            "I-ETHNIC CATEGORY",
            "B-FAMILY NAME",
            "I-FAMILY NAME",
            "B-INITIAL NAME",
            "I-INITIAL NAME",
            "B-MARITAL STATUS",
            "I-MARITAL STATUS",
            "B-PROFESSION",
            "I-PROFESSION",
            "B-ROLE",
            "I-ROLE",
            "B-NATIONALITY",
            "I-NATIONALITY",
            "B-TITLE",
            "I-TITLE",
            "B-URL",
            "I-URL",
            "B-TYPE",
            "I-TYPE",
        ],
    }

    def __init__(self, subset: str):
        super().__init__()
        assert subset in list(TASK_CODE_MAPPING.keys()) + ["all"], f"Unknown subset: {subset}"
        self.subsets = [subset] if subset != "all" else list(TASK_CODE_MAPPING.keys())
        self.random: random.Random = random.Random(42)

    def get_instances_for_subset(self, config: str, output_path: str) -> List[Instance]:
        task_code = TASK_CODE_MAPPING[config]
        # Load dataset
        cache_dir = str(Path(output_path) / "data")
        dataset: Any = load_dataset(self.dataset_name, config, cache_dir=cache_dir)

        if task_code == TaskType.SLTC:
            class_label = dataset["train"].features["label"]
            label_classes = class_label.names
        elif task_code == TaskType.MLTC:
            # construct the label classes
            label_classes = set()
            for split in self.splits_mapping.values():
                for example in dataset[split]:
                    label_classes |= set(example["label"])  # add all new labels to the set
            label_classes = sorted(list(map(str, label_classes)))  # convert everything to a string
        elif task_code == TaskType.NER:
            label_classes = self.ner_class_mapping[config]

        def generate_instance(example, split: str):
            # get correct labels
            if task_code == TaskType.SLTC:
                correct_label = class_label.int2str(example["label"])  # get label name for correct label
                correct_labels = correct_label if isinstance(correct_label, list) else [correct_label]
            elif task_code == TaskType.MLTC:
                correct_labels = list(map(str, example["label"]))  # here we don't have any mapping to label names
            elif task_code == TaskType.NER:
                correct_labels = [label_classes[label] for label in example["label"]]

            # construct wrong references
            wrong_references = []
            if task_code in [TaskType.SLTC, TaskType.MLTC]:
                for label_name in label_classes:
                    if label_name not in correct_labels:
                        wrong_reference = Reference(output=Output(label_name), tags=[])  # Wrong output
                        wrong_references.append(wrong_reference)
            elif task_code == TaskType.NER:
                if len(set(correct_labels)) > 1:  # make sure that the correct labels are not only 'O's
                    for label_name in label_classes:
                        if label_name not in correct_labels and label_name != "O":
                            # just replace the non-'O' labels with the new label_name for a fake example
                            new_labels = [label_name if label != "O" else label for label in correct_labels]
                            wrong_reference = Reference(
                                output=Output(construct_ner_sequence(new_labels)), tags=[]
                            )  # Wrong output
                            wrong_references.append(wrong_reference)

            wrong_references = reduce_wrong_reference_count(wrong_references)

            # construct correct references and input
            if task_code in [TaskType.SLTC, TaskType.MLTC]:
                input_text = example["input"]
                if "multi_eurlex" in config:
                    input_text = ast.literal_eval(input_text)
                    assert isinstance(input_text, dict)
                    languages = list([lang for lang, content in input_text.items() if content is not None])
                    input_text = input_text[self.random.choice(languages)]  # just choose a random language
                correct_references = [
                    Reference(output=Output(correct_label), tags=[CORRECT_TAG]) for correct_label in correct_labels
                ]  # for MLTC we have multiple correct ones
            elif task_code == TaskType.NER:
                input_text = construct_ner_sequence(example["input"])
                correct_references = [
                    Reference(output=Output(construct_ner_sequence(correct_labels)), tags=[CORRECT_TAG])
                ]
            return Instance(input=Input(input_text), references=wrong_references + correct_references, split=split)

        def construct_ner_sequence(ner_list):
            return '"' + self.delimiter.join(ner_list) + '"'

        def reduce_wrong_reference_count(wrong_references):
            self.random.shuffle(wrong_references)  # shuffle wrong references
            if len(wrong_references) > self.max_number_of_wrong_answers:
                # if there are too many wrong references, only take a subset
                wrong_references = wrong_references[: self.max_number_of_wrong_answers]
            return wrong_references

        def generate_instances(split: str):
            split_dataset = dataset[self.splits_mapping[split]]
            return [generate_instance(example, split) for example in split_dataset]

        return generate_instances(TRAIN_SPLIT) + generate_instances(VALID_SPLIT) + generate_instances(TEST_SPLIT)

    def get_instances(self, output_path: str) -> List[Instance]:
        instances = []
        for subset in self.subsets:
            instances.extend(self.get_instances_for_subset(subset, output_path))
        return instances
