import random
from pathlib import Path
from typing import List

import datasets
from datasets import load_dataset

from .scenario import Scenario, Instance, Reference, CORRECT_TAG, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT

# SLTC: Single Class Text Classification
# MLTC: Multi Class Text Classification
# NER: Named Entity Recognition
task_code_mapping = {
    'brazilian_court_decisions_judgment': 'SLTC',
    'brazilian_court_decisions_unanimity': 'SLTC',
    'german_argument_mining': 'SLTC',
    'greek_legal_code_chapter_level': 'SLTC',
    'greek_legal_code_subject_level': 'SLTC',
    'greek_legal_code_volume_level': 'SLTC',
    'swiss_judgment_prediction': 'SLTC',
    'online_terms_of_service_unfairness_levels': 'SLTC',
    'online_terms_of_service_clause_topics': 'MLTC',
    'covid19_emergency_event': 'MLTC',
    'multi_eurlex_level_1': 'MLTC',
    'multi_eurlex_level_2': 'MLTC',
    'multi_eurlex_level_3': 'MLTC',
    'greek_legal_ner': 'NER',
    'legalnero': 'NER',
    'lener_br': 'NER',
    'mapa_ner_coarse_grained': 'NER',
    'mapa_ner_fine_grained': 'NER',
}

task_max_train_instances_mapping = {
    'brazilian_court_decisions_judgment': 4,  # ~ max 1024 tokens
    'brazilian_court_decisions_unanimity': 4,  # ~ max 1024 tokens
    'german_argument_mining': 5,  # ~ max 256 tokens
    'greek_legal_code_chapter_level': 1,  # ~ max 4096 tokens
    'greek_legal_code_subject_level': 1,  # ~ max 4096 tokens
    'greek_legal_code_volume_level': 1,  # ~ max 4096 tokens
    'swiss_judgment_prediction': 2,  # ~ max 2048 tokens
    'online_terms_of_service_unfairness_levels': 5,  # ~ max 256 tokens
    'online_terms_of_service_clause_topics': 5,  # ~ max 256 tokens
    'covid19_emergency_event': 5,  # ~ max 256 tokens
    'multi_eurlex_level_1': 1,  # ~ max 4096 tokens
    'multi_eurlex_level_2': 1,  # ~ max 4096 tokens
    'multi_eurlex_level_3': 1,  # ~ max 4096 tokens
    'greek_legal_ner': 5,  # ~ max 512 tokens
    'legalnero': 5,  # ~ max 512 tokens
    'lener_br': 5,  # ~ max 512 tokens
    'mapa_ner_coarse_grained': 5,  # ~ max 512 tokens
    'mapa_ner_fine_grained': 5,  # ~ max 512 tokens
}


def get_lextreme_max_train_instances(subset):
    return task_max_train_instances_mapping[subset]


instructions = {
    "brazilian_court_decisions_judgment": "In this task, you are given the case description from a decision heard at the State Supreme Court of Alagoas (Brazil). "
                                          "Predict the judgment of the case "
                                          "(no: The appeal was denied, "
                                          "partial: For partially favourable decisions, "
                                          "yes: For fully favourable decisions)",
    "brazilian_court_decisions_unanimity": "In this task, you are given the case description from a decision heard at the State Supreme Court of Alagoas (Brazil). "
                                           "Predict the unanimity of the case (unanimity, not-unanimity, not_determined)",
    "german_argument_mining": "In this task, you are given sentences from German court decisions. "
                              "Predict the major component of German Urteilsstil "
                              "(conclusion: Overall result, "
                              "definition: Abstract legal facts and consequences, "
                              "subsumption: Determination sentence / Concrete facts, "
                              "other: Anything else)",
    "greek_legal_code_chapter_level": "In this task, you are given a Greek legislative document. "
                                      "Predict the chapter level category of the 'Permanent Greek Legislation Code - Raptarchis (Ραπτάρχης)' the document belongs to.",
    "greek_legal_code_subject_level": "In this task, you are given a Greek legislative document. "
                                      "Predict the subject level category of the 'Permanent Greek Legislation Code - Raptarchis (Ραπτάρχης)' the document belongs to.",
    "greek_legal_code_volume_level": "In this task, you are given a Greek legislative document. "
                                      "Predict the volume level category of the 'Permanent Greek Legislation Code - Raptarchis (Ραπτάρχης)' the document belongs to.",
    "swiss_judgment_prediction": "In this task, you are given the facts description from a decision heard at the Swiss Federal Supreme Court. "
                                 "Predict the judgment of the case (approval or dismissal)",
    "online_terms_of_service_unfairness_levels": "In this task, you are given a sentence from a Terms of Service (ToS) document. "
                                                 "Predict the unfairness level of the sentence (potentially_unfair, clearly_unfair, clearly_fair, untagged)",
    "online_terms_of_service_clause_topics": "In this task, you are given a sentence from a Terms of Service (ToS) document. "
                                             "Predict the clause topics of the sentence "
                                             "(a: Arbitration, "
                                             "ch: Unilateral change, "
                                             "cr: Content removal, "
                                             "j: Jurisdiction, "
                                             "law: Choice of law, "
                                             "ltd: Limitation of liability, "
                                             "ter: Unilateral termination, "
                                             "use: Contract by using, "
                                             "pinc: Privacy included)",
    "covid19_emergency_event": "In this task, you are given a sentence from a European legislative document. "
                               "Predict the applicable measurements against COVID-19 "
                               "(event1: State of Emergency, "
                               "event2: Restrictions of fundamental rights and civil liberties, "
                               "event3: Restrictions of daily liberties, "
                               "event4: Closures / lockdown, "
                               "event5: Suspension of international cooperation and commitments, "
                               "event6: Police mobilization, "
                               "event7: Army mobilization, "
                               "event8: Government oversight)",
    "multi_eurlex_level_1": "In this task, you are given a document from an EU law. "
                            "Predict the level 1 concept in the EUROVOC taxonomy.",
    "multi_eurlex_level_2": "In this task, you are given a document from an EU law. "
                            "Predict the level 2 concept in the EUROVOC taxonomy.",
    "multi_eurlex_level_3": "In this task, you are given a document from an EU law. "
                            "Predict the level 3 concept in the EUROVOC taxonomy.",
    "greek_legal_ner": "In this task, you are given a sentence from Greek legislation. "
                 "Predict the named entity type for each token.",
    "legalnero": "In this task, you are given a sentence from Romanian legislation. "
                 "Predict the named entity type for each token.",
    "lener_br": "In this task, you are given a sentence from Brazilian legal documents (court decisions and legislation). "
                 "Predict the named entity type for each token.",
    "mapa_ner_coarse_grained": "In this task, you are given a sentence from the EUR-Lex database. "
                               "Predict the coarse grained named entity type for each token.",
    "mapa_ner_fine_grained": "In this task, you are given a sentence from the EUR-Lex database. "
                               "Predict the fine grained named entity type for each token.",
}


def get_lextreme_instructions(subset):
    return instructions[subset]


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
        TEST_SPLIT: datasets.Split.TEST
    }

    dataset_name = "joelito/lextreme"
    max_number_of_wrong_answers = 30
    mltc_no_label_name = 'No Label'
    delimiter = "|"  # we choose the pipe as a delimiter because it is very unlikely to occur in the data

    ner_class_mapping = {
        "leber_br": [
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
        "legalnero": [
            'O',
            'B-TIME',
            'I-TIME',
            'B-LEGAL',
            'I-LEGAL',
            'B-ORG',
            'I-ORG',
            'B-LOC',
            'I-LOC',
            'B-PER',
            'I-PER',
        ],
        "greek_legal_ner": [
            'O',
            'B-ORG',
            'I-ORG',
            'B-GPE',
            'I-GPE',
            'B-LEG-REFS',
            'I-LEG-REFS',
            'B-PUBLIC-DOCS',
            'I-PUBLIC-DOCS',
            'B-PERSON',
            'I-PERSON',
            'B-FACILITY',
            'I-FACILITY',
            'B-LOCATION-UNK',
            'I-LOCATION-UNK',
            'B-LOCATION-NAT',
            'I-LOCATION-NAT'
        ],
        "mapa_course": [
            'O',
            'B-ORGANISATION',
            'I-ORGANISATION',
            'B-ADDRESS',
            'I-ADDRESS',
            'B-DATE',
            'I-DATE',
            'B-PERSON',
            'I-PERSON',
            'B-AMOUNT',
            'I-AMOUNT',
            'B-TIME',
            'I-TIME'
        ],
        "mapa_fine": [
            'O',
            'B-BUILDING',
            'I-BUILDING',
            'B-CITY',
            'I-CITY',
            'B-COUNTRY',
            'I-COUNTRY',
            'B-PLACE',
            'I-PLACE',
            'B-TERRITORY',
            'I-TERRITORY',
            'I-UNIT',
            'B-UNIT',
            'B-VALUE',
            'I-VALUE',
            'B-YEAR',
            'I-YEAR',
            'B-STANDARD ABBREVIATION',
            'I-STANDARD ABBREVIATION',
            'B-MONTH',
            'I-MONTH',
            'B-DAY',
            'I-DAY',
            'B-AGE',
            'I-AGE',
            'B-ETHNIC CATEGORY',
            'I-ETHNIC CATEGORY',
            'B-FAMILY NAME',
            'I-FAMILY NAME',
            'B-INITIAL NAME',
            'I-INITIAL NAME',
            'B-MARITAL STATUS',
            'I-MARITAL STATUS',
            'B-PROFESSION',
            'I-PROFESSION',
            'B-ROLE',
            'I-ROLE',
            'B-NATIONALITY',
            'I-NATIONALITY',
            'B-TITLE',
            'I-TITLE',
            'B-URL',
            'I-URL',
            'B-TYPE',
            'I-TYPE',
        ],
    }

    def __init__(self, subset: str):
        assert subset in list(task_code_mapping.keys()) + ["all"], f"Unknown subset: {subset}"
        self.subsets = [subset] if subset != "all" else list(task_code_mapping.keys())
        self.random: random.Random = random.Random(42)

    def get_instances_for_subset(self, config: str) -> List[Instance]:
        task_code = task_code_mapping[config]
        # Load dataset
        cache_dir = str(Path(self.output_path) / "data")
        dataset = load_dataset(self.dataset_name, config, cache_dir=cache_dir)

        if task_code == 'SLTC':
            class_label = dataset['train'].features["label"]
            label_classes = class_label.names
        elif task_code == 'MLTC':
            # construct the label classes
            label_classes = set()
            for split in self.splits_mapping.values():
                for example in dataset[split]:
                    label_classes |= set(example["label"])  # add all new labels to the set
            label_classes = sorted(list(map(str, label_classes)))  # convert everything to a string
        elif task_code == 'NER':
            label_classes = self.ner_class_mapping[config]

        def generate_instance(example, split: str):
            # get correct labels
            if task_code == 'SLTC':
                correct_label = class_label.int2str(example['label'])  # get label name for correct label
                correct_labels = correct_label if isinstance(correct_label, list) else [correct_label]
            elif task_code == 'MLTC':
                correct_labels = list(map(str, example['label']))  # here we don't have any mapping to label names
            elif task_code == 'NER':
                correct_labels = [label_classes[label] for label in example['label']]

            # construct wrong references
            wrong_references = []
            if task_code in ['SLTC', 'MLTC']:
                for label_name in label_classes:
                    if label_name not in correct_labels:
                        wrong_reference = Reference(output=label_name, tags=[])  # Wrong output
                        wrong_references.append(wrong_reference)
            elif task_code == 'NER':
                if len(set(correct_labels)) > 1:  # make sure that the correct labels are not only 'O's
                    for label_name in label_classes:
                        if label_name not in correct_labels and label_name != 'O':
                            # just replace the non-'O' labels with the new label_name for a fake example
                            new_labels = [label_name if label != 'O' else label for label in correct_labels]
                            wrong_reference = Reference(output=self.delimiter.join(new_labels), tags=[])  # Wrong output
                            wrong_references.append(wrong_reference)

            wrong_references = reduce_wrong_reference_count(wrong_references)

            if task_code == 'MLTC':  # special case for multilabel classification tasks
                if correct_labels:  # if we have a correct label
                    # add the no_label to the wrong references
                    # IMPORTANT: add it after the reduce_wrong_reference_count call, to make sure the no label is always there
                    wrong_references.append(Reference(output=self.mltc_no_label_name, tags=[]))
                else:  # if we don't have a correct label
                    # add the no_label to the correct labels
                    correct_labels = [self.mltc_no_label_name]

            # construct correct references and input
            if task_code in ['SLTC', 'MLTC']:
                input_text = example['input']
                correct_references = [Reference(output=correct_label, tags=[CORRECT_TAG])
                                      for correct_label in correct_labels]  # for MLTC we have multiple correct ones
            elif task_code == 'NER':
                correct_references = [Reference(output=self.delimiter.join(correct_labels), tags=[CORRECT_TAG])]
                input_text = ','.join(example['input'])
            return Instance(input=input_text, references=wrong_references + correct_references, split=split)

        def reduce_wrong_reference_count(wrong_references):
            self.random.shuffle(wrong_references)  # shuffle wrong references
            if len(wrong_references) > self.max_number_of_wrong_answers:
                # if there are too many wrong references, only take a subset
                wrong_references = wrong_references[:self.max_number_of_wrong_answers]
            return wrong_references

        def generate_instances(split: str):
            split_dataset = dataset[self.splits_mapping[split]]
            return [generate_instance(example, split) for example in split_dataset]

        return generate_instances(TRAIN_SPLIT) + generate_instances(VALID_SPLIT) + generate_instances(TEST_SPLIT)

    def get_instances(self) -> List[Instance]:
        instances = []
        for subset in self.subsets:
            instances.extend(self.get_instances_for_subset(subset))
        return instances
