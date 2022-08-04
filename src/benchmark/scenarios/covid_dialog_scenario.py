import os
import random
import re
from typing import List

import cleantext

from common.general import ensure_directory_exists, ensure_file_downloaded, hlog
from .scenario import Scenario, Instance, Reference, ALL_SPLITS, CORRECT_TAG, VALID_SPLIT


class COVIDDialogScenario(Scenario):
    """
    From https://github.com/UCSD-AI4H/COVID-Dialogue, "COVID-Dialogue-Dataset-English is an English medical dialogue
    dataset about COVID-19 and other types of pneumonia. Patients who are concerned that they may be infected by
    COVID-19 or other pneumonia consult doctors and doctors provide advice. There are 603 consultations. Each
    consultation consists of ID, URL, Description of patient’s medical condition and Dialogue."

    The following is an example from the dataset:

    Description
    I have cough with no travel history. Is this a symptom of Covid-19?

    Dialogue
    Patient:
    Hello doctor, I get a cough for the last few days, which is heavy during night times. No raise in temperature but
    feeling tired with no travel history. No contact with any Covid-19 persons. It has been four to five days and has
    drunk a lot of Benadryl and took Paracetamol too. Doctors have shut the OP so do not know what to do? Please help.
    Doctor:
    Hello, I understand your concern. I just have a few more questions.
    Does your cough has phlegm? Any other symptoms like difficulty breathing? Any other medical condition such as
    asthma, hypertension? Are you a smoker? Alcoholic beverage drinker?
    Patient:
    Thank you doctor,
    I have phlegm but not a lot. A tiny amount comes out most of the time. I have no difficulty in breathing. No
    medical conditions and not a smoker nor a drinker.
    Doctor:
    Hi, I would recommend you take n-acetylcysteine 200 mg powder dissolved in water three times a day. You may also
    nebulize using PNSS (saline nebulizer) three times a day. This will help the phlegm to come out. I would also
    recommend you take vitamin C 500 mg and zinc to boost your immune system. If symptoms persist, worsen or new onset
    of symptoms has been noted, further consult is advised.


    - How is the covid dialog in mistral constructed @michi

    Some edge cases to consider:
    - Multiple doctors can respond
    - We should throw out questions that are just "Do I have covid 19?" or "I have covid 19 symptoms?". The responses are
      "Without any details it's impossible to say."
    - Some answers by doctors are not helpful: "Doctor: Maybe. Do video w/"
    - Some doctors end their answer with "Would you like to video or text chat with me?".
    - Some doctors included their name in their response: "Dr Bhagyesh V. Patel, General Surgeon"
    - Should we include the description from the original dataset?
    - A lot of typos in patient's questions and some typos in doctor's answers? Should we construct a 'clean' version?
    - all lowercase

    @article{ju2020CovidDialog,
      title={CovidDialog: Medical Dialogue Datasets about COVID-19},
      author={Ju, Zeqian and Chakravorty, Subrato and He, Xuehai and Chen, Shu and Yang, Xingyi and Xie, Pengtao},
      journal={ https://github.com/UCSD-AI4H/COVID-Dialogue},
      year={2020}
    }
    """

    name = "covid_dialog"
    description = "Medical dialogue dataset of conversations between doctors and patients on their COVID-19 concerns"
    tags = ["dialogue", "biomedical"]

    # TODO: use our clean version -Tony
    DOWNLOAD_URL: str = (
        "https://raw.githubusercontent.com/UCSD-AI4H/COVID-Dialogue/master/COVID-Dialogue-Dataset-English.txt"
    )

    def __init__(self):
        pass

    def get_instances(self) -> List[Instance]:
        """
        Adapted from the official download script in the GitHub repository:
        https://github.com/UCSD-AI4H/COVID-Dialogue.
        """
        data_path: str = os.path.join(self.output_path, "data")
        ensure_directory_exists(data_path)
        dataset_path: str = os.path.join(data_path, "COVID-Dialogue-Dataset-English.txt")
        ensure_file_downloaded(source_url=COVIDDialogScenario.DOWNLOAD_URL, target_path=dataset_path)

        dialogues = []
        for text in open(dataset_path).read().split("id="):
            text = text[text.find("Description") :].strip()
            description: str = text[len("Description\n") : text.find("\nDialogue")]
            if description:
                description = cleantext.clean(description, extra_spaces=True, lowercase=True)

            text = text[text.find("\nPatient:") :]

            utterances, last_person, valid = [], "None", True
            for x in re.finditer("Doctor:|Patient:", text):
                if x.group() == last_person:
                    valid = False
                    break
                else:
                    last_person = x.group()

                utterance = text[x.end() :].split("Patient:")[0].split("Doctor:")[0]
                utterances.append(cleantext.clean(utterance, extra_spaces=True, lowercase=True))

            if valid and utterances:
                dialogues.append({"description": description, "utterances": utterances})

        hlog(f"Number of dialogs: {len(dialogues)}")

        # Used the exact same seed value in the official download script
        random.seed(11111)
        random.shuffle(dialogues)

        train_size = int(0.8 * len(dialogues))
        dev_size = int(0.1 * len(dialogues))
        train_split = dialogues[:train_size]
        dev_split = dialogues[train_size : train_size + dev_size]
        test_split = dialogues[train_size + dev_size :]

        import pdb

        pdb.set_trace()
        instances: List[Instance] = []

        return instances
