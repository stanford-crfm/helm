import json
import os
from typing import List

from helm.common.general import ensure_directory_exists, ensure_file_downloaded
from helm.benchmark.scenarios.scenario import Scenario, Instance, Reference, ALL_SPLITS, CORRECT_TAG, Input, Output


class MedDialogScenario(Scenario):
    """
    "The MedDialog dataset (English) contains conversations between doctors and patients.
    It has 0.26 million dialogues. The data is continuously growing and more dialogues will be added.
    The raw dialogues are from healthcaremagic.com and icliniq.com. All copyrights of the data belong
    to healthcaremagic.com and icliniq.com."

    The following is an example from the healthcaremagic.com subset:

    Patient: I get cramps on top of my left forearm and hand and it causes my hand and fingers to draw up and it
    hurts. It mainly does this when I bend my arm. I ve been told that I have a slight pinch in a nerve in my neck.
    Could this be a cause? I don t think so. Doctor: Hi there. It may sound difficult to believe it ,but the nerves
    which supply your forearms and hand, start at the level of spinal cord and on their way towards the forearm and
    hand regions which they supply, the course of these nerves pass through difference fascial and muscular planes
    that can make them susceptible to entrapment neuropathies. Its a group of conditions where a nerve gets
    compressed between a muscle and a bone, or between the fibers of a muscle that it pierces or passes through.
    Also, the compression can happen when the nerves are travelling around a blood vessel which can mechanically put
    pressure on them. Usually patients who would be having such a problem present with a dull aching pain over the
    arm and forearm. If it is not too severe and does not cause any neurological deficits then conservative management
    with Pregabalin and Vitamin B complex tablets, activity modifications and physiotherapy can be started which
    will provide relief. Avoid the activities which exaggerate your problem.

    Could painful forearms be related to pinched nerve in neck?


    The following is an example from the icliniq.com subset:

    Patient: Hello doctor,  We are looking for a second opinion on my friend's MRI scan of both the knee joints as he
    is experiencing excruciating pain just above the patella. He has a sudden onset of severe pain on both the knee
    joints about two weeks ago. Previously he had a similar episode about two to three months ago and it subsided
    after resting and painkillers. Doctor: Hi. I viewed the right and left knee MRI images. (attachment removed to
    protect patient identity).  Left knee: The MRI, left knee joint shows a complex tear in the posterior horn of the
    medial meniscus area and mild left knee joint effusion. There is some fluid between the semimembranous and medial
    head of gastrocnemius muscles. There is a small area of focal cartilage defect in the upper pole of the patella
    with mild edematous fat. The anterior and posterior cruciate ligaments are normal. The medial and lateral
    collateral ligaments are normal. Right knee: The right knee joint shows mild increased signal intensity in the
    posterior horn of the medial meniscus area and minimal knee joint effusion. There is minimal fluid in the back
    of the lower thigh and not significant. There is a suspicious strain in the left anterior cruciate ligament
    interiorly but largely the attachments are normal. The posterior cruciate ligament is normal. There are subtle
    changes in the upper pole area of the right patella and mild edema. There is mild edema around the bilateral
    distal quadriceps tendons, but there is no obvious tear of the tendons.

    My friend has excruciating knee pain. Please interpret his MRI report


    Paper: https://arxiv.org/abs/2004.03329
    Code: https://github.com/UCSD-AI4H/Medical-Dialogue-System

    @article{chen2020meddiag,
      title={MedDialog: a large-scale medical dialogue dataset},
      author={Chen, Shu and Ju, Zeqian and Dong, Xiangyu and Fang, Hongchao and Wang, Sicheng and Yang, Yue and Zeng,
              Jiaqi and Zhang, Ruisi and Zhang, Ruoyu and Zhou, Meng and Zhu, Penghui and Xie, Pengtao},
      journal={arXiv preprint arXiv:2004.03329},
      year={2020}
    }

    We used the data preprocessing from "BioBART: Pretraining and Evaluation o A Biomedical Generative Language Model"
    (Yuan et al.) and generated the following splits:

    |Dataset         | Train      | Valid   | Test   |
    |--------------- |------------|---------|--------|
    |HealthCareMagic | 181,122    | 22,641  | 22,642 |
    |iCliniq         | 24,851     | 3,105   | 3,108  |

    Yuan et al. described, "HealthCareMagic's summaries are more abstractive and are written in a formal style,
    unlike iCliniq's patient-written summaries."

    Paper: https://arxiv.org/abs/2204.03905
    Code: https://github.com/GanjinZero/BioBART

    @misc{https://doi.org/10.48550/arxiv.2204.03905,
      doi = {10.48550/ARXIV.2204.03905},
      url = {https://arxiv.org/abs/2204.03905},
      author = {Yuan, Hongyi and Yuan, Zheng and Gan, Ruyi and Zhang, Jiaxing and Xie, Yutao and Yu, Sheng},
      keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences,
                  FOS: Computer and information sciences},
      title = {BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model},
      publisher = {arXiv},
      year = {2022},
      copyright = {arXiv.org perpetual, non-exclusive license}
    }
    """

    name = "med_dialog"
    description = "A collection of doctor-patient conversations with corresponding summaries."
    tags = ["dialogue", "biomedical"]

    def __init__(self, subset: str):
        super().__init__()
        assert subset in ["healthcaremagic", "icliniq"], f"Invalid subset specified for {self.name}: {subset}."
        self.subset: str = subset

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path: str = os.path.join(output_path, self.subset)
        ensure_directory_exists(data_path)

        instances: List[Instance] = []

        for split in ALL_SPLITS:
            # Limit to zero shot setting
            if split == "test":
                split_file_name: str = f"{split}.json"
                split_path: str = os.path.join(data_path, split_file_name)
                ensure_file_downloaded(
                    source_url="https://worksheets.codalab.org/rest/bundles/0x82f0c47f6d3e4462ae9ef8ea39eebe64/"
                    f"contents/blob/{self.subset}/{split_file_name}",
                    target_path=split_path,
                    unpack=False,
                )

                with open(split_path, "r") as f:
                    examples: List = json.load(f)["data"]
                    for example in examples:
                        instances.append(
                            Instance(
                                input=Input(text=example["src"]),
                                references=[Reference(Output(text=example["tgt"]), tags=[CORRECT_TAG])],
                                split=split,
                            )
                        )

        return instances
