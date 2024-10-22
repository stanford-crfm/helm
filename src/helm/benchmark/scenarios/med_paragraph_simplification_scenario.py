import os
from typing import List

from helm.common.general import ensure_directory_exists, ensure_file_downloaded
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    ALL_SPLITS,
    CORRECT_TAG,
    VALID_SPLIT,
    Input,
    Output,
)


class MedParagraphSimplificationScenario(Scenario):
    """
    "Paragraph-level Simplification of Medical Texts" (Devaraj et al.) studies the problem of learning to simplify
    medical texts. One of their contributions is a new corpus that is composed of technical abstracts and their
    lay summaries on various clinical topics.

    The author generated train/val/test splits, which are available in the GitHub repository linked in the paper.

    The following is an example from the dataset:

    {
        "doi": "10.1002/14651858.CD011112.pub2",
        "abstract": "We included six studies (reported as seven papers) involving 326 participants whose ages ranged
        from 39 to 83 years, with a gender bias towards men (73% to 95% across studies), reflecting the characteristics
        of patients with HNC. The risk of bias in the studies was generally high. We did not pool data from studies
        because of significant differences in the interventions and outcomes evaluated. We found a lack of
        standardisation and consistency in the outcomes measured and the endpoints at which they were evaluated.
        We found no evidence that therapeutic exercises were better than TAU, or any other treatment, in improving the
        safety and efficiency of oral swallowing (our primary outcome) or in improving any of the secondary outcomes.
        Using the GRADE system, we classified the overall quality of the evidence for each outcome as very low, due to
        the limited number of trials and their low quality. There were no adverse events reported that were directly
        attributable to the intervention (swallowing exercises). We found no evidence that undertaking therapeutic
        exercises before, during and/or immediately after HNC treatment leads to improvement in oral swallowing. This
        absence of evidence may be due to the small participant numbers in trials, resulting in insufficient power to
        detect any difference. Data from the identified trials could not be combined due to differences in the choice
        of primary outcomes and in the measurement tools used to assess them, and the differing baseline and endpoints
        across studies. Designing and implementing studies with stronger methodological rigour is essential. There needs
        to be agreement about the key primary outcomes, the choice of validated assessment tools to measure them and the
        time points at which those measurements are made.",
        "pls": "We included six studies with 326 participants who undertook therapeutic exercises before, during and/or
        after HNC treatment. We could not combine the results of the studies because of the variation in participants'
        cancers, their treatments, the outcomes measured and the tools used to assess them, as well as the differing
        time points for testing. Researchers have compared: (i) therapeutic exercises versus treatment as usual (TAU);
        (ii) therapeutic exercises versus sham therapy; (iii) therapeutic exercises plus TAU versus TAU. The therapeutic
        exercises varied in their design, timing and intensity. TAU involved managing patients' dysphagia when it
        occurred, including inserting a tube for non-oral feeding. The evidence is up to date to 1 July 2016. We found
        no evidence that therapeutic exercises were better than TAU, or any other treatment, in improving the safety and
        efficiency of oral swallowing (our primary outcome) or in improving any of the secondary outcomes. However,
        there is insufficient evidence to draw any clear conclusion about the effects of undertaking therapeutic
        exercises before during and/or immediately after HNC treatment on preventing or reducing dysphagia. Studies had
        small participant numbers, used complex interventions and varied in the choice of outcomes measured, making it
        difficult to draw reliable conclusions. There were no reported adverse events directly attributable to the
        intervention (swallowing exercises). The current quality of the evidence to support the use of therapeutic
        exercises before, during and/or immediately after HNC treatment to prevent/reduce dysphagia is very low. We need
        better designed, rigorous studies with larger participant numbers and agreed endpoints and outcome measurements
        in order to draw clear(er) conclusions."
    },

    where "pls" stands for "plain-language summary".

    Paper: http://arxiv.org/abs/2104.05767
    Code: https://github.com/AshOlogn/Paragraph-level-Simplification-of-Medical-Texts

    @inproceedings{devaraj-etal-2021-paragraph,
        title = "Paragraph-level Simplification of Medical Texts",
        author = "Devaraj, Ashwin and Marshall, Iain and Wallace, Byron and Li, Junyi Jessy",
        booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for
                     Computational Linguistics",
        month = jun,
        year = "2021",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/2021.naacl-main.395",
        pages = "4972--4984",
    }
    """

    DOWNLOAD_URL_TEMPLATE: str = (
        "https://raw.githubusercontent.com/AshOlogn/Paragraph-level-Simplification-of-Medical-Texts/"
        "main/data/data-1024/{file_name}"
    )

    name = "med_paragraph_simplification"
    description = "Corpus with technical abstracts and their lay summaries on various clinical topics"
    tags = ["summarization", "biomedical"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path: str = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)

        instances: List[Instance] = []
        for split in ALL_SPLITS:
            # Original abstracts
            abstract_file_name: str = f"{'val' if split == VALID_SPLIT else split}.source"
            abstract_path: str = os.path.join(data_path, abstract_file_name)
            ensure_file_downloaded(
                source_url=MedParagraphSimplificationScenario.DOWNLOAD_URL_TEMPLATE.format(
                    file_name=abstract_file_name
                ),
                target_path=abstract_path,
            )

            # Plain-language summaries of the abstracts
            pls_file_name: str = f"{'val' if split == VALID_SPLIT else split}.target"
            pls_path: str = os.path.join(data_path, pls_file_name)
            ensure_file_downloaded(
                source_url=MedParagraphSimplificationScenario.DOWNLOAD_URL_TEMPLATE.format(file_name=pls_file_name),
                target_path=pls_path,
            )

            with open(abstract_path, "r") as abstract_file:
                with open(pls_path, "r") as pls_file:
                    for abstract_line, summary_line in zip(abstract_file, pls_file):
                        abstract: str = abstract_line.rstrip()
                        summary: str = summary_line.rstrip()
                        instance: Instance = Instance(
                            input=Input(text=abstract),
                            references=[Reference(Output(text=summary), tags=[CORRECT_TAG])],
                            split=split,
                        )
                        instances.append(instance)

        return instances
