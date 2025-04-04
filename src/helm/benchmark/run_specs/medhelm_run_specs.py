"""Run spec functions for the MedHELM leaderboard.

Website: https://crfm.stanford.edu/helm/medhelm/
"""

from typing import Union

from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_MULTIPLE_CHOICE_JOINT,
)
from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec,
    get_multiple_choice_adapter_spec,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_metric_specs,
    get_exact_match_metric_specs,
    get_open_ended_generation_metric_specs,
    get_summarization_metric_specs,
    get_generic_metric_specs,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.common.gpu_utils import get_torch_device_name


@run_spec_function("medcalc_bench")
def get_medcalc_bench_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.medcalc_bench_scenario.MedCalcBenchScenario")

    adapter_spec = get_generation_adapter_spec(
        instructions="Given a patient note and a clinical question, compute the requested medical value.",
        input_noun=None,
        newline_after_input_noun=False,
        output_noun="Answer only the requested quantity without units. No explanation needed",
        max_tokens=10,
        max_train_instances=0,
    )

    metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.medcalc_bench_metrics.MedCalcBenchMetric",
            args={},
        )
    ] + get_exact_match_metric_specs()

    return RunSpec(
        name="medcalc_bench",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["medcalc_bench"],
    )


@run_spec_function("clear")
def get_clear_spec(condition: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.clear_scenario.CLEARScenario", args={"condition": condition}
    )

    condition_display = condition.replace("_", " ")

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions=(
            f"Answer 'A' for 'Has a history of {condition_display}', "
            f"'B' for 'Does not have a history of {condition_display}', or "
            f"'C' for 'Uncertain'"
        ),
        input_noun=None,
        output_noun="Respond only with 'A', 'B', or 'C'. Do not add any other text, punctuation, or symbols",
        max_train_instances=0,
        max_tokens=1,
    )

    return RunSpec(
        name=f"clear:condition={condition}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["clear"],
    )


@run_spec_function("mtsamples_replicate")
def get_mtsamples_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.mtsamples_replicate_scenario.MTSamplesReplicateScenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Given various information about a patient, return a reasonable treatment plan for the patient.",
        input_noun=None,
        newline_after_input_noun=False,
        output_noun="Answer",
        max_tokens=512,
        max_train_instances=0,
        stop_sequences=[],
    )

    annotator_specs = [
        AnnotatorSpec(class_name="helm.benchmark.annotation.mtsamples_replicate_annotator.MTSamplesReplicateAnnotator")
    ]

    metric_args = {
        "task": "mtsamples_replicate",
        "device": get_torch_device_name(),
        "bertscore_model": "distilbert-base-uncased",
        "rescale_with_baseline": False,
    }

    metric_specs = get_summarization_metric_specs(metric_args) + [
        MetricSpec(class_name="helm.benchmark.metrics.mtsamples_replicate_metrics.MTSamplesReplicateMetric", args={})
    ]

    return RunSpec(
        name="mtsamples_replicate",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["mtsamples_replicate"],
    )


@run_spec_function("medec")
def get_medec_run_spec() -> RunSpec:
    """
    RunSpec for the MEDEC dataset.
    This configuration evaluates the model's ability to summarize doctor-patient
    dialogues into structured clinical notes.
    """
    # Define the scenario
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.medec_scenario.MedecScenario",
        args={},
    )

    # Define the adapter
    adapter_spec = get_generation_adapter_spec(
        instructions=(
            "The following is a medical narrative about a patient. "
            "You are a skilled medical doctor reviewing the clinical text. "
            "The text is either correct or contains one error. "
            "The text has a sentence per line. Each line starts with the "
            "sentence ID, followed by a space character then the sentence to check. "
            "Check every sentence of the text. "
            "If the text is correct return the following output: CORRECT. "
            "If the text has a medical error, return the sentence ID of the "
            "sentence containing the error, followed by a space, "
            "and a corrected version of the sentence."
        ),
        input_noun="Clinical Note",
        output_noun="Answer",
        max_tokens=256,
        max_train_instances=0,
    )

    # Define the metrics
    metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.medec_metrics.MedecMetric",
            args={},
        )
    ] + get_basic_metric_specs([])

    # Return the RunSpec
    return RunSpec(
        name="medec",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["clinical", "medec"],
    )


@run_spec_function("ehrshot")
def get_ehrshot_spec(subject: str, max_length: int = 100000) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.ehrshot_scenario.EHRSHOTScenario",
        args={"subject": subject, "max_length": max_length},
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Answer A for yes, B for no.",
        input_noun="",
        output_noun="Respond with only 'A' for yes or 'B' for no. Do not add any other text, punctuation, or symbols",
        max_train_instances=0,
        max_tokens=1,
    )

    return RunSpec(
        name=f"ehrshot:subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["ehrshot"],
    )


@run_spec_function("head_qa")
def get_head_qa_run_spec(language: str = "en", category: Union[str, None] = None) -> RunSpec:
    """
    RunSpec for the HEAD-QA dataset.
    This configuration evaluates the model's ability to answer challenging multiple-choice biomedical questions.
    """
    # Define the scenario
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.headqa_scenario.HeadQAScenario",
        args={"language": language, "category": category},
    )

    # Define the adapter
    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions=(
            "You are a highly knowledgeable AI assistant specializing in biomedical sciences. Your task is to answer "
            "multiple-choice questions accurately based on the options provided. "
            "Each question will relate to biomedical concepts, "
            "and you will be asked to choose the most appropriate answer.\n\n"
            "Select the correct answer by outputting only the letter corresponding to your choice (A, B, C, or D)."
        ),
        input_noun="Question",
        output_noun="Answer",
        max_tokens=1,
        max_train_instances=0,
    )

    # Define the metrics
    metric_specs = get_exact_match_metric_specs()

    # Return the RunSpec
    return RunSpec(
        name=f"head_qa:language={language},category={category}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["biomedical", "head_qa"],
    )


@run_spec_function("medbullets")
def get_medbullets_run_spec() -> RunSpec:
    """
    RunSpec for the MedBullets dataset.
    This configuration evaluates the model's ability to answer challenging multiple-choice clinical questions.
    """
    # Define the scenario
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.medbullets_scenario.MedBulletsScenario",
        args={},
    )

    # Define the adapter
    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions=(
            "You are a highly knowledgeable AI assistant specializing in medicine. "
            "Your task is to answer medical questions similar to those found on the USMLE Step 2/3 exams. "
            "You will be provided with a clinical scenario followed by several multiple-choice options.\n\n"
            "Select the correct answer by outputting only the letter corresponding to your choice (A, B, C, D, or E)."
        ),
        input_noun="Clinical Scenario",
        output_noun="Answer",
        max_tokens=1,
        max_train_instances=0,
    )

    # Define the metrics
    metric_specs = get_exact_match_metric_specs()

    # Return the RunSpec
    return RunSpec(
        name="medbullets",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["clinical", "medbullets"],
    )


@run_spec_function("medbullets_freetext")
def get_medbullets_freetext_run_spec() -> RunSpec:
    """RunSpec for the MedBullets Free-text dataset."""
    # Define the scenario
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.medbullets_scenario.MedBulletsFreeTextScenario",
        args={},
    )

    # Define the adapter
    adapter_spec = get_generation_adapter_spec(
        instructions=(
            "You are a helpful and highly knowledgeable AI assistant specializing in medicine. "
            "Your task is to answer medical questions similar to those found on the USMLE Step 2/3 exams. "
            "You will be provided with a clinical scenario, "
            "and for each question, you must:\n"
            "- Provide an answer to the question.\n"
            "- Give a concise explanation for why that answer is correct, based on the clinical scenario provided."
        ),
        input_noun="Clinical Scenario",
        output_noun="Answer",
    )

    # Define the metrics
    metric_specs = get_open_ended_generation_metric_specs()

    # Return the RunSpec
    return RunSpec(
        name="medbullets-freetext",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["clinical", "medbullets-freetext"],
    )


@run_spec_function("medalign")
def get_medalign_spec(max_length: int = 40000) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.medalign_scenario.MedalignScenario", args={"max_length": max_length}
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="",
        input_noun=None,
        newline_after_input_noun=False,
        output_noun=None,
        max_tokens=256,
        stop_sequences=[],
        max_train_instances=0,
    )

    annotator_specs = [AnnotatorSpec(class_name="helm.benchmark.annotation.medalign_annotator.MedalignAnnotator")]

    metric_args = {
        "task": "medalign",
        "device": get_torch_device_name(),
        "bertscore_model": "distilbert-base-uncased",
        "rescale_with_baseline": False,
    }
    metric_specs = get_summarization_metric_specs(metric_args) + [
        MetricSpec(class_name="helm.benchmark.metrics.medalign_metrics.MedalignMetric", args={})
    ]

    return RunSpec(
        name="medalign",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["medalign"],
    )


@run_spec_function("shc_ptbm_med")
def get_shc_ptbm_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.shc_ptbm_scenario.SHCPTBMMedScenario", args={})

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Answer A or B.",
        input_noun="",
        output_noun="",
    )

    return RunSpec(
        name="shc_ptbm_med",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["shc_ptbm_med"],
    )


@run_spec_function("shc_sei_med")
def get_shc_sei_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.shc_sei_scenario.SHCSEIMedScenario", args={})

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Answer A or B.",
        input_noun="",
        output_noun="",
    )

    return RunSpec(
        name="shc_sei_med",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["shc_sei_med"],
    )


@run_spec_function("dischargeme")
def get_dischargeme_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.dischargeme_scenario.DischargeMeScenario")

    adapter_spec = get_generation_adapter_spec(
        instructions=(
            "Given a discharge text, a radiology report text, and a target "
            "document of either discharge instructions or a brief hospital course, "
            "return the generated target document from the context provided."
        ),
        input_noun=None,
        newline_after_input_noun=False,
        output_noun="Answer",
        max_tokens=300,
        stop_sequences=[],
        max_train_instances=0,
    )

    annotator_specs = [AnnotatorSpec(class_name="helm.benchmark.annotation.dischargeme_annotator.DischargeMeAnnotator")]

    metric_args = {
        "task": "dischargeme",
        "device": get_torch_device_name(),
        "bertscore_model": "distilbert-base-uncased",
        "rescale_with_baseline": False,
    }
    metric_specs = get_summarization_metric_specs(metric_args) + [
        MetricSpec(class_name="helm.benchmark.metrics.dischargeme_metrics.DischargeMeMetric", args={})
    ]
    return RunSpec(
        name="dischargeme",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["dischargeme"],
    )


@run_spec_function("aci_bench")
def get_aci_bench_run_spec() -> RunSpec:
    """
    RunSpec for the ACI-Bench dataset.
    This configuration evaluates the model's ability to summarize
    doctor-patient dialogues into structured clinical notes.
    """
    # Define the scenario
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.aci_bench_scenario.ACIBenchScenario",
        args={},
    )

    # Define the adapter
    adapter_spec = get_generation_adapter_spec(
        instructions=(
            "Summarize the conversation to generate a clinical note with four sections:\n"
            "1. HISTORY OF PRESENT ILLNESS\n"
            "2. PHYSICAL EXAM\n"
            "3. RESULTS\n"
            "4. ASSESSMENT AND PLAN\n\n"
            "The conversation is:"
        ),
        input_noun="Conversation",
        output_noun="Clinical Note",
        max_tokens=768,  # avg tokens in response is 618.9
        max_train_instances=0,
        stop_sequences=[],
    )

    annotator_specs = [AnnotatorSpec(class_name="helm.benchmark.annotation.aci_bench_annotator.ACIBenchAnnotator")]

    # Define the metrics
    metric_args = {
        "task": "aci_bench",
        "device": get_torch_device_name(),
        "bertscore_model": "distilbert-base-uncased",
        "rescale_with_baseline": False,
    }
    metric_specs = get_summarization_metric_specs(metric_args) + [
        MetricSpec(class_name="helm.benchmark.metrics.aci_bench_metrics.ACIBenchMetric", args={})
    ]

    # Return the RunSpec
    return RunSpec(
        name="aci_bench",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["clinical", "aci_bench"],
    )


@run_spec_function("mtsamples_procedures")
def get_mtsamples_procedures_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.mtsamples_procedures_scenario.MTSamplesProceduresScenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Here are information about a patient, return a reasonable treatment plan for the patient.",
        input_noun="Patient Notes",
        newline_after_input_noun=False,
        output_noun="Answer",
        max_tokens=512,
        max_train_instances=0,
        stop_sequences=[],
    )

    annotator_specs = [
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.mtsamples_procedures_annotator.MTSamplesProceduresAnnotator"
        )
    ]

    metric_args = {
        "task": "mtsamples_procedures",
        "device": get_torch_device_name(),
        "bertscore_model": "distilbert-base-uncased",
        "rescale_with_baseline": False,
    }

    metric_specs = get_summarization_metric_specs(metric_args) + [
        MetricSpec(class_name="helm.benchmark.metrics.mtsamples_procedures_metrics.MTSamplesProceduresMetric", args={})
    ]

    return RunSpec(
        name="mtsamples_procedures",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["mtsamples_procedures"],
    )


@run_spec_function("mimic_rrs")
def get_mimic_rrs_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.mimic_rrs_scenario.MIMICRRSScenario", args={})

    adapter_spec = get_generation_adapter_spec(
        instructions=(
            "Generate the impression section of the radiology report based on its findings. "
            "This will not be used to diagnose nor treat any patients. Be as concise as possible."
        ),
        input_noun="Findings",
        output_noun="Impression",
        newline_after_input_noun=True,
        newline_after_output_noun=True,
        max_tokens=128,
        max_train_instances=0,
        stop_sequences=[],
    )
    annotator_specs = [AnnotatorSpec(class_name="helm.benchmark.annotation.mimic_rrs_annotator.MIMICRRSAnnotator")]

    metric_args = {
        "task": "mimic_rrs",
        "device": get_torch_device_name(),
        "bertscore_model": "distilbert-base-uncased",
        "rescale_with_baseline": False,
    }
    metric_specs = get_summarization_metric_specs(metric_args) + [
        MetricSpec(class_name="helm.benchmark.metrics.mimic_rrs_metrics.MIMICRRSMetric", args={})
    ]
    return RunSpec(
        name="mimic_rrs",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["mimic_rrs"],
    )


@run_spec_function("mimic_bhc")
def get_mimic_bhc_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.mimic_bhc_scenario.MIMICBHCScenario", args={})

    adapter_spec = get_generation_adapter_spec(
        instructions=("Summarize the clinical note into a brief hospital course."),
        input_noun="Clinical Note",
        output_noun="Brief Hospital Course",
        newline_after_input_noun=True,
        newline_after_output_noun=True,
        max_tokens=1024,
        max_train_instances=0,
        stop_sequences=[],
    )
    metric_args = {
        "task": "mimic_bhc",
        "device": get_torch_device_name(),
        "bertscore_model": "distilbert-base-uncased",
        "rescale_with_baseline": False,
    }
    return RunSpec(
        name="mimic_bhc",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_summarization_metric_specs(metric_args),
        groups=["mimic_bhc"],
    )


@run_spec_function("chw_care_plan")
def get_chw_care_plan_run_spec() -> RunSpec:
    """
    RunSpec for the chw_care_plan dataset.
    This configuration evaluates the model's ability to summarize
    doctor-patient dialogues into structured clinical notes.
    """
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.chw_care_plan_scenario.CHWCarePlanScenario",
        args={},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=(
            "Follow the instructions provided regarding conversion of a patient note into a specified format."
        ),
        input_noun="",
        output_noun="",
        max_tokens=768,
        max_train_instances=0,
        stop_sequences=[],
    )
    annotator_specs = [
        AnnotatorSpec(class_name="helm.benchmark.annotation.chw_care_plan_annotator.CHWCarePlanAnnotator")
    ]

    metric_args = {
        "task": "chw_care_plan",
        "device": get_torch_device_name(),
        "bertscore_model": "distilbert-base-uncased",
        "rescale_with_baseline": False,
    }
    metric_specs = get_summarization_metric_specs(metric_args) + [
        MetricSpec(class_name="helm.benchmark.metrics.chw_care_plan_metrics.CHWCarePlanMetric", args={})
    ]
    # Return the RunSpec
    return RunSpec(
        name="chw_care_plan",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["clinical", "chw_care_plan"],
    )


@run_spec_function("medication_qa")
def get_medication_qa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.medication_qa_scenario.MedicationQAScenario")

    adapter_spec = get_generation_adapter_spec(
        instructions="Please answer the following consumer health question.",
        input_noun="Question",
        output_noun="Answer",
        max_train_instances=0,
        max_tokens=512,
        stop_sequences=[],
    )
    annotator_specs = [
        AnnotatorSpec(class_name="helm.benchmark.annotation.medication_qa_annotator.MedicationQAAnnotator")
    ]
    metric_args = {
        "task": "medication_qa",
        "device": get_torch_device_name(),
        "bertscore_model": "distilbert-base-uncased",
        "rescale_with_baseline": False,
    }
    metric_specs = get_summarization_metric_specs(metric_args) + [
        MetricSpec(class_name="helm.benchmark.metrics.medication_qa_metrics.MedicationQAMetric", args={})
    ]
    return RunSpec(
        name="medication_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["medication_qa"],
    )


@run_spec_function("starr_patient_instructions")
def get_starr_patient_instructions_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.starr_patient_instructions_scenario.StarrPatientInstructionsScenario",
        args={},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=(
            "You are a medical professional tasked with generating personalized post-procedure "
            "patient instructions. Given the following case details which include the patient's "
            "diagnosis, the planned procedure, the history & physical note, and the operative report, "
            "generate clear and actionable instructions for the patient to follow after their procedure. "
            "Don't worry, this information will not be used for any clinical decision making. "
            "This will not be used to diagnose nor treat any patients."
        ),
        input_noun="Case Details",
        output_noun="Patient Instructions",
        max_tokens=256,
        max_train_instances=0,
        stop_sequences=[],
    )
    annotator_specs = [
        AnnotatorSpec(
            class_name=(
                "helm.benchmark.annotation.starr_patient_instructions_annotator.StarrPatientInstructionsAnnotator"
            )
        )
    ]

    metric_args = {
        "task": "starr_patient_instructions",
        "device": get_torch_device_name(),
        "bertscore_model": "distilbert-base-uncased",
        "rescale_with_baseline": False,
    }
    metric_specs = (
        get_summarization_metric_specs(metric_args)
        + [
            MetricSpec(
                class_name="helm.benchmark.metrics.starr_patient_instructions_metrics.StarrPatientInstructionsMetric",
                args={},
            )
        ]
        + get_basic_metric_specs([])
    )
    return RunSpec(
        name="starr_patient_instructions",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["starr_patient_instructions"],
    )


@run_spec_function("med_dialog")
def get_med_dialog_spec(subset: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.med_dialog_scenario.MedDialogScenario", args={"subset": subset}
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Generate a one sentence summary of this patient-doctor conversation.",
        input_noun="Patient-Doctor",
        output_noun="Summary",
        max_tokens=80,
        max_train_instances=0,
    )
    annotator_specs = [AnnotatorSpec(class_name="helm.benchmark.annotation.med_dialog_annotator.MedDialogAnnotator")]

    metric_args = {
        "task": "med_dialog",
        "device": get_torch_device_name(),
        "bertscore_model": "distilbert-base-uncased",
        "rescale_with_baseline": False,
    }
    metric_specs = get_summarization_metric_specs(metric_args) + [
        MetricSpec(class_name="helm.benchmark.metrics.med_dialog_metrics.MedDialogMetric", args={})
    ]
    return RunSpec(
        name=f"med_dialog,subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["med_dialog"],
    )


@run_spec_function("shc_conf_med")
def get_shc_conf_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.shc_conf_scenario.SHCCONFMedScenario", args={})

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Answer A or B.",
        input_noun="",
        output_noun="",
    )

    return RunSpec(
        name="shc_conf_med",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["shc_conf_med"],
    )


@run_spec_function("medi_qa")
def get_medi_qa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.medi_qa_scenario.MediQAScenario", args={})

    adapter_spec = get_generation_adapter_spec(
        instructions="Answer the following consumer health question.",
        input_noun="Question",
        output_noun="Answer",
        max_tokens=1024,
        max_train_instances=0,
        stop_sequences=[],
    )
    annotator_specs = [AnnotatorSpec(class_name="helm.benchmark.annotation.medi_qa_annotator.MediQAAnnotator")]

    metric_args = {
        "task": "medi_qa",
        "device": get_torch_device_name(),
        "bertscore_model": "distilbert-base-uncased",
        "rescale_with_baseline": False,
    }
    metric_specs = get_summarization_metric_specs(metric_args) + [
        MetricSpec(class_name="helm.benchmark.metrics.medi_qa_metrics.MediQAMetric", args={})
    ]
    return RunSpec(
        name="medi_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["medi_qa"],
    )


@run_spec_function("mental_health")
def get_mental_health_spec() -> RunSpec:
    """
    Returns the run specification for the mental health counseling scenario.
    This scenario evaluates a model's ability to generate appropriate counseling responses
    in mental health conversations.
    """
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.mental_health_scenario.MentalHealthScenario")

    adapter_spec = get_generation_adapter_spec(
        instructions=(
            "Given a mental health conversation history, generate an empathetic and appropriate counselor response."
        ),
        input_noun=None,  # No specific input noun needed as format is defined in scenario
        newline_after_input_noun=False,
        output_noun="Counselor response",
        max_tokens=512,
    )
    annotator_specs = [
        AnnotatorSpec(class_name="helm.benchmark.annotation.mental_health_annotator.MentalHealthAnnotator")
    ]

    metric_args = {
        "task": "mental_health",
        "device": get_torch_device_name(),
        "bertscore_model": "distilbert-base-uncased",
        "rescale_with_baseline": False,
    }
    metric_specs = get_summarization_metric_specs(metric_args) + [
        MetricSpec(class_name="helm.benchmark.metrics.mental_health_metrics.MentalHealthMetric", args={})
    ]

    return RunSpec(
        name="mental_health",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["mental_health"],
    )


@run_spec_function("pubmed_qa")
def get_pubmed_qa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.pubmed_qa_scenario.PubMedQAScenario", args={})

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Answer A for yes, B for no or C for maybe.",
        input_noun="Question",
        output_noun="Answer",
        max_train_instances=0,
    )

    return RunSpec(
        name="pubmed_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["pubmed_qa"],
    )


@run_spec_function("ehr_sql")
def get_ehr_sql_run_spec() -> RunSpec:
    """
    RunSpec for the EHR SQL dataset.
    This configuration evaluates the model's ability to generate accurate SQL queries from natural language questions.
    """

    # Define the scenario
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.ehr_sql_scenario.EhrSqlScenario",
        args={},
    )

    # Define the adapter
    adapter_spec = get_generation_adapter_spec(
        instructions=(
            "You are a highly skilled AI specializing in medical SQL queries. "
            "Given a database schema and a medical question, generate a valid SQL query "
            "that retrieves the required information from the database. "
            "Output only the SQL query without explanations.\n\n"
            "Input: A database schema followed by a natural language question.\n"
            "Output: A valid SQL query ending with ;. Only return SQL query, don't add additional text.\n\n"
            "If the question is unanswerable, return an empty string without additional text or comments."
        ),
        input_noun="Medical Question + Schema",
        output_noun="SQL Query",
        max_tokens=1024,
        temperature=0,
        max_train_instances=0,
        stop_sequences=[],
    )

    annotator_specs = [AnnotatorSpec(class_name="helm.benchmark.annotation.ehr_sql_annotator.EhrSqlAnnotator")]

    # Define the metrics
    metric_specs = [
        MetricSpec(class_name="helm.benchmark.metrics.ehr_sql_metrics.EhrSqlMetric", args={})
    ] + get_exact_match_metric_specs()

    # Return the RunSpec
    return RunSpec(
        name="ehr_sql",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["ehr_sql"],
    )


@run_spec_function("shc_bmt_med")
def get_shc_bmt_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.shc_bmt_scenario.SHCBMTMedScenario", args={})

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Answer A or B.",
        input_noun="",
        output_noun="",
    )

    return RunSpec(
        name="shc_bmt_med",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["shc_bmt_med"],
    )


@run_spec_function("race_based_med")
def get_race_based_med_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.race_based_med_scenario.RaceBasedMedScenario", args={}
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Answer A for yes, B for no.",
        input_noun=None,
        output_noun="Respond with only 'A' for yes or 'B' for no. Do not add any other text, punctuation, or symbols",
        max_train_instances=0,
    )

    return RunSpec(
        name="race_based_med",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["race_based_med"],
    )


@run_spec_function("medhallu")
def get_medhallu_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.medhallu_scenario.MedHalluScenario", args={})

    adapter_spec = get_generation_adapter_spec(
        instructions="""You are an AI assistant with extensive knowledge in the field of medicine. I want you act as an answer judge. Given a question and an answer, your objective is to determine if the provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

You are trying to determine if there is a factual contradiction between the answer and the world knowledge. Some information in the answer might be fabricated. Answer '0' if the answer is factual and '1' if the answer is hallucinated. You can categorize the answer as hallucinated if it falls in either of the category:
Type: Misinterpretation of #Question#: These are hallucinated answers that misunderstands the question, leading to an off-topic or irrelevant response.
Type: Incomplete Information: These are hallucinated answers that Point out what is not true without providing correct information.
Type: Mechanism and Pathway Misattribution - These are hallucinated answer that falsely attribution of biological mechanisms, molecular pathways, or disease processes that contradicts established medical knowledge
Type: Methodological and Evidence Fabrication - Inventing false research methods, statistical data, or specific clinical outcomes

Do not return anything else, just the answer.
Return just an integer value, '0' if the answer is factual and '1' if the answer is hallucinated. No letter or word, just the integer value.""",  # noqa: E501
        input_noun=None,
        output_noun=(
            """Return just an integer value, '0' if the answer is factual and '1' if the answer is hallucinated.
No letter or word, just the integer value.

Your Judgment"""  # noqa: E501
        ),
        max_train_instances=0,
    )

    return RunSpec(
        name="medhallu",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["medhallu"],
    )


@run_spec_function("n2c2_ct_matching")
def get_n2c2_ct_matching_spec(subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.n2c2_ct_matching_scenario.N2C2CTMatchingScenario",
        args={"subject": subject},
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Answer A for yes, B for no.",
        input_noun="",
        output_noun="Answer A for yes, B for no",
        max_train_instances=0,
    )

    return RunSpec(
        name=f"n2c2_ct_matching:subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["n2c2_ct_matching"],
    )


@run_spec_function("shc_gip_med")
def get_shc_gip_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.shc_gip_scenario.SHCGIPMedScenario", args={})

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Answer A or B.",
        input_noun="",
        output_noun="",
    )

    return RunSpec(
        name="shc_gip_med",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["shc_gip_med"],
    )


@run_spec_function("mimiciv_billing_code")
def get_mimiciv_billing_code_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.mimiciv_billing_code_scenario.MIMICIVBillingCodeScenario",
        args={
            "data_file": "/share/pi/nigam/data/medhelm/mimiciv_billing_codes/mimiciv_icd10.feather",
        },
    )
    adapter_spec = get_generation_adapter_spec(
        instructions="Given the following clinical note, identify all relevant ICD-10 codes.",
        input_noun="Note",
        output_noun="Predicted ICD-10 Codes",
        newline_after_input_noun=True,
        newline_after_output_noun=True,
        max_tokens=256,
        max_train_instances=0,
        stop_sequences=[],
    )
    # Define the metrics
    metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.mimiciv_billing_code_metrics.MIMICIVBillingCodeMetric",
            args={},
        )
    ] + get_generic_metric_specs()

    # Return the RunSpec
    return RunSpec(
        name="mimiciv_billing_code",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["mimiciv_billing_code"],
    )


@run_spec_function("shc_sequoia_med")
def get_shc_sequoia_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.shc_sequoia_scenario.SHCSequoiaMedScenario", args={}
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Answer A or B.",
        input_noun="",
        output_noun="",
    )

    return RunSpec(
        name="shc_sequoia_med",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["shc_sequoia_med"],
    )


@run_spec_function("shc_cdi_med")
def get_shc_cdi_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.shc_cdi_scenario.SHCCDIMedScenario", args={})

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Answer A or B.",
        input_noun="",
        output_noun="",
    )

    return RunSpec(
        name="shc_cdi_med",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["shc_cdi_med"],
    )


@run_spec_function("shc_ent_med")
def get_shc_ent_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.shc_ent_scenario.SHCENTMedScenario", args={})

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Answer A, B, or C.",
        input_noun="",
        output_noun="",
    )

    return RunSpec(
        name="shc_ent_med",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["shc_ent_med"],
    )
