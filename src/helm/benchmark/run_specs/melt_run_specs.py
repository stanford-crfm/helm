from helm.benchmark.adaptation.adapter_spec import ADAPT_GENERATION, AdapterSpec
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec, TRAIN_SPLIT, TEST_SPLIT, VALID_SPLIT
from helm.benchmark.adaptation.common_adapter_specs import (
    get_completion_adapter_spec,
    get_generation_adapter_spec,
    get_language_modeling_adapter_spec,
    get_multiple_choice_adapter_spec,
    get_ranking_binary_adapter_spec,
    get_summarization_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_metric_specs,
    get_bias_metric_specs,
    get_classification_metric_specs,
    get_copyright_metric_specs,
    get_disinformation_metric_specs,
    get_exact_match_metric_specs,
    get_f1_metric_specs,
    get_generative_harms_metric_specs,
    get_language_modeling_metric_specs,
    get_numeracy_metric_specs,
    get_open_ended_generation_metric_specs,
    get_summarization_metric_specs,
    get_basic_generation_metric_specs,
    get_basic_reference_metric_specs,
    get_generic_metric_specs,
)


@run_spec_function("melt_question_answering_mlqa")
def get_question_answering_mlqa_spec(prompt_style: str = "normal") -> RunSpec:
    assert prompt_style in ["weak", "medium", "normal"], f"Invalid prompt style: {prompt_style}"

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.melt_scenario.MELTQAScenario", args={
        "dataset_name": "facebook/mlqa",
        "subset": "mlqa.vi.vi",
        "passage_prefix": "Ngữ cảnh: ",
        "question_prefix": "Câu hỏi: ",
        "splits": {
            VALID_SPLIT: "validation",
            TEST_SPLIT: "test",
        }
    })
    if prompt_style == "weak":
        instruction = ""
    elif prompt_style == "medium":
        instruction = "Hãy trả lời câu hỏi bên dưới bằng tiếng Việt với các thông tin được cung cấp trong phần ngữ cảnh. Nếu trong ngữ cảnh không có đủ thông tin , hãy trả lời \"Tôi không biết\"."
    else:
        instruction = "Bạn là một trợ lý hữu dụng sử dụng tiếng Việt, biết tôn trọng và thành thật. Bạn luôn luôn trả lời các câu hỏi một cách có ích nhiều nhất có thể, nhưng đồng thời phải an toàn. Câu trả lời của bạn không được bao gồm các ngôn từ độc hại, phân biệt chủng tộc, phân biệt giới tính, nguy hiểm, nội dung vi phạm pháp luật. Làm ơn hãy chắc chắn câu trả lời của bạn tự nhiên, tích cực và không thiên vị bất cứ cái gì. Nếu có câu hỏi không hợp lý hoặc không rõ ràng thì hãy giải thích tại sao thay vì trả lời không đúng sự thật. Nếu bạn không biết câu trả lời thì đừng chia sẻ thông tin sai sự thật."
    adapter_spec = get_generation_adapter_spec(instructions=instruction, output_noun="Trả lời", max_tokens=128)

    return RunSpec(
        name=f"question_answering_mlqa:prompt_style={prompt_style}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_f1_metric_specs() + get_generative_harms_metric_specs(),
        groups=["melt", "question_answering_mlqa"],
    )


@run_spec_function("melt_question_answering_xquad")
def get_question_answering_xquad_spec(prompt_style: str = "normal") -> RunSpec:
    assert prompt_style in ["weak", "medium", "normal"], f"Invalid prompt style: {prompt_style}"

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.melt_scenario.MELTQAScenario", args={
        "dataset_name": "juletxara/xquad_xtreme",
        "subset": "vi",
        "passage_prefix": "Ngữ cảnh: ",
        "question_prefix": "Câu hỏi: ",
        "splits": {
            VALID_SPLIT: "translate_train",
            TEST_SPLIT: "test",
        }
    })

    if prompt_style == "weak":
        instruction = ""
    elif prompt_style == "medium":
        instruction = "Hãy trả lời câu hỏi bên dưới bằng tiếng Việt với các thông tin được cung cấp trong phần ngữ cảnh. Nếu trong ngữ cảnh không có đủ thông tin , hãy trả lời \"Tôi không biết\"."
    else:
        instruction = "Bạn là một trợ lý hữu dụng sử dụng tiếng Việt, biết tôn trọng và thành thật. Bạn luôn luôn trả lời các câu hỏi một cách có ích nhiều nhất có thể, nhưng đồng thời phải an toàn. Câu trả lời của bạn không được bao gồm các ngôn từ độc hại, phân biệt chủng tộc, phân biệt giới tính, nguy hiểm, nội dung vi phạm pháp luật. Làm ơn hãy chắc chắn câu trả lời của bạn tự nhiên, tích cực và không thiên vị bất cứ cái gì. Nếu có câu hỏi không hợp lý hoặc không rõ ràng thì hãy giải thích tại sao thay vì trả lời không đúng sự thật. Nếu bạn không biết câu trả lời thì đừng chia sẻ thông tin sai sự thật."
    adapter_spec = get_generation_adapter_spec(instructions=instruction, output_noun="Trả lời", max_tokens=128)

    return RunSpec(
        name=f"question_answering_xquad:prompt_style={prompt_style},",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_f1_metric_specs() + get_generative_harms_metric_specs(),
        groups=["melt", "question_answering_xquad"],
    )


@run_spec_function("melt_summarization_vietnews")
def get_summarization_vietnews_spec(prompt_style: str = "normal", temperature: float = 1.0) -> RunSpec:
    assert prompt_style in ["weak", "medium", "normal"], f"Invalid prompt style: {prompt_style}"
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.melt_scenario.MELTSummarizationScenario",
        args={
            "dataset_name": "Yuhthe/vietnews",
            "train_min_length": 64,
            "train_max_length": 256,
            "doc_max_length": 2048,
            "article_key": "article",
            "summary_key": "abstract",
            "splits": {
                TRAIN_SPLIT: "train",
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            }
        },
    )

    if prompt_style == "weak":
        instruction = ""
    elif prompt_style == "medium":
        instruction = "Nhiệm vụ của bạn là tóm tắt đoạn văn bản sau, đưa ra câu trả lời là bản tóm tắt."
    else:
        instruction = "Bạn là một trợ lý hữu dụng, biết tôn trọng và thành thật. Bạn luôn luôn trả lời các câu hỏi một cách có ích nhiều nhất có thể, nhưng đồng thời phải an toàn. Câu trả lời của bạn không được bao gồm các ngôn từ độc hại, phân biệt chủng tộc, phân biệt giới tính, nguy hiểm, nội dung vi phạm pháp luật. Nhiệm vụ của bạn là tóm tắt đoạn văn bản nằm trong triple backtick. Bài tóm tắt phải đầy đủ các thông tin quan trọng, ngắn gọn và thu hút người đọc. Ngôn ngữ bạn phải sử dụng để tóm tắt là tiếng Việt."
    adapter_spec = get_generation_adapter_spec(
        instructions=instruction,
        input_noun="Đoạn văn",
        output_noun="Tóm tắt đoạn văn trên",
        max_tokens=256,
        temperature=temperature,
    )

    return RunSpec(
        name=f"summarization_vietnews:prompt_style={prompt_style},temperature={temperature}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_summarization_metric_specs({"task": "summarization_vietnews", "language": "vi"})
        + get_generative_harms_metric_specs(),
        groups=["melt", "summarization_vietnews"],
    )


@run_spec_function("melt_summarization_wikilingua")
def get_wsummarization_wikilingua_spec(prompt_style: str = "normal", temperature: float = 1.0) -> RunSpec:
    assert prompt_style in ["weak", "medium", "normal"], f"Invalid prompt style: {prompt_style}"
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.melt_scenario.MELTSummarizationScenario",
        args={
            "dataset_name": "GEM/wiki_lingua",
            "train_min_length": 64,
            "train_max_length": 256,
            "doc_max_length": 2048,
            "article_key": "source",
            "summary_key": "target",
            "splits": {
                TRAIN_SPLIT: "train",
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            }
        },
    )

    if prompt_style == "weak":
        instruction = ""
    elif prompt_style == "medium":
        instruction = "Nhiệm vụ của bạn là tóm tắt đoạn văn bản sau, đưa ra câu trả lời là bản tóm tắt."
    else:
        instruction = "Bạn là một trợ lý hữu dụng, biết tôn trọng và thành thật. Bạn luôn luôn trả lời các câu hỏi một cách có ích nhiều nhất có thể, nhưng đồng thời phải an toàn. Câu trả lời của bạn không được bao gồm các ngôn từ độc hại, phân biệt chủng tộc, phân biệt giới tính, nguy hiểm, nội dung vi phạm pháp luật. Nhiệm vụ của bạn là tóm tắt đoạn văn bản nằm trong triple backtick. Bài tóm tắt phải đầy đủ các thông tin quan trọng, ngắn gọn và thu hút người đọc. Ngôn ngữ bạn phải sử dụng để tóm tắt là tiếng Việt."
    adapter_spec = get_generation_adapter_spec(
        instructions=instruction,
        input_noun="Đoạn văn",
        output_noun="Tóm tắt đoạn văn trên",
        max_tokens=256,
        temperature=temperature,
    )

    return RunSpec(
        name=f"summarization_wikilingua:prompt_style={prompt_style},temperature={temperature}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_summarization_metric_specs({"task": "summarization_wikilingua", "language": "vi"})
        + get_generative_harms_metric_specs(),
        groups=["melt", "summarization_wikilingua"],
    )


@run_spec_function("melt_sentiment_analysis_uitvsfc")
def get_sentiment_analysis_uitvsfc_spec() -> RunSpec:
    raise NotImplementedError()


@run_spec_function("melt_sentiment_analysis_vlsp2016")
def get_sentiment_analysis_vlsp2016_spec() -> RunSpec:
    raise NotImplementedError()


@run_spec_function("melt_text_classification_phoatis")
def get_text_classification_phoatis_spec() -> RunSpec:
    raise NotImplementedError()


@run_spec_function("melt_text_classification_uitvsmec")
def get_text_classification_uitvsmec_spec() -> RunSpec:
    raise NotImplementedError()


@run_spec_function("melt_knowledge_uitvimmrc")
def get_knowledge_uitvimmrc_spec() -> RunSpec:
    raise NotImplementedError()


@run_spec_function("melt_knowledge_zaloe2e")
def get_knowledge_zaloe2e_spec() -> RunSpec:
    raise NotImplementedError()


@run_spec_function("melt_toxicity_detection_uitvictsd")
def get_toxicity_detection_uitvictsd_spec() -> RunSpec:
    raise NotImplementedError()


@run_spec_function("melt_toxicity_detection_uitvihsd")
def get_toxicity_detection_uitvihsd_spec() -> RunSpec:
    raise NotImplementedError()


@run_spec_function("melt_information_retrieval_mmarco")
def get_information_retrieval_mmarco_spec() -> RunSpec:
    raise NotImplementedError()


@run_spec_function("melt_information_retrieval_mrobust04")
def get_information_retrieval_mrobust04_spec() -> RunSpec:
    raise NotImplementedError()


@run_spec_function("melt_language_modeling_mlqa")
def get_language_modeling_mlqa_spec() -> RunSpec:
    raise NotImplementedError()


@run_spec_function("melt_language_modeling_vsec")
def get_language_modeling_vsec_spec() -> RunSpec:
    raise NotImplementedError()


@run_spec_function("melt_reasoning_math")
def get_math_reasoning_spec() -> RunSpec:
    raise NotImplementedError()


@run_spec_function("melt_reasoning_synthetic_reasoning")
def get_reasoning_synthetic_reasoning_spec(mode: str) -> RunSpec:
    raise NotImplementedError()


@run_spec_function("melt_reasoning_synthetic_reasoning_natural")
def get_reasoning_synthetic_reasoning_natural_spec(difficulty: str) -> RunSpec:
    raise NotImplementedError()


@run_spec_function("melt_translation_opus100")
def get_translation_opus100_spec() -> RunSpec:
    raise NotImplementedError()


@run_spec_function("melt_translation_phomt")
def get_translation_phomt_spec() -> RunSpec:
    raise NotImplementedError()
