from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec,
    get_machine_translation_adapter_spec,
)
from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION,
    AdapterSpec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_exact_match_metric_specs,
    get_f1_metric_specs,
    get_generative_harms_metric_specs,
    get_summarization_metric_specs,
    get_basic_metric_specs,
    get_open_ended_generation_metric_specs,
)


@run_spec_function("melt_question_answering_mlqa")
def get_melt_question_answering_mlqa_spec(prompt_style: str = "normal") -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.melt_scenarios.MELTQAMLQAScenario")

    if prompt_style == "weak":
        instruction = ""
    elif prompt_style == "medium":
        instruction = (
            "Hãy trả lời câu hỏi bên dưới bằng tiếng Việt với các thông tin được cung cấp trong phần ngữ cảnh. "
            'Nếu trong ngữ cảnh không có đủ thông tin , hãy trả lời "Tôi không biết".'
        )
    elif prompt_style == "normal":
        instruction = (
            "Bạn là một trợ lý hữu dụng sử dụng tiếng Việt, biết tôn trọng và thành thật. "
            "Bạn luôn luôn trả lời các câu hỏi một cách có ích nhiều nhất có thể, nhưng đồng thời phải an toàn. "
            "Câu trả lời của bạn không được bao gồm các ngôn từ độc hại, phân biệt chủng tộc, "
            "phân biệt giới tính, nguy hiểm, nội dung vi phạm pháp luật. Làm ơn hãy chắc chắn câu trả lời "
            "của bạn tự nhiên, tích cực và không thiên vị bất cứ cái gì. Nếu có câu hỏi không hợp lý hoặc không "
            "rõ ràng thì hãy giải thích tại sao thay vì trả lời không đúng sự thật. Nếu bạn không biết câu trả lời "
            "thì đừng chia sẻ thông tin sai sự thật."
        )
    else:
        raise ValueError(f"Invalid prompt style: {prompt_style}")

    adapter_spec = get_generation_adapter_spec(instructions=instruction, output_noun="Trả lời", max_tokens=128)

    return RunSpec(
        name=f"melt_question_answering_mlqa:prompt_style={prompt_style}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_f1_metric_specs() + get_generative_harms_metric_specs(),
        groups=["melt", "question_answering_mlqa"],
    )


@run_spec_function("melt_question_answering_xquad")
def get_melt_question_answering_xquad_spec(prompt_style: str = "normal") -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.melt_scenarios.MELTQAXQuADScenario")

    if prompt_style == "weak":
        instruction = ""
    elif prompt_style == "medium":
        instruction = (
            "Hãy trả lời câu hỏi bên dưới bằng tiếng Việt với các thông tin được cung cấp trong phần ngữ cảnh. "
            'Nếu trong ngữ cảnh không có đủ thông tin , hãy trả lời "Tôi không biết".'
        )
    elif prompt_style == "normal":
        instruction = (
            "Bạn là một trợ lý hữu dụng sử dụng tiếng Việt, biết tôn trọng và thành thật. "
            "Bạn luôn luôn trả lời các câu hỏi một cách có ích nhiều nhất có thể, nhưng đồng thời phải an toàn. "
            "Câu trả lời của bạn không được bao gồm các ngôn từ độc hại, phân biệt chủng tộc, "
            "phân biệt giới tính, nguy hiểm, nội dung vi phạm pháp luật. Làm ơn hãy chắc chắn câu trả lời "
            "của bạn tự nhiên, tích cực và không thiên vị bất cứ cái gì. Nếu có câu hỏi không hợp lý hoặc không "
            "rõ ràng thì hãy giải thích tại sao thay vì trả lời không đúng sự thật. Nếu bạn không biết câu trả lời "
            "thì đừng chia sẻ thông tin sai sự thật."
        )
    else:
        raise ValueError(f"Invalid prompt style: {prompt_style}")

    adapter_spec = get_generation_adapter_spec(instructions=instruction, output_noun="Trả lời", max_tokens=128)

    return RunSpec(
        name=f"melt_question_answering_xquad:prompt_style={prompt_style},",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_f1_metric_specs() + get_generative_harms_metric_specs(),
        groups=["melt", "question_answering_xquad"],
    )


@run_spec_function("melt_summarization_vietnews")
def get_melt_summarization_vietnews_spec(prompt_style: str = "normal", temperature: float = 1.0) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.melt_scenarios.MELTSummarizationVietnewsScenario",
        args={
            "train_min_length": 64,
            "train_max_length": 256,
            "doc_max_length": 2048,
        },
    )

    if prompt_style == "weak":
        instruction = ""
    elif prompt_style == "medium":
        instruction = "Nhiệm vụ của bạn là tóm tắt đoạn văn bản sau, đưa ra câu trả lời là bản tóm tắt."
    elif prompt_style == "normal":
        instruction = (
            "Bạn là một trợ lý hữu dụng, biết tôn trọng và thành thật. Bạn luôn luôn trả lời các câu hỏi một cách "
            "có ích nhiều nhất có thể, nhưng đồng thời phải an toàn. Câu trả lời của bạn không được bao gồm "
            "các ngôn từ độc hại, phân biệt chủng tộc, phân biệt giới tính, nguy hiểm, nội dung vi phạm pháp luật. "
            "Nhiệm vụ của bạn là tóm tắt đoạn văn bản nằm trong triple backtick. Bài tóm tắt phải đầy đủ các thông tin "
            "quan trọng, ngắn gọn và thu hút người đọc. Ngôn ngữ bạn phải sử dụng để tóm tắt là tiếng Việt."
        )
    else:
        raise ValueError(f"Invalid prompt style: {prompt_style}")

    adapter_spec = get_generation_adapter_spec(
        instructions=instruction,
        input_noun="Đoạn văn",
        output_noun="Tóm tắt đoạn văn trên",
        max_tokens=256,
        temperature=temperature,
    )

    return RunSpec(
        name=f"melt_summarization_vietnews:prompt_style={prompt_style},temperature={temperature}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_summarization_metric_specs({"task": "summarization_vietnews", "language": "vi"})
        + get_generative_harms_metric_specs(),
        groups=["melt", "summarization_vietnews"],
    )


@run_spec_function("melt_summarization_wikilingua")
def get_melt_summarization_wikilingua_spec(prompt_style: str = "normal", temperature: float = 1.0) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.melt_scenarios.MELTSummarizationWikilinguaScenario",
        args={
            "train_min_length": 64,
            "train_max_length": 256,
            "doc_max_length": 2048,
        },
    )

    if prompt_style == "weak":
        instruction = ""
    elif prompt_style == "medium":
        instruction = "Nhiệm vụ của bạn là tóm tắt đoạn văn bản sau, đưa ra câu trả lời là bản tóm tắt."
    elif prompt_style == "normal":
        instruction = (
            "Bạn là một trợ lý hữu dụng, biết tôn trọng và thành thật. Bạn luôn luôn trả lời các câu hỏi một cách "
            "có ích nhiều nhất có thể, nhưng đồng thời phải an toàn. Câu trả lời của bạn không được bao gồm "
            "các ngôn từ độc hại, phân biệt chủng tộc, phân biệt giới tính, nguy hiểm, nội dung vi phạm pháp luật. "
            "Nhiệm vụ của bạn là tóm tắt đoạn văn bản nằm trong triple backtick. Bài tóm tắt phải đầy đủ các thông tin "
            "quan trọng, ngắn gọn và thu hút người đọc. Ngôn ngữ bạn phải sử dụng để tóm tắt là tiếng Việt."
        )
    else:
        raise ValueError(f"Invalid prompt style: {prompt_style}")

    adapter_spec = get_generation_adapter_spec(
        instructions=instruction,
        input_noun="Đoạn văn",
        output_noun="Tóm tắt đoạn văn trên",
        max_tokens=256,
        temperature=temperature,
    )

    return RunSpec(
        name=f"melt_summarization_wikilingua:prompt_style={prompt_style},temperature={temperature}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_summarization_metric_specs({"task": "summarization_wikilingua", "language": "vi"})
        + get_generative_harms_metric_specs(),
        groups=["melt", "summarization_wikilingua"],
    )


@run_spec_function("melt_reasoning_vie_synthetic_reasoning")
def get_melt_reasoning_vie_synthetic_reasoning_spec(mode: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.melt_synthetic_reasoning_scenario.MELTSyntheticReasoningScenario",
        args={"mode": mode},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Hãy giải bài toán sau.",
        input_noun="Bài toán",
        output_noun="Lời giải",
        max_train_instances=5,
        stop_sequences=["\n"],
        max_tokens=50,  # answer upperbounded by 50 tokens
    )

    return RunSpec(
        name=f"melt_reasoning_vie_synthetic_reasoning:mode={mode}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_generative_harms_metric_specs(),
        groups=["melt", "synthetic_reasoning", f"synthetic_reasoning_{mode}"],
    )


@run_spec_function("melt_reasoning_vie_synthetic_reasoning_natural")
def get_melt_reasoning_vie_synthetic_reasoning_natural_spec(difficulty: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.melt_srn_scenario.MELTSRNScenario",
        args={"difficulty": difficulty},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Hãy giải quyết vấn đề sau.",
        input_noun="Quy luật",
        newline_after_input_noun=True,
        output_noun=None,
        max_train_instances=3,  # limited by the context length
        max_tokens=20,
    )
    srn_metric_specs = get_basic_metric_specs(["f1_set_match", "iou_set_match", "exact_set_match"])

    return RunSpec(
        name=f"melt_reasoning_vie_synthetic_reasoning_natural:difficulty={difficulty}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=srn_metric_specs + get_generative_harms_metric_specs(),
        groups=["melt", "synthetic_reasoning", "synthetic_reasoning_natural"],
    )


@run_spec_function("melt_math")
def get_math_spec(
    subject: str,
    level: str,
    use_official_examples: str = "False",
    use_chain_of_thought: str = "False",
) -> RunSpec:
    # Convert to bools and remove the str versions
    use_official_examples_bool: bool = use_official_examples.lower() == "true"
    use_chain_of_thought_bool: bool = use_chain_of_thought.lower() == "true"
    del use_official_examples
    del use_chain_of_thought

    if use_chain_of_thought_bool:
        assert not use_official_examples_bool, "Cannot use official examples when use_chain_of_thought is True."
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.melt_scenarios.MELTMATHScenario",
        args={
            "subject": subject,
            "level": level,
            "use_official_examples": use_official_examples_bool,
            "use_chain_of_thought": use_chain_of_thought_bool,
        },
    )

    if use_chain_of_thought_bool:  # Include the solution in the output as per https://arxiv.org/abs/2201.11903
        output_prefix = "Lời giải: "  # Don't include LaTeX '$' delimiters
        output_suffix = "\n"
        instance_prefix = "###\n"  # Don't include LaTeX '$' delimiters
        max_tokens = 400  # Increase the number of tokens to generate
        stop_sequences = ["###"]  # Break at the next instance; extraneous output will be stripped out
        groups = ["math_chain_of_thought"]
    else:
        output_prefix = "Lời giải: $"
        output_suffix = "$\n"
        instance_prefix = "###\n"
        max_tokens = 20
        stop_sequences = ["$"]  # Break at the nearest LaTeX closing delimiter
        groups = ["math_regular"]

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Cho một bài toán, hãy tìm ra lời giải. Rút gọn câu trả lời của bạn càng nhiều càng tốt.\n",
        max_train_instances=8,
        num_outputs=1,
        temperature=0.0,
        stop_sequences=stop_sequences,
        max_tokens=max_tokens,
        input_prefix="Bài toán: ",
        input_suffix="\n",
        output_prefix=output_prefix,
        output_suffix=output_suffix,
        instance_prefix=instance_prefix,
    )

    return RunSpec(
        name=f"melt_math:subject={subject},level={level},"
        f"use_official_examples={use_official_examples_bool},use_chain_of_thought={use_chain_of_thought_bool}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_metric_specs(
            ["math_equiv_chain_of_thought" if use_chain_of_thought_bool else "math_equiv"]
        )
        + get_generative_harms_metric_specs(),
        groups=["melt"] + groups,
    )


@run_spec_function("melt_translation_opus100")
def get_melt_translation_opus100_spec(language_pair: str, max_train_instances: int = 0) -> RunSpec:
    FULL_LANGUAGE_NAMES = {
        "vi": "Vietnamese",
        "en": "English",
    }
    source_language, target_language = language_pair.split("-")

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.melt_translation_scenario.MELTTranslationOPUS100Scenario",
        args={"source_language": source_language, "target_language": target_language},
    )

    adapter_spec = get_machine_translation_adapter_spec(
        source_language=FULL_LANGUAGE_NAMES[source_language],
        target_language=FULL_LANGUAGE_NAMES[target_language],
        max_train_instances=max_train_instances,
    )

    return RunSpec(
        name=f"melt_translation_opus100:language_pair={language_pair},max_train_instances={max_train_instances}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs(),
        groups=["melt", "melt_translation_opus100"],
    )


@run_spec_function("melt_translation_phomt")
def get_melt_translation_phomt_spec(language_pair: str, max_train_instances: int = 0) -> RunSpec:
    FULL_LANGUAGE_NAMES = {
        "vi": "Vietnamese",
        "en": "English",
    }
    source_language, target_language = language_pair.split("-")

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.melt_translation_scenario.MELTTranslationPhoMTScenario",
        args={"source_language": source_language, "target_language": target_language},
    )

    adapter_spec = get_machine_translation_adapter_spec(
        source_language=FULL_LANGUAGE_NAMES[source_language],
        target_language=FULL_LANGUAGE_NAMES[target_language],
        max_train_instances=max_train_instances,
    )

    return RunSpec(
        name=f"melt_translation_phomt:language_pair={language_pair},max_train_instances={max_train_instances}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs(),
        groups=["melt", "melt_translation_phomt"],
    )


@run_spec_function("melt_lm_mask_filling_mlqa")
def get_melt_lm_mask_filling_mlqaa_spec(max_train_instances: int = 0) -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.melt_lm_scenarios.MELTLMMaskFillingMLQAScenario")

    instruction = (
        "Hãy hoàn thành câu sau bằng cách điền vào các vị trí trống được đánh dấu bằng [MASK]. "
        "Chỉ trả lời bằng câu đã hoàn thành và không thêm gì khác."
    )
    adapter_spec = get_generation_adapter_spec(
        instructions=instruction,
        input_noun="Câu có chỗ trống",
        output_noun="Câu đã hoàn thành",
        max_train_instances=max_train_instances,
        num_outputs=1,
        max_tokens=1024,
        temperature=0.0,
    )

    return RunSpec(
        name=f"melt_lm_mask_filling_mlqa:max_train_instances={max_train_instances}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_f1_metric_specs(),
        groups=["melt", "melt_lm_mask_filling_mlqa"],
    )


@run_spec_function("melt_lm_spelling_correction_vsec")
def get_melt_lm_spelling_correction_vsec_spec(max_train_instances: int = 0) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.melt_lm_scenarios.MELTLMSpellingCorrectionVSECScenario"
    )

    instruction = "Hãy sửa lỗi chính tả trong câu sau. Chỉ trả lời bằng câu đã sửa đúng chính tả và không thêm gì khác."
    adapter_spec = get_generation_adapter_spec(
        instructions=instruction,
        input_noun="Câu có lỗi",
        output_noun="Câu đã sửa",
        max_train_instances=max_train_instances,
        num_outputs=1,
        max_tokens=1024,
        temperature=0.0,
    )

    return RunSpec(
        name=f"melt_lm_spelling_correction_vsec:max_train_instances={max_train_instances}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_f1_metric_specs(),
        groups=["melt", "melt_lm_spelling_correction_vsec"],
    )
