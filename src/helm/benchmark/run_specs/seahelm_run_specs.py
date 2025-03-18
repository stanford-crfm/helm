from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
)
from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec,
    get_multiple_choice_separate_adapter_spec,
)
from helm.benchmark.metrics.seahelm_metrics_specs import (
    get_seahelm_machine_translation_metric_specs,
    get_seahelm_qa_metric_specs,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_metric_specs,
    get_exact_match_metric_specs,
    get_classification_metric_specs,
)
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec

# SEA-HELM Run Specs
#   A. Natural Language Understanding
#   B. Natural Language Generation
#   C. Natural Language Reasoning
#   D. Linguistic Diagnostics

# A. Natural Language Understanding
#   1. Question Answering
#   2. Sentiment Analysis
#   3. Toxicity Detection/Classification


# 1. Question Answering
# 1.1 Indonesian: TyDiQA
@run_spec_function("tydiqa")
def get_tydiqa_spec() -> RunSpec:
    name = "tydiqa"

    adapter_spec = get_generation_adapter_spec(
        instructions="Anda akan diberikan sebuah paragraf dan sebuah pertanyaan. Jawablah pertanyaannya dengan "
        "mengekstrak jawaban dari paragraf tersebut.",
        output_noun="Jawaban",
        stop_sequences=["\n"],
        max_tokens=256,
    )

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.seahelm_scenario.TyDiQAScenario")

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_seahelm_qa_metric_specs(
            args={
                "language": "id",
            }
        ),
        groups=["seahelm_nlu", "tydiqa"],
    )


# 1.2 Vietnamese & Thai: XQuAD
XQUAD_PROMPTS = {
    "th": {
        "instructions": "คุณจะได้รับข้อความและคำถาม กรุณาตอบคำถามโดยแยกคำตอบจากข้อความ",
        "output_noun": "คำตอบ",
    },
    "vi": {
        "instructions": "Bạn sẽ được cho một đoạn văn và một câu hỏi. Trả lời câu hỏi bằng cách trích xuất câu "
        "trả lời từ đoạn văn.",
        "output_noun": "Câu trả lời",
    },
}


@run_spec_function("xquad")
def get_xquad_spec(language="th") -> RunSpec:
    name = f"xquad_{language}"

    adapter_spec = get_generation_adapter_spec(
        instructions=XQUAD_PROMPTS[language]["instructions"],
        output_noun=XQUAD_PROMPTS[language]["output_noun"],
        stop_sequences=["\n"],
        max_tokens=256,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.seahelm_scenario.XQuADScenario",
        args={
            "language": language,
        },
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_seahelm_qa_metric_specs(
            args={
                "language": language,
            }
        ),
        groups=["seahelm_nlu", f"xquad_{language}"],
    )


# 1.3 Tamil: IndicQA
@run_spec_function("indicqa")
def get_indicqa_spec() -> RunSpec:
    name = "indicqa"
    i = "உங்களுக்கு ஒரு பத்தியும் ஒரு கேள்வியும் தரப்படும். தரப்பட்ட பத்தியிலிருந்து கேள்விக்கான பதிலைக் கண்டறியவும்."

    adapter_spec = get_generation_adapter_spec(
        instructions=i,
        output_noun="பதில்",
        stop_sequences=["\n"],
        max_tokens=256,
    )

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.seahelm_scenario.IndicQAScenario")

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_seahelm_qa_metric_specs(
            args={
                "language": "ta",
            }
        ),
        groups=["seahelm_nlu", "indicqa"],
    )


# 2. Sentiment Analysis
# 2.1 Indonesian: NusaX Sentiment
@run_spec_function("nusax")
def get_nusax_spec() -> RunSpec:
    name = "nusax"

    adapter_spec = get_generation_adapter_spec(
        instructions="Apa sentimen dari kalimat berikut ini?\nJawablah dengan satu kata saja:"
        "\n- Positif\n- Negatif\n- Netral",
        input_noun="Kalimat",
        output_noun="Jawaban",
        stop_sequences=["\n"],
        max_tokens=16,
    )

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.seahelm_scenario.NusaXScenario")

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_classification_metric_specs(),
        groups=["seahelm_nlu", "nusax"],
    )


# 2.2 Vietnamese: UIT-VSFC
@run_spec_function("uitvsfc")
def get_uitvsfc_spec() -> RunSpec:
    name = "uitvsfc"

    adapter_spec = get_generation_adapter_spec(
        instructions="Sắc thái của câu sau đây là gì?\nTrả lời với một từ duy nhất:"
        "\n- Tích cực\n- Tiêu cực\n- Trung lập",
        input_noun="Câu văn",
        output_noun="Câu trả lời",
        stop_sequences=["\n"],
        max_tokens=16,
    )

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.seahelm_scenario.UITVSFCScenario")

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_classification_metric_specs(),
        groups=["seahelm_nlu", "uitvsfc"],
    )


# 2.3 Thai: Wisesight Sentiment
@run_spec_function("wisesight")
def get_wisesight_spec() -> RunSpec:
    name = "wisesight"
    i = "อารมณ์ความรู้สึกของข้อความต่อไปนี้เป็นอย่างไร?\nกรุณาตอบโดยใช้คำเดียวเท่านั้น:\n- แง่บวก\n- แง่ลบ\n- เฉยๆ"

    adapter_spec = get_generation_adapter_spec(
        instructions=i,
        input_noun="ข้อความ",
        output_noun="คำตอบ",
        stop_sequences=["\n"],
        max_tokens=16,
    )

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.seahelm_scenario.WisesightScenario")

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_classification_metric_specs(),
        groups=["seahelm_nlu", "wisesight"],
    )


# 2.4 Tamil: IndicSentiment
@run_spec_function("indicsentiment")
def get_indicsentiment_spec() -> RunSpec:
    name = "indicsentiment"

    adapter_spec = get_generation_adapter_spec(
        instructions="பின்வரும் வாக்கியத்தில் வெளிப்படுத்தப்படும் உணர்வு எது?\nஒரு சொல்லில் மட்டும் பதிலளிக்கவும்:"
        "\n- நேர்மறை\n- எதிர்மறை",
        input_noun="வாக்கியம்",
        output_noun="பதில்",
        stop_sequences=["\n"],
        max_tokens=16,
    )

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.seahelm_scenario.IndicSentimentScenario")

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_classification_metric_specs() + get_basic_metric_specs([]),
        groups=["seahelm_nlu", "indicsentiment"],
    )


# 3. Toxicity Detection/Classification
# 3.1 Indonesian: Multi-Label Hate Speech Detection
@run_spec_function("mlhsd")
def get_mlhsd_spec() -> RunSpec:
    name = "mlhsd"

    adapter_spec = get_generation_adapter_spec(
        instructions="Anda adalah pendeteksi ujaran kebencian. Definisi dari labelnya adalah sebagai berikut:"
        "\nBersih: Tidak ada ujaran kebencian.\nKasar: Ada ujaran kebencian dan kata-kata kasar, namun "
        "tidak menyerang pihak tertentu.\nBenci: Ada ujaran kebencian atau serangan langsung terhadap pihak "
        "tertentu.\nBerdasarkan definisi labelnya, klasifikasikan kalimat berikut ini dengan satu kata saja:"
        "\n- Bersih\n- Kasar\n- Benci",
        input_noun="Kalimat",
        output_noun="Jawaban",
        stop_sequences=["\n"],
        max_tokens=16,
    )

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.seahelm_scenario.MLHSDScenario")

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_classification_metric_specs(),
        groups=["seahelm_nlu", "mlhsd"],
    )


# 3.2 Vietnamese: ViHSD
@run_spec_function("vihsd")
def get_vihsd_spec() -> RunSpec:
    name = "vihsd"

    adapter_spec = get_generation_adapter_spec(
        instructions="Bạn là máy phát hiện phát ngôn thù ghét. Các nhãn được định nghĩa như sau:"
        "\nSạch: Không quấy rối.\nCông kích: Bao gồm quấy rối và thậm chí chửi thề, nhưng không "
        "tấn công bất kì đối tượng cụ thể nào.\nThù ghét: Trực tiếp quấy rối hay lăng mạ một đối "
        "tượng cụ thể.\nVới các định nghĩa của nhãn, hãy phân loại câu dưới đây với một từ duy nhất:"
        "\n- Sạch\n- Công kích\n- Thù ghét",
        input_noun="Câu văn",
        output_noun="Câu trả lời",
        stop_sequences=["\n"],
        max_tokens=16,
    )

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.seahelm_scenario.ViHSDScenario")

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_classification_metric_specs(),
        groups=["seahelm_nlu", "vihsd"],
    )


# 3.3 Thai: Thai Toxicity Tweets
@run_spec_function("thaitoxicitytweets")
def get_thaitoxicitytweets_spec() -> RunSpec:
    name = "thaitoxicitytweets"

    adapter_spec = get_generation_adapter_spec(
        instructions="คุณเป็นเครื่องตรวจจับข้อความที่เป็นพิษ\nข้อความเป็นพิษถ้ามันทำร้าย สร้างความเสียหาย "
        "หรือสื่อเจตนาเชิงลบถึงบุคคลหรือกลุ่มบุคคล และคุณควรคำนึงถึงการประชดประชันด้วย\nเมื่อได้รับข้อความ "
        "ให้ตอบด้วยตัวอักษร Y หากเป็นพิษ และ N หากไม่เป็นพิษ",
        input_noun="ข้อความ",
        output_noun="คำตอบ",
        stop_sequences=["\n"],
        max_tokens=16,
    )

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.seahelm_scenario.ThaiToxicityTweetsScenario")

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_classification_metric_specs(),
        groups=["seahelm_nlu", "thaitoxicitytweets"],
    )


# B. Natural Language Generation
#   1. Machine Translation

# 1. Machine Translation: FLoRes-200
TRANSLATION_PROMPTS = {
    "en_id": {
        "instructions": "Terjemahkan teks berikut ini ke dalam Bahasa Indonesia.",
        "input_noun": "Teks",
        "output_noun": "Terjemahan",
    },
    "en_ta": {
        "instructions": "பின்வரும் உரையைத் தமிழ் மொழிக்கு மொழிபெயர்க்கவும்.",
        "input_noun": "உரை",
        "output_noun": "மொழிபெயர்ப்பு",
    },
    "en_th": {
        "instructions": "กรุณาแปลข้อความต่อไปนี้เป็นภาษาไทย",
        "input_noun": "ข้อความ",
        "output_noun": "คำแปล",
    },
    "en_vi": {
        "instructions": "Dịch văn bản dưới đây sang Tiếng Việt.",
        "input_noun": "Văn bản",
        "output_noun": "Bản dịch",
    },
    "id_en": {
        "instructions": "Terjemahkan teks berikut ini ke dalam Bahasa Inggris.",
        "input_noun": "Teks",
        "output_noun": "Terjemahan",
    },
    "ta_en": {
        "instructions": "பின்வரும் உரையை ஆங்கில மொழிக்கு மொழிபெயர்க்கவும்.",
        "input_noun": "உரை",
        "output_noun": "மொழிபெயர்ப்பு",
    },
    "th_en": {
        "instructions": "กรุณาแปลข้อความต่อไปนี้เป็นภาษาอังกฤษ",
        "input_noun": "ข้อความ",
        "output_noun": "คำแปล",
    },
    "vi_en": {
        "instructions": "Dịch văn bản dưới đây sang Tiếng Anh.",
        "input_noun": "Văn bản",
        "output_noun": "Bản dịch",
    },
}


@run_spec_function("flores")
def get_flores_spec(source="en", target="id") -> RunSpec:
    pair = f"{source}_{target}"
    name = f"flores_{pair}"

    adapter_spec = get_generation_adapter_spec(
        instructions=TRANSLATION_PROMPTS[pair]["instructions"],
        input_noun=TRANSLATION_PROMPTS[pair]["input_noun"],
        output_noun=TRANSLATION_PROMPTS[pair]["output_noun"],
        stop_sequences=["\n"],
        max_tokens=256,
        sample_train=False,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.seahelm_scenario.FloresScenario",
        args={
            "pair": pair,
        },
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_seahelm_machine_translation_metric_specs(),
        groups=["seahelm_nlg", f"flores_{pair}"],
    )


# C. Natural Language Reasoning
#   1. Natural Language Inference
#   2. Causal Reasoning


# 1. Natural Language Inference
# 1.1 Indonesian: IndoNLI
@run_spec_function("indonli")
def get_indonli_spec() -> RunSpec:
    name = "indonli"

    adapter_spec = get_generation_adapter_spec(
        instructions="Anda akan diberikan dua kalimat, X dan Y.\nTentukan mana dari pernyataan berikut "
        "ini yang paling sesuai untuk kalimat X dan Y.\nA: Kalau X benar, maka Y juga harus benar."
        "\nB: X bertentangan dengan Y.\nC: Ketika X benar, Y mungkin benar atau mungkin tidak benar."
        "\nJawablah dengan satu huruf saja, A, B atau C.",
        output_noun="Jawaban",
        stop_sequences=["\n"],
        max_tokens=2,
    )

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.seahelm_scenario.IndoNLIScenario")

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_classification_metric_specs(),
        groups=["seahelm_nlr", "indonli"],
    )


# 1.2 Vietnamese & Thai: XNLI
XNLI_PROMPTS = {
    "th": {
        "instructions": "คุณจะได้รับสองข้อความ X และ Y",
        "input_suffix": "กรุณาพิจารณาว่า ข้อความใดต่อไปนี้ใช้กับข้อความ X และ Y ได้ดีที่สุด"
        "\nA: ถ้า X เป็นจริง Y จะต้องเป็นจริง\nB: X ขัดแย้งกับ Y\nC: เมื่อ X เป็นจริง Y อาจเป็นจริงหรือไม่ก็ได้"
        "\nกรุณาตอบด้วยตัวอักษร A, B หรือ C ตัวเดียวเท่านั้น",
        "output_noun": "คำตอบ",
    },
    "vi": {
        "instructions": "Bạn sẽ được cho hai câu, X và Y.",
        "input_suffix": "Xác định câu nào sau đây là câu phù hợp nhất cho câu X và Y."
        "\nA: Nếu X đúng thì Y phải đúng.\nB: X mâu thuẫn với Y."
        "\nC: Khi X đúng, Y có thể đúng hoặc không đúng.\nTrả lời với một chữ cái duy nhất A, B, hoặc C.",
        "output_noun": "Câu trả lời",
    },
}


@run_spec_function("xnli")
def get_xnli_spec(language="vi") -> RunSpec:
    name = f"xnli_{language}"

    adapter_spec = get_generation_adapter_spec(
        instructions=XNLI_PROMPTS[language]["instructions"] + "\n" + XNLI_PROMPTS[language]["input_suffix"],
        output_noun=XNLI_PROMPTS[language]["output_noun"],
        stop_sequences=["\n"],
        max_tokens=2,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.seahelm_scenario.XNLIScenario",
        args={
            "language": language,
        },
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_classification_metric_specs(),
        groups=["seahelm_nlr", f"xnli_{language}"],
    )


# 1.3 Tamil: IndicXNLI
@run_spec_function("indicxnli")
def get_indicxnli_spec() -> RunSpec:
    name = "indicxnli"

    adapter_spec = get_generation_adapter_spec(
        instructions="உங்களுக்கு இரண்டு வாக்கியங்கள், X மற்றும் Y, தரப்படும்."
        "\nபின்வரும் கூற்றுகளில் எது X மற்றும் Y வாக்கியங்களுடன் மிகப் பொருந்துகிறது எனக் கண்டறியவும்."
        "\nA: X உண்மை என்றால் Y உம் உண்மையாக இருக்க வேண்டும்.\nB: X உம் Y உம் முரண்படுகின்றன."
        "\nC: X உண்மையாக இருக்கும்போது Y உண்மையாக இருக்கலாம் அல்லது இல்லாமல் இருக்கலாம்."
        "\nA அல்லது B அல்லது C என்ற ஒறே எழுத்தில் மட்டும் பதிலளிக்கவும்.",
        output_noun="பதில்",
        stop_sequences=["\n"],
        max_tokens=2,
    )

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.seahelm_scenario.IndicXNLIScenario")

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_classification_metric_specs(),
        groups=["seahelm_nlr", "indicxnli"],
    )


# 2. Causal Reasoning: XCOPA
XCOPA_PROMPTS = {
    "id": {
        "input_noun": "Situasi",
        "output_noun": "Jawaban",
    },
    "ta": {
        "input_noun": "சூழ்நிலை",
        "output_noun": "பதில்",
    },
    "th": {
        "input_noun": "สถานการณ์",
        "output_noun": "คำตอบ",
    },
    "vi": {
        "input_noun": "Tình huống",
        "output_noun": "Câu trả lời",
    },
}


@run_spec_function("xcopa")
def get_xcopa_spec(language="id") -> RunSpec:
    name = f"xcopa_{language}"

    adapter_spec = get_generation_adapter_spec(
        input_noun=XCOPA_PROMPTS[language]["input_noun"],
        output_noun=XCOPA_PROMPTS[language]["output_noun"],
        stop_sequences=["\n"],
        max_tokens=2,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.seahelm_scenario.XCOPAScenario",
        args={
            "language": language,
        },
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_classification_metric_specs(),
        groups=["seahelm_nlr", f"xcopa_{language}"],
    )


# D. Linguistic Diagnostics (LINDSEA)
#   1. Syntax
#   2. Pragmatics

# 1. Syntax: LINDSEA Minimal Pairs
LINDSEA_OUTPUT_NOUNS = {"id": "Jawaban"}


@run_spec_function("lindsea_syntax_minimal_pairs")
def get_lindsea_syntax_minimal_pairs_spec(language: str = "id", method: str = "mcq") -> RunSpec:
    name = f"lindsea_syntax_minimal_pairs_{language}"
    if method == "mcq":
        adapter_spec = get_generation_adapter_spec(output_noun=LINDSEA_OUTPUT_NOUNS[language], max_tokens=2)
    else:
        adapter_spec = get_multiple_choice_separate_adapter_spec(
            method=ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
            empty_input=True,
        )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.seahelm_scenario.LINDSEASyntaxMinimalPairsScenario",
        args={
            "method": method,
            "language": language,
        },
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=[
            "seahelm_linguistic",
            f"lindsea_syntax_minimal_pairs_{language}",
            f"lindsea_syntax_minimal_pairs_{method}_{language}",
        ],
    )


# 2.1. Pragmatics: LINDSEA Presuppositions
@run_spec_function("lindsea_pragmatics_presuppositions")
def get_lindsea_pragmatics_presuppositions_spec(language: str = "id", subset: str = "all") -> RunSpec:
    name = f"lindsea_pragmatics_presuppositions_{subset}_{language}"

    adapter_spec = get_generation_adapter_spec(
        output_noun=LINDSEA_OUTPUT_NOUNS[language],
        stop_sequences=["\n"],
        max_train_instances=0,
        max_tokens=8,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.seahelm_scenario.LINDSEAPragmaticsPresuppositionsScenario",
        args={
            "language": language,
            "subset": subset,
        },
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=[
            "seahelm_linguistic",
            f"lindsea_pragmatics_presuppositions_{language}",
            f"lindsea_pragmatics_presuppositions_{subset}_{language}",
        ],
    )


# 2.2. Pragmatics: LINDSEA Scalar Implicatures
@run_spec_function("lindsea_pragmatics_scalar_implicatures")
def get_lindsea_pragmatics_scalar_implicatures_spec(language: str = "id", subset: str = "all") -> RunSpec:
    name = f"lindsea_pragmatics_scalar_implicatures_{subset}_{language}"

    adapter_spec = get_generation_adapter_spec(
        output_noun=LINDSEA_OUTPUT_NOUNS[language],
        stop_sequences=["\n"],
        max_train_instances=0,
        max_tokens=8,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.seahelm_scenario.LINDSEAPragmaticsScalarImplicaturesScenario",
        args={
            "language": language,
            "subset": subset,
        },
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=[
            "seahelm_linguistic",
            f"lindsea_pragmatics_scalar_implicatures_{language}",
            f"lindsea_pragmatics_scalar_implicatures_{subset}_{language}",
        ],
    )
