from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
)
from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec,
    get_multiple_choice_separate_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_metric_specs,
    get_classification_metric_specs,
    get_exact_match_metric_specs,
    get_f1_metric_specs,
)
from helm.benchmark.metrics.bhasa_metrics import (
    get_bhasa_summarization_metric_specs,
    get_bhasa_machine_translation_metric_specs,
    get_bhasa_qa_metric_specs,
)
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec

# NLU
@run_spec_function("indicqa")
def get_indicqa_spec() -> RunSpec:
    name = "indicqa"

    adapter_spec = get_generation_adapter_spec(
        instructions="உங்களுக்கு ஒரு பத்தியும் ஒரு கேள்வியும் தரப்படும். தரப்பட்ட பத்தியிலிருந்து கேள்விக்கான பதிலைக் கண்டறியவும்.",
        output_noun="பதில்",
        stop_sequences=["<|endoftext|>", "\n"],
        max_tokens=128,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.IndicQAScenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_bhasa_qa_metric_specs(args={
            "language": 'ta',
        }),
        groups=["bhasa_nlu"],
    )

@run_spec_function("tydiqa")
def get_tydiqa_spec() -> RunSpec:
    name = "tydiqa"

    adapter_spec = get_generation_adapter_spec(
        instructions="Anda akan diberikan sebuah paragraf dan sebuah pertanyaan. Jawablah pertanyaannya dengan mengekstrak jawaban dari paragraf tersebut.",
        output_noun="Jawaban",
        stop_sequences=["<|endoftext|>", "\n"],
        max_tokens=128,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.TyDiQAScenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_bhasa_qa_metric_specs(args={
            "language": 'id',
        }),
        groups=["bhasa_nlu"],
    )

xquad_prompts = {
    "th": {
        "instructions": "คุณจะได้รับข้อความและคำถาม กรุณาตอบคำถามโดยแยกคำตอบจากข้อความ",
        "output_noun": "คำตอบ",
    },
    "vi": {
        "instructions": "Bạn sẽ được cho một đoạn văn và một câu hỏi. Trả lời câu hỏi bằng cách trích xuất câu trả lời từ đoạn văn.",
        "output_noun": "Câu trả lời",
    },
}

def generate_xquad_run_spec(language="th"):
    name = f"xquad_{language}"
        
    adapter_spec = get_generation_adapter_spec(
        instructions=xquad_prompts[language]['instructions'],
        output_noun=xquad_prompts[language]['output_noun'],
        stop_sequences=["<|endoftext|>", "\n"],
        max_tokens=128,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.XQuADScenario",
        args={
            "language": language,
        },
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_bhasa_qa_metric_specs(args={
            "language": language,
        }),
        groups=["bhasa_nlu"],
    )
    
@run_spec_function("xquad")
def get_xquad_spec(language='th') -> RunSpec:
    return generate_xquad_run_spec(language)

@run_spec_function("indicsentiment")
def get_indicsentiment_spec() -> RunSpec:
    name = "indicsentiment"

    adapter_spec = get_generation_adapter_spec(
        instructions="பின்வரும் வாக்கியத்தில் வெளிப்படுத்தப்படும் உணர்வு எது?\nஒரு சொல்லில் மட்டும் பதிலளிக்கவும்:\n- நேர்மறை\n- எதிர்மறை",
        input_noun="வாக்கியம்",
        output_noun="பதில்",
        stop_sequences=["<|endoftext|>", "\n"],
        max_tokens=8,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.IndicSentimentScenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_classification_metric_specs() + get_basic_metric_specs([]),
        groups=["bhasa_nlu"],
    )

@run_spec_function("nusax")
def get_nusax_spec() -> RunSpec:
    name = "nusax"

    adapter_spec = get_generation_adapter_spec(
        instructions="Apa sentimen dari kalimat berikut ini?\nJawablah dengan satu kata saja:\n- Positif\n- Negatif\n- Netral",
        input_noun="Kalimat",
        output_noun="Jawaban",
        stop_sequences=["<|endoftext|>", "\n"],
        max_tokens=8,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.NusaXScenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("wisesight")
def get_wisesight_spec() -> RunSpec:
    name = "wisesight"

    adapter_spec = get_generation_adapter_spec(
        instructions="อารมณ์ความรู้สึกของข้อความต่อไปนี้เป็นอย่างไร?\nกรุณาตอบโดยใช้คำเดียวเท่านั้น:\n- แง่บวก\n- แง่ลบ\n- เฉยๆ",
        input_noun="ข้อความ",
        output_noun="คำตอบ",
        stop_sequences=["<|endoftext|>", "\n"],
        max_tokens=8,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.WisesightScenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("uitvsfc")
def get_uitvsfc_spec() -> RunSpec:
    name = "uitvsfc"

    adapter_spec = get_generation_adapter_spec(
        instructions="Sắc thái của câu sau đây là gì?\nTrả lời với một từ duy nhất:\n- Tích cực\n- Tiêu cực\n- Trung lập",
        input_noun="Câu văn",
        output_noun="Câu trả lời",
        stop_sequences=["<|endoftext|>", "\n"],
        max_tokens=8,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.UITVSFCScenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("mlhsd")
def get_mlhsd_spec() -> RunSpec:
    name = "mlhsd"

    adapter_spec = get_generation_adapter_spec(
        instructions="Anda adalah pendeteksi ujaran kebencian. Definisi dari labelnya adalah sebagai berikut:\nBersih: Tidak ada ujaran kebencian.\nKasar: Ada ujaran kebencian dan kata-kata kasar, namun tidak menyerang pihak tertentu.\nBenci: Ada ujaran kebencian atau serangan langsung terhadap pihak tertentu.\nBerdasarkan definisi labelnya, klasifikasikan kalimat berikut ini dengan satu kata saja:\n- Bersih\n- Kasar\n- Benci",
        input_noun="Kalimat",
        output_noun="Jawaban",
        stop_sequences=["<|endoftext|>", "\n"],
        max_tokens=8,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.MLHSDScenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("thaitoxicitytweets")
def get_thaitoxicitytweets_spec() -> RunSpec:
    name = "thaitoxicitytweets"
    
    adapter_spec = get_generation_adapter_spec(
        instructions="คุณเป็นเครื่องตรวจจับข้อความที่เป็นพิษ\nข้อความเป็นพิษถ้ามันทำร้าย สร้างความเสียหาย หรือสื่อเจตนาเชิงลบถึงบุคคลหรือกลุ่มบุคคล และคุณควรคำนึงถึงการประชดประชันด้วย\nเมื่อได้รับข้อความ ให้ตอบด้วยตัวอักษร Y หากเป็นพิษ และ N หากไม่เป็นพิษ",
        input_noun="ข้อความ",
        output_noun="คำตอบ",
        stop_sequences=["<|endoftext|>", "\n"],
        max_tokens=8,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.ThaiToxicityTweetsScenario"
    ) 

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("vihsd")
def get_vihsd_spec() -> RunSpec:
    name = "vihsd"

    adapter_spec = get_generation_adapter_spec(
        instructions="Bạn là máy phát hiện phát ngôn thù ghét. Các nhãn được định nghĩa như sau:\nSạch: Không quấy rối.\nCông kích: Bao gồm quấy rối và thậm chí chửi thề, nhưng không tấn công bất kì đối tượng cụ thể nào.\nThù ghét: Trực tiếp quấy rối hay lăng mạ một đối tượng cụ thể.\nVới các định nghĩa của nhãn, hãy phân loại câu dưới đây với một từ duy nhất:\n- Sạch\n- Công kích\n- Thù ghét",
        input_noun="Câu văn",
        output_noun="Câu trả lời",
        stop_sequences=["<|endoftext|>", "\n"],
        max_tokens=8,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.ViHSDScenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

# NLG
flores_prompts = {
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

def generate_flores_run_spec(source="en", target="id"):
    pair = f"{source}_{target}"
    name = f"flores_{pair}"
        
    adapter_spec = get_generation_adapter_spec(
        instructions=flores_prompts[pair]['instructions'],
        input_noun=flores_prompts[pair]['input_noun'],
        output_noun=flores_prompts[pair]['output_noun'],
        stop_sequences=["<|endoftext|>", "\n"],
        max_tokens=128,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.FloresScenario",
        args={
            "pair": pair,
        },
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_bhasa_machine_translation_metric_specs(),
        groups=["bhasa_nlg"],
    )
    
@run_spec_function("flores")
def get_flores_spec(source="en", target="id") -> RunSpec:
    return generate_flores_run_spec(source, target)

xlsum_prompts = {
    "id": {
        "instructions": "Rangkumkan artikel Bahasa Indonesia berikut ini dalam 1 atau 2 kalimat. Jawabannya harus ditulis dalam Bahasa Indonesia.",
        "input_noun": "Artikel",
        "output_noun": "Rangkuman",
    },
    "ta": {
        "instructions": "பின்வரும் தமிழ்க் கட்டுரைக்கு 1 அல்லது 2 வாக்கியங்களில் பொழிப்பு எழுதவும். பதில் தமிழ் மொழியில் இருக்கவேண்டும்.",
        "input_noun": "கட்டுரை",
        "output_noun": "கட்டுரைப் பொழிப்பு",
    },
    "th": {
        "instructions": "กรุณาสรุปบทความภาษาไทยต่อไปนี้ใน 1 หรือ 2 ประโยค คำตอบควรเป็นภาษาไทย",
        "input_noun": "บทความ",
        "output_noun": "บทสรุป",
    },
    "vi": {
        "instructions": "Tóm tắt bài báo Tiếng Việt dưới đây với 1 hay 2 câu. Câu trả lời nên được viết bằng tiếng Việt.",
        "input_noun": "Bài báo",
        "output_noun": "Bản tóm tắt",
    }
}

def generate_xlsum_run_spec(language="id"):
    name = f"xlsum_{language}"
        
    adapter_spec = get_generation_adapter_spec(
        instructions=xlsum_prompts[language]['instructions'],
        input_noun=xlsum_prompts[language]['input_noun'],
        output_noun=xlsum_prompts[language]['output_noun'],
        stop_sequences=["<|endoftext|>", "\n"],
        max_tokens=128,
        temperature=0.3,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.XLSumScenario",
        args={
            "language": language,
        },
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_bhasa_summarization_metric_specs(args={
            "language": language
        }),
        groups=["bhasa_nlg"],
    )
    
@run_spec_function("xlsum")
def get_xlsum_spec(language="id") -> RunSpec:
    return generate_xlsum_run_spec(language)

# NLR

@run_spec_function("indicxnli")
def get_indicxnli_spe() -> RunSpec:
    name = "indicxnli"

    adapter_spec = get_generation_adapter_spec(
        instructions="உங்களுக்கு இரண்டு வாக்கியங்கள், X மற்றும் Y, தரப்படும்.\nபின்வரும் கூற்றுகளில் எது X மற்றும் Y வாக்கியங்களுடன் மிகப் பொருந்துகிறது எனக் கண்டறியவும்.\nA: X உண்மை என்றால் Y உம் உண்மையாக இருக்க வேண்டும்.\nB: X உம் Y உம் முரண்படுகின்றன.\nC: X உண்மையாக இருக்கும்போது Y உண்மையாக இருக்கலாம் அல்லது இல்லாமல் இருக்கலாம்.\nA அல்லது B அல்லது C என்ற ஒறே எழுத்தில் மட்டும் பதிலளிக்கவும்.",
        output_noun="பதில்",
        stop_sequences=["<|endoftext|>", "\n"],
        max_tokens=8,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.IndicXNLIScenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlr"],
    )

@run_spec_function("indonli")
def get_indonli_spec() -> RunSpec:
    name = "indonli"

    adapter_spec = get_generation_adapter_spec(
        instructions="Anda akan diberikan dua kalimat, X dan Y.\nTentukan mana dari pernyataan berikut ini yang paling sesuai untuk kalimat X dan Y.\nA: Kalau X benar, maka Y juga harus benar.\nB: X bertentangan dengan Y.\nC: Ketika X benar, Y mungkin benar atau mungkin tidak benar.\nJawablah dengan satu huruf saja, A, B atau C.",
        output_noun="Jawaban",
        stop_sequences=["<|endoftext|>", "\n"],
        max_tokens=8,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.IndoNLIScenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlr"],
    )

xnli_prompts = {
    "th": {
        "instructions": "คุณจะได้รับสองข้อความ X และ Y",
        "input_suffix": "กรุณาพิจารณาว่า ข้อความใดต่อไปนี้ใช้กับข้อความ X และ Y ได้ดีที่สุด\nA: ถ้า X เป็นจริง Y จะต้องเป็นจริง\nB: X ขัดแย้งกับ Y\nC: เมื่อ X เป็นจริง Y อาจเป็นจริงหรือไม่ก็ได้\nกรุณาตอบด้วยตัวอักษร A, B หรือ C ตัวเดียวเท่านั้น",
        "output_noun": "คำตอบ",
    },
    "vi": {
        "instructions": "Bạn sẽ được cho hai câu, X và Y.",
        "input_suffix": "Xác định câu nào sau đây là câu phù hợp nhất cho câu X và Y.\nA: Nếu X đúng thì Y phải đúng.\nB: X mâu thuẫn với Y.\nC: Khi X đúng, Y có thể đúng hoặc không đúng.\nTrả lời với một chữ cái duy nhất A, B, hoặc C.",
        "output_noun": "Câu trả lời",
    },
}

def generate_xnli_run_spec(language="vi"):
    name = f"xnli_{language}"

    adapter_spec = get_generation_adapter_spec(
        instructions=xnli_prompts[language]['instructions'] + '\n' + xnli_prompts[language]['input_suffix'],
        output_noun=xnli_prompts[language]['output_noun'],
        stop_sequences=["<|endoftext|>", "\n"],
        max_tokens=8,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.XNLIScenario",
        args={
            "language": language,
        },
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlr"],
    )

@run_spec_function("xnli")
def get_xnli_spec(language="vi") -> RunSpec:
    return generate_xnli_run_spec(language)

xcopa_prompts = {
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
    }
}

def generate_xcopa_run_spec(language="id"):
    name = f"xcopa_{language}"
        
    adapter_spec = get_generation_adapter_spec(
        input_noun=xcopa_prompts[language]['input_noun'],
        output_noun=xcopa_prompts[language]['output_noun'],
        stop_sequences=["<|endoftext|>", "\n"],
        max_tokens=8,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.XCOPAScenario",
        args={
            "language": language,
        },
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlr"],
    )

@run_spec_function("xcopa")
def get_xcopa_spec(language="id") -> RunSpec:
    return generate_xcopa_run_spec(language)

# LD

lindseamp_prompts = {
    "id": {
        "instructions": "Anda adalah seorang ahli bahasa Indonesia.",
        "output_suffix": "Jawablah dengan menggunakan A atau B saja.",
        "output_noun": "Jawaban",
    },
}

def generate_lindseamp_run_spec(language="id") -> RunSpec:
    name = f"lindseamp_{language}"

    adapter_spec = get_multiple_choice_separate_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
        empty_input=True,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.LINDSEAMPScenario",
        args={
            "language": language,
        }
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["bhasa_ld"],
    )

@run_spec_function("lindseamp")
def get_lindseamp_spec(language="id") -> RunSpec:
    return generate_lindseamp_run_spec(language)