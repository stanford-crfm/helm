import csv
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class ArabicWritingStyleScenario(Scenario):

    name = "arabic_writing_style"
    description = "Arabic Writing Style"
    tags = ["finance"]
    _PROMPTS = {
        "en": {
            "CompanyFinance": "Read the following text carefully and infer its overall writing style. Then write a new text in Arabic on a different topic: an analytical commentary on current market conditions and their impact on the company’s financial results, keeping the same general style without copying sentences or repeating the content verbatim. Make the length of the new text similar to the original text.",
            "CompanyNews": "Read the following text carefully and infer its overall writing style. Then write a new text in Arabic on a different topic: announcing a strategic partnership with a regional tech startup, keeping the same general style without copying sentences or repeating the content verbatim. Make the length of the new text similar to the original text.",
            "CompanyStatements": "Read the following text carefully and infer its overall writing style. Then write a new text in Arabic on a different topic: an internal message to employees about embracing digital transformation in the company, keeping the same general style without copying sentences or repeating the content verbatim. Make the length of the new text similar to the original text.",
        },
        "ar": {
            "CompanyFinance": "اقرأ النص التالي جيدًا، واستخلص أسلوبه العام في الكتابة. بعد ذلك اكتب نصًا جديدًا باللغة العربية عن موضوع مختلف هو: تعليق تحليلي حول أوضاع السوق الحالية وتأثيرها على نتائج الشركة المالية، بحيث يحمل نفس الأسلوب العام دون نسخ الجمل أو تكرار المحتوى حرفيًا. اجعل طول النص الجديد قريبًا من طول النص الأصلي.",
            "CompanyNews": "اقرأ النص التالي جيدًا، واستخلص أسلوبه العام في الكتابة. بعد ذلك اكتب نصًا جديدًا باللغة العربية عن موضوع مختلف هو: إعلان شراكة استراتيجية مع شركة ناشئة إقليمية في مجال التكنولوجيا، بحيث يحمل نفس الأسلوب العام دون نسخ الجمل أو تكرار المحتوى حرفيًا. اجعل طول النص الجديد قريبًا من طول النص الأصلي.",
            "CompanyStatements": "اقرأ النص التالي جيدًا، واستخلص أسلوبه العام في الكتابة. بعد ذلك اكتب نصًا جديدًا باللغة العربية عن موضوع مختلف هو: رسالة داخلية للموظفين حول تبني التحول الرقمي في الشركة، بحيث يحمل نفس الأسلوب العام دون نسخ الجمل أو تكرار المحتوى حرفيًا. اجعل طول النص الجديد قريبًا من طول النص الأصلي.",
        },
    }

    _PROMPT_DELIMITERS = {"en": "Original text:", "ar": "النص الأصلي:"}

    def __init__(self, lang: str = "ar"):
        super().__init__()
        self.lang = lang

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        csv_path = "/home/yifanmai/oss/helm/arabic-bench/writing_style/style_benchmark_company.csv"

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                print(row)
                category = row["category"]
                source_text = row["text"]
                text = f"{self._PROMPTS[self.lang][category]}\n\n{self._PROMPT_DELIMITERS[self.lang]} {source_text}"
                input = Input(text=text)
                instances.append(
                    Instance(
                        id=row["id"],
                        input=input,
                        references=[],
                        split=TEST_SPLIT,
                    )
                )

        return instances
