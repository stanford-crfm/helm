from typing import Any

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.hierarchical_logger import hwarn
from helm.common.request import Request


class ALRAGEAnnotator(Annotator):
    """The ALRAGEAnnotator autograder."""

    name = "alrage"

    _ANNOTATOR_MODEL = "openai/gpt-4o-2024-11-20"

    def __init__(self, auto_client: AutoClient):
        self._auto_client = auto_client

    def _apply_annotator_template(self, question: str, answer: str, gold: str) -> Any:
        return [
            {
                "role": "system",
                "content": """أنت مقيّم محايد خبير باللغة العربية. يجب عليك:
        1. تقييم دقة الإجابة مقارنة بالإجابة الصحيحة
        2. التحقق من أن الإجابة مدعومة بالسياق المقدم
        3. تقييم جودة وشمولية الإجابة

        مهم جداً: يجب أن يكون ردك رقماً فقط من 0 إلى 10. لا تضف أي نص أو تفسير.""",
            },
            {
                "role": "user",
                "content": f"""السؤال: {question}

        الإجابة المقدمة: {answer}

        الإجابة الصحيحة: {gold}

        أعط تقييماً من 0 إلى 10:
        0-2: إجابة خاطئة تماماً
        3-4: إجابة جزئية مع أخطاء
        5-6: إجابة متوسطة
        7-8: إجابة جيدة
        9-10: إجابة ممتازة

        اكتب رقماً فقط من 0 إلى 10 بدون أي نص إضافي:""",
            },
        ]

    def _parse_annotator_response(self, response: str) -> float:
        """Process the judge's response to extract the score"""
        try:
            # Extract the first number from the response content
            score = float(next(num for num in response.split() if num.replace(".", "", 1).isdigit()))
            return min(max(score / 10.0, 0.0), 1.0)

        except Exception as e:
            hwarn(f"Error while processing judge response: {e}")
            return 0.0

    def annotate(self, request_state: RequestState) -> Any:
        question = request_state.instance.input.text
        assert request_state.result
        assert len(request_state.result.completions) == 1
        answer = request_state.result.completions[0].text
        assert len(request_state.instance.all_correct_references) == 1
        gold = request_state.instance.all_correct_references[0].output.text
        messages = self._apply_annotator_template(question, answer, gold)
        judge_request = Request(
            model=self._ANNOTATOR_MODEL,
            model_deployment=self._ANNOTATOR_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=2000,
        )
        judge_response = self._auto_client.make_request(judge_request)
        if not judge_response.success:
            raise Exception(
                "ALRAGEAnnotator got an error response from " f"{self._ANNOTATOR_MODEL}: {judge_response.error}"
            )
        assert len(judge_response.completions) == 1
        prompt = messages[-1]["content"]
        response = judge_response.completions[0].text
        score = self._parse_annotator_response(response)

        return {
            "prompt": prompt,
            "response": response,
            "score": score,
        }
