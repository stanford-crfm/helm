import json
import os
import re
from typing import List, Dict, Any, Optional

from helm.common.request import Request
from helm.benchmark.model_deployment_registry import get_default_model_deployment_for_model
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.common.hierarchical_logger import hlog
from helm.common.general import write
from helm.benchmark.run_spec import RunSpec


class LLMJudger:
    """
    Class to judge model predictions.
    """

    def __init__(self, executor_service, judge_model: str = "openai/gpt2", prompt_file: str = "default_prompt.txt"):
        self.executor_service = executor_service
        self.judge_model = judge_model
        self.prompt_file = prompt_file

    def judge_predictions(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply judgment to a list of predictions.
        Each prediction should be a dictionary with keys "input", "prediction", and optionally "instance_id".
        Returns a list of dictionaries with the same keys plus "judgement" and "explanation".
        """
        judgements = []

        for prediction in predictions:
            input_text = prediction.get("input", "")
            model_response = prediction.get("prediction", "")

            prompt_tamplate = self._load_prompt_template(self.prompt_file)
            prompt = prompt_tamplate.replace("{input}", input_text).replace("{response}", model_response)

            judged_value, explanation = self.call_llm(prompt)

            judgements.append(
                {
                    "instance_id": prediction.get("instance_id"),
                    "input": input_text,
                    "prediction": model_response,
                    "judgement": judged_value,
                    "explanation": explanation,
                }
            )

        return judgements

    def call_llm(self, prompt: str) -> tuple[int, str]:
        """
        Call the LLM judge with the given prompt and return the judgement and explanation.
        The prompt should be a string formatted according to the judge model's requirements.
        Returns a tuple (judgement, explanation).
        """
        request = Request(
            model=self.judge_model,
            model_deployment=self._resolve_model_deployment(),
            prompt=prompt,
            temperature=0.7,
            max_tokens=300,
        )

        result = self.executor_service.make_request(request)

        if not result.success:
            raise Exception(f"LLM Judge request failed: {result.error}")

        if result.completions:
            text = result.completions[0].text.strip()

            # Extract fields from JSON-like output using regular expressions
            judgement_match = re.search(r'"judgement"\s*:\s*(\d)', text)
            explanation_match = re.search(r'"explanation"\s*:\s*"(.+?)"', text, re.DOTALL)

            if judgement_match and explanation_match:
                judgement = int(judgement_match.group(1))
                explanation = explanation_match.group(1).strip()
                return judgement, explanation
            else:
                print("WARNING: Could not extract expected fields.")
                print("Raw model output:\n", text)
                return 0, "Malformed or incomplete response."

        return 0, "No response from LLM judge."

    def judge(
        self, predictions: List[Dict[str, Any]], instructions: str = "", scenario_state: Optional[ScenarioState] = None
    ) -> List[Dict[str, Any]]:
        """
        Apply LLM judgment and return all complete judgments (without saving to file).
        For multiple choice benchmarks, include lettered options as context above the prompt.
        """
        judgements = []

        is_multiple_choice = scenario_state and scenario_state.adapter_spec.method == "multiple_choice_joint"

        for i, prediction in enumerate(predictions):
            input_text = prediction.get("input", "")
            model_response = prediction.get("prediction", "")

            context_parts = []

            if instructions.strip():
                context_parts.append(f"Benchmark instructions:\n{instructions.strip()}")

            if is_multiple_choice:
                try:
                    if scenario_state is not None:
                        output_mapping = scenario_state.request_states[i].output_mapping
                        if output_mapping:
                            choices_text = "Alternatives:\n" + "\n".join(
                                f"{k}. {v.strip()}" for k, v in output_mapping.items()
                            )
                            context_parts.append(choices_text)
                except Exception as e:
                    hlog(f"Error extracting alternatives for instance {i}: {e}")

            context_block = "\n\n".join(context_parts).strip()
            if context_block:
                context_block += "\n\n"

            prompt_template = self._load_prompt_template(self.prompt_file)

            prompt = context_block + prompt_template.replace("{input}", input_text).replace(
                "{response}", model_response
            )

            judged_value, explanation = self.call_llm(prompt)

            judgements.append(
                {
                    "instance_id": prediction.get("instance_id"),
                    "input": input_text,
                    "prediction": model_response,
                    "judgement": judged_value,
                    "explanation": explanation,
                }
            )

        return judgements

    def _resolve_model_deployment(self) -> str:
        """
        Resolve the model deployment for the judge model.
        Returns the name of the model deployment.
        Raises an exception if the deployment cannot be found.
        """
        deployment_name = get_default_model_deployment_for_model(self.judge_model)
        if not deployment_name:
            raise Exception(
                f"Could not find a model deployment for judge model '{self.judge_model}'. "
                f"Make sure the model is correctly registered in the HELM model YAML."
            )
        return deployment_name

    def _load_prompt_template(self, prompt_file: str) -> str:
        """
        Load the prompt template from a .txt file located in prompts/
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, "prompts", prompt_file)

        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file '{prompt_path}' not found. Make sure the file exists.")

        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()


def extract_predictions(scenario_state: ScenarioState) -> List[Dict[str, Any]]:
    predictions = []
    for request_state in scenario_state.request_states:
        instance = request_state.instance
        result = request_state.result
        if result is None or not result.completions:
            continue
        completion = result.completions[0]
        prediction = {
            "instance_id": instance.id,
            "input": getattr(instance.input, "text", ""),
            "prediction": getattr(completion, "text", None),
        }
        predictions.append(prediction)
    return predictions


def save_judge_summary(run_spec: RunSpec, run_path: str, judge_model: str, judgements: List[Dict[str, Any]]):
    valid_judgements = [j for j in judgements if j.get("explanation") != "Malformed or incomplete response."]
    total_valid = len(valid_judgements)
    total_judged = len(judgements)
    agreements = sum(1 for j in judgements if j.get("judgement") == 1)
    invalid_instances = total_judged - total_valid
    agreement_level = agreements / total_valid if total_valid > 0 else 0.0

    summary = {
        "benchmark": run_spec.name,
        "main_model": run_spec.adapter_spec.model,
        "judge_model": judge_model,
        "agreement_level": round(agreement_level, 4),
        "agreements": agreements,
        "total_valid_instances": total_valid,
        "total_judged_instances": total_judged,
        "invalid_instances": invalid_instances,
        "tasks": judgements,
    }

    output_file = os.path.join(run_path, "llm_judge_summary.json")
    write(output_file, json.dumps(summary, indent=2, ensure_ascii=False))
    hlog(f"Saved LLM Judge summary to {output_file}")
    print(f"\n✅ LLM-Judge Agreement Level: {round(agreement_level * 100, 2)}% " f"({agreements}/{total_valid})\n")
