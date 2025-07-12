from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, Output, Reference, VALID_SPLIT, CORRECT_TAG
import pandas as pd
import requests


class CodeInsightsStudentMistakeScenario(Scenario):
    name = "codeinsights_student_mistake"
    description = "Mimic how students mistake their C++ codes on foundational questions"
    tags = ["codeinsights", "c++", "student_mistake"]

    def __init__(self, num_testcases: int = 1):
        super().__init__()
        self.num_testcases = num_testcases

    def get_instances(self, output_path: str):
        df = pd.read_csv("https://huggingface.co/datasets/Kazchoko/my_dataset/resolve/main/Scenario3_data.csv")
        student_topic = pd.read_csv(
            "https://huggingface.co/datasets/Kazchoko/my_dataset/resolve/main/student_performace_by_topic.csv"
        )

        instances = []
        for student_id, student_df in df.groupby("student_id"):
            student_df = student_df.sort_values(by=["student_id", "question_unittest_id", "timestamp"])
            if len(student_df) < 4:
                continue
            first = student_df.iloc[0]
            second = student_df.iloc[1]
            third = student_df.iloc[2]
            target = student_df.iloc[3]

            # Get test cases for this question
            question_id = target.get("question_unittest_id", None)
            question_test_cases = []
            tc_parsing_success = True

            for testcase_str in target["question_unittests"].split("Unittest")[1:]:
                testcase_str = testcase_str[testcase_str.find(":") + 1 :]
                input_idx = testcase_str.find("Input:")
                std_in_idx = testcase_str.find("STD input:")
                output_idx = testcase_str.find("Output:")
                if input_idx == -1 or std_in_idx == -1 or output_idx == -1:
                    tc_parsing_success = False
                    break

                testcase = {
                    "input": testcase_str[input_idx + 6 : std_in_idx].strip(),
                    "std_in": testcase_str[std_in_idx + 10 : output_idx].strip(),
                    "output": testcase_str[output_idx + 7 :].strip(),
                }
                question_test_cases.append(testcase)

            if not tc_parsing_success:
                continue

            if len(question_test_cases) < self.num_testcases:
                # If not enough test cases, skip this question
                continue
            if self.num_testcases >= 0:
                # If more than one test case is requested, only take the first ones
                question_test_cases = question_test_cases[: self.num_testcases]

            # Get student pass (0 or 1) for the target question
            student_correctness_pattern = target.get("pass", None)
            main_part = int(student_correctness_pattern)  # "1111111111"
            # Convert each character to an int
            student_correctness_list = [int(ch) for ch in str(main_part)]  # [1,1,1,1,1,1,1,1,1,1]

            # Student specific topic performance in previous attempts
            student_level_prompt = f"Student {student_id} has the following performance across topics:\n"
            topic_performance = student_topic[student_topic["student_id"] == student_id]
            for _, row in topic_performance.iterrows():
                topic = row["topic"]
                pass_rate = round(row["pass_rate"], 2)
                perfect = round(row["perfect"], 2)

                student_level_prompt += (
                    f"- For topic '{topic}', the unit test pass rate is {pass_rate}, "
                    f"and the rate of passing all unit tests is {perfect}.\n"
                )

            prompt = (
                "=== Student Profile ===\n"
                f"{student_level_prompt}\n"
                "When students submit a code to the platform, it will be tested by number of unit tests, where"
                "- Unit test pass rate = proportion of unit tests passed with the code \n"
                "- Full pass rate   = proportion of code passing all unit tests\n\n"
                "=== Past Mistake Examples ===\n"
                "Example 1 (Week {first['week']}, Topic: {first['topic']}):\n"
                f"Question: {first['question_name']} — {first['question_text']}\n"
                "Template:\n"
                f"{first['question_template']}\n"
                "Student's Response Code with Error:\n"
                f"{first['response_mistake']}\n\n"
                "Example 2 (Week {second['week']}, Topic: {second['topic']}):\n"
                f"Question: {second['question_name']} — {second['question_text']}\n"
                "Template:\n"
                f"{second['question_template']}\n"
                "Student's Response Code with Error:\n"
                f"{second['response_mistake']}\n\n"
                "Example 3 (Week {third['week']}, Topic: {third['topic']}):\n"
                f"Question: {third['question_name']} — {third['question_text']}\n"
                "Template:\n"
                f"{third['question_template']}\n"
                "Student's Response Code with Error:\n"
                f"{third['response_mistake']}\n\n"
                "=== New Target Problem ===\n"
                f"Week: {target['week']}, Topic: {target['topic']}\n"
                f"Question: {target['question_name']} — {target['question_text']}\n"
                f"Unit Test Input: {question_test_cases}\n\n"
                if question_test_cases
                else ""
                "Template:\n"
                f"{target['question_template']}\n\n"
                "⚠**Instructions:**\n"
                "1. Mimic your own coding style, naming conventions, indentation, and typical error patterns.\n"
                "2. Introduce mistake you are likely to make (e.g., off‐by‐one index, wrong initialization, "
                "missing edge case).\n"
                "3. Do **not** produce a fully correct solution or add unfamiliar optimizations.\n\n"
                "Provide ONLY your C++ implementation following the given template, where the answer will replace the {{ STUDENT_ANSWER }} block in the template. "
                "DO NOT reproduce the template part as the generated code would be inserted to the template, "
                "and make sure the code is compatible with the Unit Test Input. "
                "int main() is always declared already so DO NOT produce that initialization on the code. "
                "Ensure your code is includes any class definition when needed. "
                "Return the code in C++ code block format, and nothing else."
            )

            print(f"\n=== DEBUG INFO FOR STUDENT {student_id}, QUESTION {question_id} ===")
            print(f"Test cases loaded: {len(question_test_cases)}")
            print(f"Student correctness pattern: {student_correctness_list}")
            print(f"Original pass field: {target.get('pass', 'MISSING')}")
            print(f"Question template exists: {'question_template' in target}")
            print(f"Question name: {target.get('question_name', 'MISSING')}")

            # Also add this validation in your UnitTestAlignmentMetric evaluate_generation method:
            def evaluate_generation(self, adapter_spec, request_state, metric_service, eval_cache_path):
                print("\n=== UNIT TEST METRIC DEBUG ===")
                print(f"Has extra_data: {hasattr(request_state.instance, 'extra_data')}")
                if hasattr(request_state.instance, "extra_data"):
                    extra_data = request_state.instance.extra_data
                    print(f"Extra data keys: {list(extra_data.keys())}")
                    print(f"Test cases: {len(extra_data.get('test_cases', []))}")
                    print(f"Student pattern: {extra_data.get('student_correctness_pattern', 'MISSING')}")

            instances.append(
                Instance(
                    id=f"{student_id}_{target['question_unittest_id']}",
                    input=Input(text=prompt),
                    references=[Reference(output=Output(text=target["response_mistake"]), tags=[CORRECT_TAG])],
                    extra_data={
                        "question_template": target["question_template"],
                        "test_cases": question_test_cases,
                        "question_id": str(question_id) if question_id else None,
                        "question_name": target.get("question_name", ""),
                        "student_id": str(student_id),
                        "student_correctness_pattern": student_correctness_list,
                    },
                    split=VALID_SPLIT,
                )
            )
        return instances

    def _load_test_cases(self):
        """
        Load test cases from external source or return None if not available.
        This method should be implemented based on where your test cases are stored.

        Expected format:
        {
            "question_id": [
                {
                    "unittest": "test_id",
                    "input": "test input code",
                    "output": "expected output"
                },
                ...
            ],
            ...
        }
        """
        try:
            response = requests.get(
                "https://huggingface.co/datasets/Kazchoko/my_dataset/resolve/main/test_cases_by_qid.json"
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Failed to load test cases from URL: {e}")
            return {}
