from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, VALID_SPLIT
import pandas as pd


class CodeInsightsCorrectCodeScenario(Scenario):
    name = "codeinsights_correct_code"
    description = "Generate correct response code for C++ programming questions"
    tags = ["codeinsights", "c++", "correct_code"]

    def __init__(self, num_testcases: int = 1):
        super().__init__()
        self.num_testcases = num_testcases

    def get_instances(self, output_path: str):
        df = pd.read_csv("https://huggingface.co/datasets/Kazchoko/my_dataset/resolve/main/Scenario1_2_data.csv")

        # Load test cases (unit tests)
        instances = []
        for question_id, question_df in df.groupby("question_unittest_id"):
            target = question_df.iloc[0]
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

            prompt = (
                f"Question: {target['question_name']} — {target['question_text']}\n\n"
                f"Unit Test Input: {question_test_cases}\n\n"
                if question_test_cases
                else ""
                "Template:\n"
                f"{target['question_template']}\n\n"
                "Provide ONLY your C++ implementation following the given template, where the answer will replace the {{ STUDENT_ANSWER }} block in the template. "
                "DO NOT reproduce the template part as the generated code would be inserted to the template, "
                "and make sure the code is compatible with the Unit Test Input. "
                "int main() is always declared already so DO NOT produce that initialization on the code. "
                "Ensure your code is correct, efficient, includes any class definition when needed, and handles all edge cases properly. "
                "Return the code in C++ code block format, and nothing else."
            )
            instances.append(
                Instance(
                    id=f"{question_id}",
                    input=Input(text=prompt),
                    references=[],
                    extra_data={
                        "question_template": target["question_template"],
                        "test_cases": question_test_cases,
                        "question_id": str(question_id) if question_id else None,
                        "question_name": target.get("question_name", ""),
                    },
                    split=VALID_SPLIT,
                )
            )
        return instances
