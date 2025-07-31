from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, Output, Reference, VALID_SPLIT, CORRECT_TAG
import pandas as pd
import requests


class CodeInsightsEdgeCaseScenario(Scenario):
    name = "codeinsights_edge_case"
    description = "Evaluate alignment in edge case failure between LLM-generated code and student code"
    tags = ["codeinsights", "c++", "edge_case"]

    def __init__(self, num_testcases: int = 1):
        super().__init__()
        self.num_testcases = num_testcases

    def get_instances(self, output_path: str):
        df = pd.read_csv("https://huggingface.co/datasets/Kazchoko/my_dataset/resolve/main/Scenario5_data.csv")

        student_topic = pd.read_csv(
            "https://huggingface.co/datasets/Kazchoko/my_dataset/resolve/main/student_performace_by_topic.csv"
        )

        # Load test cases (unit tests)
        test_cases = self._load_test_cases()

        # Get available question IDs with test cases
        available_question_ids = set()
        if test_cases:
            available_question_ids = set(test_cases.keys())
            print(f"Loaded test cases for {len(available_question_ids)} questions")
        else:
            print("WARNING: No test cases loaded!")
            return []

        instances = []
        skipped_no_tests = 0
        skipped_insufficient_data = 0

        for student_id, student_df in df.groupby("student_id"):
            student_df = student_df.sort_values("timestamp")
            target = student_df.iloc[0]

            # Check if target question has test cases BEFORE processing
            target_question_id = target.get("question_unittest_id", None)
            if not target_question_id or str(target_question_id) not in available_question_ids:
                skipped_no_tests += 1
                print(f"SKIPPING Student {student_id}, Question {target_question_id}: No test cases available")
                continue

            # Get test cases for this question (we know they exist now)
            target_test_cases = []
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
                target_test_cases.append(testcase)

            # Verify test cases are not empty
            if not tc_parsing_success:
                skipped_no_tests += 1
                print(f"SKIPPING Student {student_id}, Question {target_question_id}: Empty test cases")
                continue

            if len(target_test_cases) < self.num_testcases:
                # If not enough test cases, skip this question
                continue
            if self.num_testcases >= 0:
                # If more than one test case is requested, only take the first ones
                target_test_cases = target_test_cases[: self.num_testcases]

            # Get student pass pattern for the target question
            student_correctness_pattern = target.get("pass", None)
            if student_correctness_pattern is not None:
                main_part = int(student_correctness_pattern)
                # Convert each character to an int
                student_correctness_list = [int(ch) for ch in str(main_part)]
            else:
                student_correctness_list = []

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

            print(f"\n=== ACCEPTED INSTANCE: Student {student_id}, Question {target_question_id} ===")
            print(f"Test cases loaded: {len(target_test_cases)}")
            print(f"Student correctness pattern: {student_correctness_list}")
            print(f"Question name: {target.get('question_name', 'MISSING')}")

            prompt = (
                "You are analyzing a student’s likely mistakes on an upcoming programming problem.\n"
                "Your task: **predict exactly ONE unit-test index (0-based) that the student is most likely to fail.**\n"  # noqa: E501
                "Return *only* that integer. No explanation.\n\n"
                "=== Student Profile ===\n"
                f"{student_level_prompt}\n"
                "For the given programming question, identify which unit test the student would fail considering "
                "their past performance, as well as consideration of unit test difficulty.\n"
                f"Week: {target['week']}\n"
                f"Topic: {target['topic']}\n\n"
                f"Question: {target['question_name']} — {target['question_text']}\n"
                f"Unit Tests: {target_test_cases}\n"
                if target_test_cases
                else ""
                "Think silently about:\n"
                "• Which test seems hardest for the given topic?\n"
                "• Where has the student historically struggled?\n"
                "• Any edge-cases in the tests’ inputs/outputs?\n\n"
                "******** ANSWER FORMAT (STRICT) ********\n"
                "<integer>\n"
                "****************************************"
            )

            instances.append(
                Instance(
                    id=f"{student_id}_{target['question_unittest_id']}",
                    input=Input(text=prompt),
                    references=[Reference(output=Output(text=target["response"]), tags=[CORRECT_TAG])],
                    extra_data={
                        "question_template": target["question_template"],
                        "test_cases": target_test_cases,
                        "question_id": str(target_question_id),
                        "question_name": target.get("question_name", ""),
                        "student_id": str(student_id),
                        "student_correctness_pattern": student_correctness_list,
                    },
                    split=VALID_SPLIT,
                )
            )

        # Print summary statistics
        print("\n=== INSTANCE CREATION SUMMARY ===")
        print(f"Skipped (insufficient data): {skipped_insufficient_data}")
        print(f"Skipped (no test cases): {skipped_no_tests}")
        print(f"Available test case question IDs: {len(available_question_ids)}")

        if len(instances) >= 5:
            print("Sample created instances:")
            for i, inst in enumerate(instances[:5]):
                if inst.extra_data is None:
                    test_count = 0
                else:
                    test_count = len(inst.extra_data.get("test_cases", []))
                print(f"  {inst.id}: {test_count} test cases")

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
