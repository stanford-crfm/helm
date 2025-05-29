from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference, VALID_SPLIT, CORRECT_TAG
)
import json, os
import pandas as pd

class StudentCodingScenario(Scenario):
    name = "student_coding"
    description = "Mimic student C++ style on foundational questions"
    tags = ["coding", "c++", "student"]

    def get_instances(self, output_path: str):
        df = pd.read_csv("https://huggingface.co/datasets/Kazchoko/my_dataset/resolve/main/sample_fifty_student_full.csv")
        instances = []
        for student_id, student_df in df.groupby("student_id"):
            student_df = student_df.sort_values("timestamp")
            if len(student_df) < 4:
                continue
            first = student_df.iloc[0]
            second = student_df.iloc[1]
            third = student_df.iloc[2]
            target = student_df.iloc[3]
            prompt = (
                "You are the same student who wrote the three examples below in your foundational C++ course. "
                "Mimic exactly your personal coding style, conventions, and level of proficiency—"
                "do not over‐optimize or introduce unfamiliar patterns. "
                "Include the same sort of formatting, variable names, and minor imperfections you demonstrated. "
                "Respond ONLY with the C++ code (no commentary).\n\n"
                f"Week: {target['week']}\n"
                f"Topic: {target['topic']}\n\n"
                "Example 1:\n"
                f"Question: {first['question_name']} — {first['question_text']}\n"
                "Template:\n"
                f"{first['question_template']}\n"
                "Your Code:\n"
                f"{first['response']}\n\n"
                "Example 2:\n"
                f"Question: {second['question_name']} — {second['question_text']}\n"
                "Template:\n"
                f"{second['question_template']}\n"
                "Your Code:\n"
                f"{second['response']}\n\n"
                "Example 3:\n"
                f"Question: {third['question_name']} — {third['question_text']}\n"
                "Template:\n"
                f"{third['question_template']}\n"
                "Your Code:\n"
                f"{third['response']}\n\n"
                "Now, using that same student style, attempt this:\n"
                f"Question: {target['question_name']} — {target['question_text']}\n"
                "Template:\n"
                f"{target['question_template']}\n\n"
                "Provide ONLY your C++ implementation following the given template, "
                "writing code just as you would in class—indentation, naming, and all."
            )
            instances.append(
                Instance(
                    id=f"{student_id}_{target['question_unittest_id']}",
                    input=Input(text=prompt),
                    references=[Reference(
                        output=Output(text=target["response"]),
                        tags=[CORRECT_TAG]
                    )],
                    extra_data={"question_template": target["question_template"]},
                    split=VALID_SPLIT,
                )
            )
        return instances