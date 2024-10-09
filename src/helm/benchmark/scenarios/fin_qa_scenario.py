import os
import json
from typing import List

from helm.common.general import ensure_directory_exists, ensure_file_downloaded
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    TRAIN_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
)


DATASET_URL_PREFIX = "https://github.com/czyssrs/FinQA/raw/0f16e2867befa6840783e58be38c9efb9229d742/dataset/"
INSTRUCTIONS = """Presented with a financial report consisting of textual contents and a structured table, given a question, generate the reasoning program in the domain specific langauge (DSL) that will be executed to get the answer.

Respond with only the program in the DSL for the last question, without any preamble, elaboration, or working steps. Do not respond with anything that is not part of the DSL.

The DSL consists of mathematical operations and table operations as executable programs. The program consists of a sequence of operations. Each operation takes a list of arguments.

There are 6 mathematical operations: add, subtract, multiply, divide, greater, exp, and 4 table aggregation operations table-max, table-min, table-sum, table-average, that apply aggregation operations on table rows. The mathematical operations take arguments of either numbers from the given reports, or a numerical result from a previous step.

The table operations take arguments of table row names. We use the special token #n to denote the result from the nth step.

For example, in the example "divide(9413, 20.01), divide(8249, 9.48), subtract(#0, #1)", the program consists of 3 steps; The first and the second division steps take arguments from the table and the text, respectively, then the third step subtracts the results from the two previous steps.

Definitions of all operations:

[["Name", "Arguments", "Output", "Description"],
["add", "number1, number2", "number", "add two numbers: number1 + number2"],
["subtract", "number1, number2", "number", "subtract two numbers: number1 − number2"],
["multiply", "number1, number2", "number", "multiply two numbers: number1 * number2"],
["divide", "number1, number2", "number", "multiply two numbers: number1 / number2"],
["exp", "number1, number2", "number", "exponential: number1 ^ number2"],
["greater", "number1, number2", "bool", "comparison: number1 > number2"],
["table-sum", "table header", "number", "the summation of one table row"],
["table-average", "table header", "number", "the average of one table row"],
["table-max", "table header", "number", "the maximum number of one table row"],
["table-min", "table header", "number", "the minimum number of one table row"]]

Answer with only the program, without any additional explanation.
"""  # noqa: E501


class FinQAScenario(Scenario):
    """
    FinQA is a question answering task over financial reports that requires robust numerical reasoning.

    FinQA: A Dataset of Numerical Reasoning over Financial Data
    Paper: https://arxiv.org/abs/2109.00122
    Code: https://github.com/czyssrs/FinQA

    Presented with a financial report consisting of textual contents and a structured table, given a question,
    the task is togenerate the reasoning program in the domain specific langauge (DSL) that will be executed
    to get the answer.

    We add the sub-headers "Pre-table text", "Table", "Post-table text" to the input. Example:

    ```
    Pre-table text: printing papers net sales for 2006 decreased 3% ( 3 % ) from both 2005 and 2004 due principally...
    [more lines]
    Table: [["in millions", "2006", "2005", "2004"], ["sales", "$ 6930", "$ 7170", "$ 7135"], ["operating profit", "$ 677", "$ 473", "$ 508"]]
    Post-table text: u.s .
    uncoated papers net sales in 2006 were $ 3.5 billion , compared with $ 3.2 billion in 2005 and $ 3.3 billion in 2004 .
    [more lines]
    Question: brazilian paper sales represented what percentage of printing papers in 2005?
    Program:
    ```
    """  # noqa: E501

    name = "fin_qa"
    description = "FinQA"
    tags = ["question_answering", "financial"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path: str = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)
        # Note: only train and test splits are used; dev split is not used
        instances: List[Instance] = []
        for split in [TRAIN_SPLIT, TEST_SPLIT]:
            file_name = f"{split}.json"
            target_path = os.path.join(data_path, file_name)
            ensure_file_downloaded(
                source_url=DATASET_URL_PREFIX + file_name,
                target_path=target_path,
            )
            with open(target_path, "r") as f:
                rows = json.load(f)
                for row in rows:
                    pre_text = "Pre-table text: " + "\n".join(row["pre_text"])
                    table = "Table: " + json.dumps(row["table"])
                    post_text = "Post-table text: " + "\n".join(row["post_text"])
                    question = "Question: " + row["qa"]["question"]
                    text = "\n".join([pre_text, table, post_text, question])
                    references = [
                        Reference(
                            Output(text=str(row["qa"]["program"])),
                            tags=[CORRECT_TAG],
                        ),
                        Reference(
                            Output(text=str(row["qa"]["exe_ans"])),
                            tags=[],
                        ),
                        Reference(
                            Output(text=json.dumps(row["table"])),
                            tags=[],
                        ),
                    ]
                    instance: Instance = Instance(
                        input=Input(text=text),
                        references=references,
                        split=split,
                    )
                    instances.append(instance)
        return instances
