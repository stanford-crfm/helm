import collections
import os
import typing
from typing import Dict, List, Optional
from datasets import load_dataset, DatasetDict

from helm.common.general import ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


def remove_boxed(string: str) -> Optional[str]:
    """Source: https://github.com/hendrycks/math

    Extract the text within a \\boxed{...} environment.

    Example:
    >>> remove_boxed(\\boxed{\\frac{2}{3}})
    \\frac{2}{3}
    """
    left = "\\boxed{"
    try:
        assert string[: len(left)] == left
        assert string[-1] == "}"
        return string[len(left) : -1]
    except Exception:
        return None


def last_boxed_only_string(string: str) -> Optional[str]:
    """Source: https://github.com/hendrycks/math

    Extract the last \\boxed{...} or \\fbox{...} element from a string.
    """
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def _fix_fracs(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Reformat fractions.

    Examples:
    >>> _fix_fracs("\\frac1b")
    \frac{1}{b}
    >>> _fix_fracs("\\frac12")
    \frac{1}{2}
    >>> _fix_fracs("\\frac1{72}")
    \frac{1}{72}
    """
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Reformat fractions formatted as a/b to \\frac{a}{b}.

    Example:
    >>> _fix_a_slash_b("2/3")
    \frac{2}{3}
    """
    if len(string.split("/")) != 2:
        return string
    a_str = string.split("/")[0]
    b_str = string.split("/")[1]
    try:
        a = int(a_str)
        b = int(b_str)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _remove_right_units(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Remove units (on the right).
    "\\text{ " only ever occurs (at least in the val set) when describing units.
    """
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Reformat square roots.

    Example:
    >>> _fix_sqrt("\\sqrt3")
    \sqrt{3}
    """
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Apply the reformatting helper functions above.
    """
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
    # Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def get_answer(solution: Optional[str]) -> Optional[str]:
    if solution is None:
        return None
    last_boxed = last_boxed_only_string(solution)
    if last_boxed is None:
        return None
    answer = remove_boxed(last_boxed)
    if answer is None:
        return None
    return answer


def is_equiv(str1: Optional[str], str2: Optional[str]) -> float:
    """Returns (as a float) whether two strings containing math are equivalent up to differences of formatting in
    - units
    - fractions
    - square roots
    - superfluous LaTeX.

    Source: https://github.com/hendrycks/math
    """
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return 1.0
    if str1 is None or str2 is None:
        return 0.0

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        return float(ss1 == ss2)
    except Exception:
        return float(str1 == str2)


def is_equiv_chain_of_thought(str1: str, str2: str) -> float:
    """Strips the solution first before calling `is_equiv`."""
    ans1 = get_answer(str1)
    ans2 = get_answer(str2)

    return is_equiv(ans1, ans2)


class MATHScenario(Scenario):
    """
    The MATH dataset from the paper
    "Measuring Mathematical Problem Solving With the MATH Dataset"
    by Hendrycks et al. (2021):
    https://arxiv.org/pdf/2103.03874.pdf

    Example input, using official examples:

    ```
    Given a mathematics problem, determine the answer. Simplify your answer as much as possible.
    ###
    Problem: What is $\left(\frac{7}{8}\right)^3 \cdot \left(\frac{7}{8}\right)^{-3}$?
    Answer: $1$
    ###
    Problem: In how many ways can 4 books be selected from a shelf of 6 books if the order in which the books are selected does not matter?
    Answer: $15$
    ###
    Problem: Find the distance between the points $(2,1,-4)$ and $(5,8,-3).$
    Answer: $\sqrt{59}$
    ###
    Problem: The faces of an octahedral die are labeled with digits $1$ through $8$. What is the probability, expressed as a common fraction, of rolling a sum of $15$ with a pair of such octahedral dice?
    Answer: $\frac{1}{32}$
    ###
    Problem: The first three terms of an arithmetic sequence are 1, 10 and 19, respectively. What is the value of the 21st term?
    Answer: $181$
    ###
    Problem: Calculate $6 \cdot 8\frac{1}{3}
    Answer: $50$
    ###
    Problem: When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?
    Answer: $2$
    ###
    Problem: How many zeros are at the end of the product 25 $\times$ 240?
    Answer: $3$
    ###
    Problem: What is $\dbinom{n}{n}$ for any positive integer $n$?
    Answer: $
    ```

    Example expected output

    ```
    1$
    ```
    """  # noqa

    name = "MATH"
    description = "Mathematical Problem Solving"
    tags = ["knowledge", "reasoning"]

    subjects_mapping = {
        "number_theory": "Number Theory",
        "intermediate_algebra": "Intermediate Algebra",
        "algebra": "Algebra",
        "prealgebra": "Prealgebra",
        "geometry": "Geometry",
        "counting_and_probability": "Counting & Probability",
        "precalculus": "Precalculus",
    }
    levels = ["1", "2", "3", "4", "5"]

    def __init__(
        self, subject: str, level: str, use_official_examples: bool = False, use_chain_of_thought: bool = False
    ):
        super().__init__()
        self.subject: str = MATHScenario.subjects_mapping[subject]
        self.level: str = f"Level {level}"
        self.use_official_examples: bool = use_official_examples
        self.use_chain_of_thought: bool = use_chain_of_thought
        if use_chain_of_thought:
            assert not use_official_examples, "Cannot use official examples when use_chain_of_thought is True."

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = {}
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        data = (
            typing.cast(
                DatasetDict,
                load_dataset(
                    "hendrycks/competition_math",
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    revision="71b758ecc688b2822d07ffa7f8393299f1dc7cac",
                ),
            )
            .sort("problem")
            .shuffle(seed=42)
        )

        def group_by_key(dataset_list, key):
            dataset_per_key = collections.defaultdict(list)
            for ex in dataset_list:
                dataset_per_key[ex[key]].append(ex)
            return dataset_per_key

        instances = []
        for split, split_name in zip([TRAIN_SPLIT, TEST_SPLIT], ["train", "test"]):
            if split == TRAIN_SPLIT and self.use_official_examples:
                train_instances = [
                    ("What is $\left(\\frac{7}{8}\\right)^3 \cdot \left(\\frac{7}{8}\\right)^{-3}$?", "1"),
                    (
                        "In how many ways can 4 books be selected from a shelf of 6 books"
                        + " if the order in which the books are selected does not matter?",
                        "15",
                    ),
                    ("Find the distance between the points $(2,1,-4)$ and $(5,8,-3).$", "\sqrt{59}"),
                    (
                        "The faces of an octahedral die are labeled with digits $1$ through $8$."
                        + " What is the probability, expressed as a common fraction,"
                        + " of rolling a sum of $15$ with a pair of such octahedral dice?",
                        "\\frac{1}{32}",
                    ),
                    (
                        "The first three terms of an arithmetic sequence are 1, 10 and 19, respectively."
                        + " What is the value of the 21st term?",
                        "181",
                    ),
                    ("Calculate $6 \\cdot 8\\frac{1}{3}", "50"),
                    (
                        "When the binary number $100101110010_2$ is divided by 4,"
                        + " what is the remainder (give your answer in base 10)?",
                        "2",
                    ),
                    ("How many zeros are at the end of the product 25 $\\times$ 240?", "3"),
                ]
                dataset[TRAIN_SPLIT] = [{"problem": problem, "answer": answer} for problem, answer in train_instances]

            else:
                examples: List[Dict[str, str]] = [example for example in data[split_name]]  # Filter by split
                examples = group_by_key(examples, "type")[self.subject]  # Filter by type or subject
                examples = group_by_key(examples, "level")[self.level]  # Filter by level
                dataset[split] = examples

                for example in dataset[split]:
                    # Sanity check that we filtered correctly
                    assert (
                        example["type"] == self.subject and example["level"] == self.level
                    ), f"Wrong example was included after filtering: {example}"

                    if self.use_chain_of_thought:
                        answer = example["solution"]
                    else:
                        maybe_answer = get_answer(example["solution"])
                        if maybe_answer is None:
                            continue
                        answer = maybe_answer
                    example["answer"] = answer

            for example in dataset[split]:
                instance = Instance(
                    input=Input(text=example["problem"]),
                    references=[Reference(Output(text=example["answer"]), tags=[CORRECT_TAG])],
                    split=split,
                )
                instances.append(instance)

        return instances
