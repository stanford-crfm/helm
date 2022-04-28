import random
from typing import List, Dict, Tuple
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG

"""Synthetic Reasoning Scenario.

We define 3 synthetic reasoning tasks, "pattern matching", "variable substitution", "induction".
All 3 tasks build on three components: a rule string, a substitution dictionary and the final result.
As an example, we have:

Rule: A + B = B + A.
Substitution dictionary: {"A":"apple", "B":"peach"}
Result: "apple + peach = peach + apple"

The model hence is asked to do the following three tasks:

Pattern matching:
    Input: 4 rule strings, and a result string.
    Output: the matched rule string.

Variable substitution:
    Input: A rule string, a substitution dictionary.
    Output: the result string.

Induction:
    Input: Two result string that are induced by the same rule string.
    Output: the rule string.

"""

ANIMALS = ["zebra", "cobra", "stork", "penguin", "shark", "lion", "buffalo", "whale", "seal", "eagle", "horse", "rat"]
FRUITS = ["apple", "peach", "watermelon", "banana", "grape", "kiwi", "pear", "strawberry", "blueberry", "blackberry"]
RULE_SYMBOLS = ["X", "Y", "Z"]
MATH_SYMBOLS = ["+", "-", "*", "="]


def subst(pattern: List[str], rule_symbol: str, substitute_str: str) -> List[str]:
    """
    We substitute one rule symbols in a pattern according by a substitution str.

    example:
    pattern = "A+B=B+A"
    rule_symbol = "A"
    substitute_str = "apple"
    return: "apple+B=B+apple"

    :param pattern: A Pattern representing the rule.
    :param rule_symbol: One rule symbol.
    :param substitute_str: The substitution string.
    :return: The result of substitution.
    """
    assert rule_symbol in pattern
    # check which index is the same as rule_symbol
    indices = [i for i, x in enumerate(pattern) if x == rule_symbol]

    # form a new string with the symbol replaced
    new_string = pattern[: indices[0]] + [substitute_str]
    for i, j in zip(indices[:-1], indices[1:]):
        new_string += pattern[i + 1 : j]
        new_string += [substitute_str]
    new_string += pattern[indices[-1] + 1 :]

    return new_string


def pattern_subst(pattern: List[str], rule_symbols: List[str], substitute_dict: Dict[str, str]) -> List[str]:
    """
    We substitute the rule symbols in a pattern according to a substitution dictionary.

    example:
    pattern = "A+B=B+A"
    rule_symbols = ["A", "B"]
    substitute_dict = {"A":"apple", "B":"peach"}
    return: "apple+peach=peach+apple"

    :param pattern: A Pattern representing the rule.
    :param rule_symbols: The set of rule symbols.
    :param substitute_dict: The substitution dictionary.
    :return: The result of substitution.
    """

    out = pattern
    # we iteratively replace each rule symbol with its subsitution string
    for symbol in rule_symbols:
        out = subst(out, symbol, substitute_dict[symbol])
    return out


def gen_subst(rule_symbols: List[str], tokens: List[str]) -> Tuple[Dict[str, str], str]:
    """
    For each of the rule symbol, we sample a random substitution string composed of randomly sampled tokens.

    :param rule_symbols: A list of rule symbols.
    :param tokens: Tokens used to construct the substitution.
    :return: We return a substitution dictionary, and its string representation.
    """
    substitute_dict = {}
    substitute_str = []
    for char in rule_symbols:
        subst_len = random.randint(1, 2)
        subst = " ".join(random.choices(tokens, k=subst_len))
        substitute_dict.update({char: subst})
        substitute_str.append(char)
        substitute_str.append("by")
        substitute_str.append('"')
        substitute_str.append(subst)
        substitute_str.append('"')
        substitute_str.append(",")
    substitute_dict_str = " ".join(substitute_str[:-1])
    return substitute_dict, substitute_dict_str


def gen_pattern(math_symbols: List[str], rule_symbols: List[str]) -> List[str]:
    """
    Generate a pattern string

    Example Input: math_symbols: ["+", "-", "*"], rule_symbols: ["Y", "Y", "Z"]
    Example Output: ["Y", "Y", "+", "Z", "="]
    """
    num_math_symbols = random.choice(range(2, 4))
    pattern = rule_symbols + random.choices(math_symbols, k=num_math_symbols)
    random.shuffle(pattern)
    return pattern


class SyntheticReasoningScenario(Scenario):
    """
    Synthetic Reasoning benchmark inspired by
    "LIME: Learning Inductive Bias for Primitives of Mathematical Reasoning"
        https://arxiv.org/abs/2101.06223
    """

    name = "synthetic_reasoning"
    description = "Synthetic reasoning benchmark"
    tags = ["reasoning", "language", "pattern_matching"]

    def __init__(self, mode: str, random_seed=42):
        self.n_train_samples = 5
        self.n_valid_samples = 50
        self.n_test_samples = 50
        self.mode = mode
        assert self.mode in ["variable_substitution", "pattern_match", "induction"], f"Unsupported mode: {self.mode}"
        self.random_seed = random_seed

    def get_instances(self) -> List[Instance]:
        # We fix the seed for data generation to ensure reproducibility.
        random.seed(self.random_seed)
        # Read all the instances
        instances: List[Instance] = []

        rule_symbols = RULE_SYMBOLS
        tokens = ANIMALS + FRUITS
        math_symbols = MATH_SYMBOLS

        for sample_idx in range(self.n_train_samples + self.n_valid_samples + self.n_test_samples):

            # Sample number of rule symbols
            sampled_rule_symbols = random.choices(rule_symbols, k=random.choice(range(2, 5)))
            sampled_rule_symbols_set = list(set(sampled_rule_symbols))

            # generate the pattern
            pattern = gen_pattern(math_symbols, sampled_rule_symbols)

            # generate one substitution
            substitute_dict, substitute_dict_str = gen_subst(sampled_rule_symbols_set, tokens)
            result = pattern_subst(pattern, sampled_rule_symbols_set, substitute_dict)

            # generate another substitution
            substitute_dict_2, _ = gen_subst(sampled_rule_symbols_set, tokens)
            result_2 = pattern_subst(pattern, sampled_rule_symbols_set, substitute_dict_2)

            result_string = " ".join(result)
            pattern_string = " ".join(pattern)

            if self.mode == "induction":
                result_string_2 = " ".join(result_2)
                src = f"Two results: {result_string} | {result_string_2} "
                tgt = f"Rule: {pattern_string}"
            elif self.mode == "variable_substitution":
                src = f"Rules: {pattern_string} | Substitutions: {substitute_dict_str} "
                tgt = " ".join(result)
            elif self.mode == "pattern_match":
                # we sample 3 other rule strings as negatives for patterns matching.
                other_patterns = [" ".join(gen_pattern(math_symbols, sampled_rule_symbols_set)) for _ in range(3)]
                all_patterns = other_patterns + [pattern_string]
                random.shuffle(all_patterns)
                all_pattern_string = " | ".join(all_patterns)
                src = f"Rules: {all_pattern_string} | Result: {result_string} "
                tgt = pattern_string

            if sample_idx < self.n_train_samples:
                split = TRAIN_SPLIT
            elif sample_idx < self.n_train_samples + self.n_valid_samples:
                split = VALID_SPLIT
            else:
                split = TEST_SPLIT
            instance = Instance(input=src, references=[Reference(output=str(tgt), tags=[CORRECT_TAG])], split=split,)
            instances.append(instance)

        return instances
