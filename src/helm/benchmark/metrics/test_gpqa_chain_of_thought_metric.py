from helm.benchmark.metrics.gpqa_chain_of_thought_metric import extract_answer


def test_extract_answer_correct_answer_is_format():
    # Primary format requested by the GPQA chain-of-thought prompt.
    assert extract_answer("The correct answer is (A)") == "A"
    assert extract_answer("The correct answer is B") == "B"


def test_extract_answer_answer_colon_format():
    # Fallback format: "Answer: X" / "answer: X" (previously broken by the
    # literal-escaped brackets in the regex, which only matched "[aA]nswer:").
    assert extract_answer("Answer: C") == "C"
    assert extract_answer("The answer: B") == "B"
    assert extract_answer("answer: E") == "E"
    assert extract_answer("...Answer: D") == "D"
    assert extract_answer("[answer: F]") == "F"


def test_extract_answer_no_match():
    assert extract_answer("I am not sure.") is None
