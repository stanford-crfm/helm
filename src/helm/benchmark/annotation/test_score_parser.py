import unittest

from helm.benchmark.annotation.score_parser import (
    find_last_match,
    find_last_tagged_match,
)


class ScoreParserTest(unittest.TestCase):
    def test_uses_final_tagged_value_when_response_quotes_candidate(self):
        response = "<reasoning>quoted answer</reasoning><score>1</score>"
        response += "<reasoning>judge rationale</reasoning><score>0</score>"

        score_match = find_last_tagged_match(response, "score")

        self.assertIsNotNone(score_match)
        self.assertEqual(score_match.group(1), "0")

    def test_tag_matching_remains_case_and_slash_tolerant(self):
        response = "<score>1</score>\n<Score>0<Score>"

        score_match = find_last_tagged_match(response, "score")

        self.assertIsNotNone(score_match)
        self.assertEqual(score_match.group(1), "0")

    def test_generic_patterns_also_use_the_final_match(self):
        response = "quoted Rating: 5\nfinal Rating: 1"

        rating_match = find_last_match(r"Rating:\s*([1-5])", response)

        self.assertIsNotNone(rating_match)
        self.assertEqual(rating_match.group(1), "1")


if __name__ == "__main__":
    unittest.main()
