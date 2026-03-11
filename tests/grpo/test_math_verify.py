import pytest

from linalg_zero.grpo.verify import parse_string, verify_answers


class TestVerifyAnswersCorrectness:
    """
    Test the end-to-end correctness of verify_answers function.
    Focus on whether the permissive extract_math_content leads to wrong verification results.
    """

    def test_correct_exact_matches(self):
        """Test that exact matches are correctly verified as True."""
        test_cases = [
            ("42", "42"),
            ("-17.5", "-17.5"),
            ("[[1, 2], [3, 4]]", "[[1, 2], [3, 4]]"),
            ("[1, 2, 3]", "[1, 2, 3]"),
            ("0", "0"),
            ("1.5e-10", "1.5e-10"),
        ]

        for completion, ground_truth in test_cases:
            parsed_ground_truth = parse_string(ground_truth)
            parsed_completion = parse_string(completion)
            result = verify_answers(parsed_ground_truth, parsed_completion)
            assert result is True, f"Failed: {completion} should equal {ground_truth}"

    def test_incorrect_exact_mismatches(self):
        """Test that clear mismatches are correctly verified as False."""
        test_cases = [
            ("42", "43"),
            ("-17.5", "17.5"),
            ("[[1, 2], [3, 4]]", "[[4, 3], [2, 1]]"),
            ("[1, 2, 3]", "[3, 2, 1]"),
            ("5", "0"),
        ]

        for completion, ground_truth in test_cases:
            parsed_ground_truth = parse_string(ground_truth)
            parsed_completion = parse_string(completion)
            result = verify_answers(parsed_ground_truth, parsed_completion)
            assert result is False, f"Failed: {completion} should NOT equal {ground_truth}"

    def test_current_behavior_strict_matching(self):
        """Test that current implementation only does strict string matching."""
        test_cases = [
            ("42", "42", True),
            ("[[1, 2], [3, 4]]", "[[1, 2], [3, 4]]", True),
            ("-17.5", "-17.5", True),
            ("5", "5", True),
            ("<answer>42</answer>", "42", False),
            ("Some text <answer>[[1, 2], [3, 4]]</answer> more text", "[[1, 2], [3, 4]]", False),
            ("The determinant is <answer>-17.5</answer>", "-17.5", False),
            ("The answer is 42", "42", False),
        ]

        for completion, ground_truth, expected in test_cases:
            parsed_ground_truth = parse_string(ground_truth)
            parsed_completion = parse_string(completion)
            result = verify_answers(parsed_ground_truth, parsed_completion)
            assert result == expected, f"Failed: '{completion}' vs '{ground_truth}' expected {expected}, got {result}"

    def test_no_answer_tags_fallback(self):
        """Test that text without answer tags is processed as-is."""
        test_cases = [
            ("42", "42"),
            ("[[1, 2], [3, 4]]", "[[1, 2], [3, 4]]"),
            ("-17.5", "-17.5"),
            ("5", "5"),
            ("The answer is 42", "42"),
        ]

        for completion, ground_truth in test_cases[:-1]:
            parsed_ground_truth = parse_string(ground_truth)
            parsed_completion = parse_string(completion)
            result = verify_answers(parsed_ground_truth, parsed_completion)
            assert result is True, f"Failed: '{completion}' should match '{ground_truth}'"

        # Test that unparseable text returns False
        parsed_ground_truth = parse_string("42")
        parsed_completion = parse_string("The answer is 42")  # This should return None
        result = verify_answers(parsed_ground_truth, parsed_completion)
        assert result is False, "Text without answer tags should not flexibly extract numbers"

    def test_malformed_input_edge_cases(self):
        """
        Test that malformed input cases return False as expected.
        """
        malformed_cases = [
            ("[[1, 2], [3, 4", "[[1, 2], [3, 4]]"),
            ("42.5.6", "42.5"),
            ("Multiple answers: 42 and 24", "42"),
            ("[[1, 2]] and [[3, 4]]", "[[1, 2]]"),
            ("<answer>[[1, 2], [3, 4</answer>", "[[1, 2], [3, 4]]"),
            ("Text <answer>42</answer> and <answer>24</answer>", "42"),
        ]

        for completion, ground_truth in malformed_cases:
            parsed_ground_truth = parse_string(ground_truth)
            parsed_completion = parse_string(completion)
            result = verify_answers(parsed_ground_truth, parsed_completion)
            assert result is False, f"Malformed input should be False: '{completion}' vs '{ground_truth}'"

    def test_verify_mathematical_equivalence(self):
        """Test that mathematically equivalent but differently formatted answers are correctly identified."""
        equivalence_cases = [
            ("2.0", "2"),
            ("[[1.0, 2.0], [3.0, 4.0]]", "[[1, 2], [3, 4]]"),
            ("0.0", "0"),
            ("-0", "0"),
        ]

        for completion, ground_truth in equivalence_cases:
            parsed_ground_truth = parse_string(ground_truth)
            parsed_completion = parse_string(completion)
            result = verify_answers(parsed_ground_truth, parsed_completion)
            assert result is True, f"Mathematical equivalence failed: '{completion}' should equal '{ground_truth}'"

    def test_empty_and_invalid_inputs(self):
        """Test that empty or completely invalid inputs return False."""
        edge_cases = [
            ("", "42"),
            ("42", ""),
            ("", ""),
            ("No numbers here", "42"),
            ("The result is undefined", "42"),
        ]

        for completion, ground_truth in edge_cases:
            parsed_ground_truth = parse_string(ground_truth)
            parsed_completion = parse_string(completion)
            result = verify_answers(parsed_ground_truth, parsed_completion)
            assert result is False, f"Edge case should be False: '{completion}' vs '{ground_truth}'"

    @pytest.mark.parametrize(
        "completion,ground_truth,should_match",
        [
            # Clear correct cases
            ("42", "42", True),
            ("[[1, 2], [3, 4]]", "[[1, 2], [3, 4]]", True),
            # Clear incorrect cases
            ("42", "43", False),
            ("[[1, 2], [3, 4]]", "[[4, 3], [2, 1]]", False),
            # Cases that fail
            ("<answer>42</answer>", "42", False),
            ("The answer is 42", "42", False),
            ("The answer is 42", "43", False),
            # Malformed input
            ("[[1, 2], [3, 4", "[[1, 2], [3, 4]]", False),
            ("42.5.6", "42.5", False),
        ],
    )
    def test_verify_answers_comprehensive(self, completion, ground_truth, should_match):
        """Comprehensive test of verify_answers behavior."""
        parsed_ground_truth = parse_string(ground_truth)
        parsed_completion = parse_string(completion)
        result = verify_answers(parsed_ground_truth, parsed_completion)

        if should_match is not None:
            assert result == should_match, (
                f"Expected {should_match} for '{completion}' vs '{ground_truth}', got {result}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
