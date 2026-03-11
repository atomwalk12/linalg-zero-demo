from linalg_zero.grpo.verifiers.xml_parser import XMLParser
from linalg_zero.grpo.verify import parse_string, verify_answers
from linalg_zero.shared.types import LibTypes

parser = XMLParser()


def validate_answer(ground_truth: LibTypes, completion: str) -> bool:
    """Reward function that checks if the completion answer matches the ground truth."""
    answer = parser.extract_last_answer(completion)
    target = parse_string(answer) if answer else None
    if target is None:
        return False
    return verify_answers(ground_truth, target)


def think_correct(completion: str) -> bool:
    has_think = parser.extract_last_thought(completion)
    return has_think is not None


def answer_correct(completion: str) -> bool:
    has_answer = parser.extract_last_answer(completion)
    return has_answer is not None
