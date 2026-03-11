import ast

from math_verify import verify
from sympy import Float, Integer, Matrix

from linalg_zero.shared.types import LibTypes


def parse_string(s: str) -> LibTypes | None:
    """
    Parse string to the most appropriate library type, or None if unsuccessful.
    """
    s = s.strip()

    if not s:
        return None

    try:
        parsed = ast.literal_eval(s)

        if isinstance(parsed, int | float | list | tuple):
            return list(parsed) if isinstance(parsed, tuple) else parsed

    except (ValueError, SyntaxError):
        pass

    return None


def verify_answers(ground_truth: LibTypes | None, target_answer: LibTypes | None, timeout: int = 5) -> bool:
    """Verify if the target answer matches the ground truth using math_verify."""

    def convert_to_sympy(answer: LibTypes) -> Matrix | Float | Integer:
        """Convert the answer to a SymPy object."""
        if isinstance(answer, list):
            # NOTE[atom]: this throws a deprecation warning
            return Matrix(answer)
        elif isinstance(answer, float):
            return Float(answer)
        elif isinstance(answer, int):
            return Integer(answer)
        else:
            raise TypeError(f"Unsupported answer type: {type(answer)}")

    if ground_truth is None or target_answer is None:
        return False

    try:
        target = convert_to_sympy(target_answer)
        gt = convert_to_sympy(ground_truth)

        return verify(gt, target, timeout_seconds=timeout)
    except Exception:
        return False
