from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from linalg_zero.generator.models import DifficultyCategory, Task
from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)

# Deterministic controls
#
# Why both set_seed and DETERMINISTIC_BASE_SEED exist:
# - set_seed(seed) initializes Python/NumPy/SymPy RNGs. It makes a single
#   process deterministic only when the sequence of RNG calls is identical.
#   Across phases (e.g., analysis vs generation), RNG call order/count differs
#   (entropy sampling timing, retries, filters), so outputs can diverge despite
#   using the same seed.
# - When DETERMINISTIC_MODE is True, factories reseed per question using a
#   stable function of (DETERMINISTIC_BASE_SEED, problem_type, topic, index).
#   This pins each question's randomness to its identity, making results
#   invariant to incidental RNG call ordering differences between phases.
# - If a CLI --seed is provided and DETERMINISTIC_MODE is True, we set
#   DETERMINISTIC_BASE_SEED = --seed so users can reproduce/scan deterministic
#   sequences without code changes. When DETERMINISTIC_MODE is False, the base
#   seed is ignored and set_seed controls reproducibility as usual.
#
DETERMINISTIC_MODE: bool = True
DETERMINISTIC_BASE_SEED: int = 146959810


class Precision(Enum):
    """Precision for formatting mathematical expressions."""

    MATRIX_VECTOR_MULTIPLICATION = 2
    MATRIX_MATRIX_MULTIPLICATION = 2
    LINEAR_SYSTEM_SOLVER = 2
    DETERMINANT = 2
    FROBENIUS_NORM = 2
    MATRIX_RANK = 2
    MATRIX_TRANSPOSE = 2
    MATRIX_INVERSE = 2
    MATRIX_TRACE = 2
    MATRIX_COFACTOR = 2
    FULL = -1


class ToolCallDifficulty(Enum):
    """Tool call based difficulty levels."""

    SINGLE_TOOL = 1
    DUAL_TOOL = 2
    MULTI_TOOL = 3


@dataclass(frozen=True)
class ProblemConfig:
    """Configuration parameters for problems based on tool calls and difficulty."""

    target_tool_calls: int
    matrix_size_range: tuple[int, int]
    allow_rationals: bool

    def get_random_matrix_size(self) -> int:
        """Get a random matrix size within the allowed range."""
        return random.randint(*self.matrix_size_range)


# Possible entropy ranges:
# Moderate variability:
#   - 1 tool call: (1.2, 1.8)
#   - 2 tool calls: (2.6, 3.6)
#   - 3 tool calls: (3.8, 5.2)

# High variability:
#   - 1 tool call: (1.0, 2.0)
#   - 2 tool calls: (2.0, 4.0)
#   - 3 tool calls: (3.0, 6.0)

# Low variability:
#   - 1 tool call: (1.4, 1.6)
#   - 2 tool calls: (2.8, 3.2)
#   - 3 tool calls: (4.2, 4.8)

EASY_PROBLEM_CONFIG = ProblemConfig(target_tool_calls=1, matrix_size_range=(2, 3), allow_rationals=False)
MEDIUM_PROBLEM_CONFIG = ProblemConfig(target_tool_calls=1, matrix_size_range=(2, 3), allow_rationals=False)
HARD_PROBLEM_CONFIG = ProblemConfig(target_tool_calls=1, matrix_size_range=(2, 2), allow_rationals=False)


def determine_difficulty(problem_type: Task) -> DifficultyCategory:
    """Determine difficulty category based on problem type name."""
    if problem_type.name.startswith("THREE_"):
        return DifficultyCategory.THREE_TOOL_CALLS
    elif problem_type.name.startswith("TWO_"):
        return DifficultyCategory.TWO_TOOL_CALLS
    elif problem_type.name.startswith("ONE_"):
        return DifficultyCategory.ONE_TOOL_CALL
    else:
        raise ValueError(f"Invalid problem type: {problem_type}")


def get_problem_config(difficulty: DifficultyCategory) -> ProblemConfig:
    """Get problem configuration for a given difficulty level, topic, and problem type."""
    if difficulty == DifficultyCategory.ONE_TOOL_CALL:
        return EASY_PROBLEM_CONFIG
    elif difficulty == DifficultyCategory.TWO_TOOL_CALLS:
        return MEDIUM_PROBLEM_CONFIG
    elif difficulty == DifficultyCategory.THREE_TOOL_CALLS:
        return HARD_PROBLEM_CONFIG
    else:
        raise ValueError(f"Invalid difficulty category: {difficulty}")


def validate_tool_calls(expected: int, actual: int, problem_type: Task) -> bool:
    """Validate that a problem uses the expected number of tool calls."""
    if actual != expected:
        raise ValueError(
            f"Problem type '{problem_type}' expected {expected} tool calls, "
            f"but used {actual}. This violates the difficulty system."
        )
    return True
