import json

import pytest

from linalg_zero.generator.difficulty_config import get_problem_config
from linalg_zero.generator.entropy_control import EntropyConstraints
from linalg_zero.generator.models import DifficultyCategory, Question, Task, Topic
from linalg_zero.generator.sympy.generators.determinant_generator import (
    DeterminantGenerator,
)
from linalg_zero.shared.lib import determinant


class TestDeterminantGenerator:
    """Focused end-to-end tests for DeterminantGenerator."""

    config = get_problem_config(DifficultyCategory.TWO_TOOL_CALLS)

    def _make_generator(self, difficulty: DifficultyCategory) -> DeterminantGenerator:
        from linalg_zero.generator.generation_constraints import GenerationConstraints
        from linalg_zero.generator.sympy.template_engine import TemplateEngine

        entropy_constraints = EntropyConstraints(entropy=1.0)

        return DeterminantGenerator(
            gen_constraints=GenerationConstraints(),
            template_engine=TemplateEngine(),
            local_index=0,
            constraints={},
            entropy=entropy_constraints.sample_entropy(),
            difficulty_level=difficulty,
            problem_type=Task.ONE_DETERMINANT,
            topic=Topic.LINEAR_ALGEBRA,
        )

    def test_basic_generation_easy(self):
        generator = self._make_generator(DifficultyCategory.ONE_TOOL_CALL)
        q = generator.generate()

        assert isinstance(q, Question)
        assert q.is_valid
        assert q.topic == Topic.LINEAR_ALGEBRA
        assert q.difficulty == DifficultyCategory.ONE_TOOL_CALL
        assert q.tool_calls_required == 1
        assert len(q.question) > 0
        assert len(q.answer) > 0

        # Answer should be numeric (stringified JSON number)
        value = float(q.answer)
        assert isinstance(value, float)

    def test_medium_and_hard_generation(self):
        for difficulty in (DifficultyCategory.TWO_TOOL_CALLS, DifficultyCategory.THREE_TOOL_CALLS):
            generator = self._make_generator(difficulty)
            q = generator.generate()

            assert q.is_valid
            assert q.difficulty == difficulty
            assert q.tool_calls_required == 1

            value = float(q.answer)
            assert isinstance(value, float)

    def test_question_contains_determinant_language(self):
        generator = self._make_generator(DifficultyCategory.TWO_TOOL_CALLS)
        q = generator.generate()

        text = q.question.lower()
        assert any(kw in text for kw in ["determinant", "det(", "det ", "calculate", "compute", "find"])

    def test_multiple_generations_stability(self):
        generator = self._make_generator(DifficultyCategory.TWO_TOOL_CALLS)

        for _ in range(10):
            q = generator.generate()
            assert q.is_valid
            float(json.loads(q.answer))

    def test_determinant_tool_function_examples(self):
        # 2x2 matrix should have determinant -2
        result = determinant([[1, 2], [3, 4]])
        assert result == -2
        assert isinstance(result, float)

        # Diagonal matrix determinant is product of diagonal entries
        result = determinant([[2, 0], [0, 3]])
        assert result == 6

        # Identity determinant is 1
        result = determinant([[1, 0], [0, 1]])
        assert result == 1

        # 1x1 matrix
        result = determinant([[5]])
        assert result == 5

    def test_determinant_tool_function_error_handling(self):
        # Non-square matrix should raise ValueError
        with pytest.raises(ValueError, match="Matrix must be square"):
            determinant([[1, 2, 3], [4, 5, 6]])
