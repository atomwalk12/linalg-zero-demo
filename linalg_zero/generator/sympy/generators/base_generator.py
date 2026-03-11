import random
from typing import Any

from sympy import Integer, Matrix
from sympy.core import Rational

from linalg_zero.generator.entropy_control import EntropyController
from linalg_zero.generator.generation_constraints import GenerationConstraints
from linalg_zero.generator.models import DifficultyCategory
from linalg_zero.generator.sympy.base import ProblemContext, SympyProblemGenerator


class MatrixVectorBaseGenerator(SympyProblemGenerator):
    """Base class for matrix-vector problem generators."""

    def __init__(
        self, difficulty_level: DifficultyCategory, gen_constraints: GenerationConstraints | None, **kwargs: Any
    ) -> None:
        super().__init__(difficulty_level=difficulty_level, **kwargs)

        self.gen_constraints = (
            gen_constraints if isinstance(gen_constraints, GenerationConstraints) else GenerationConstraints()
        )

        self.entropy_controller = EntropyController()

    def _generate_matrix(self, rows: int, cols: int, entropy: float) -> Matrix:
        """Generate a matrix consisting of integers or rationals."""
        matrix_elements = []
        min_abs = self.gen_constraints.min_element_abs
        for _ in range(rows):
            row = []
            for _ in range(cols):
                if self.config.allow_rationals and random.random() < 0.3:
                    element = self.entropy_controller.generate_rational(
                        entropy, min_value_abs=min_abs, max_attempts=self.max_attempts
                    )
                    element = Rational(element)
                else:
                    number = self.entropy_controller.generate_integer(
                        entropy, min_abs=min_abs, max_attempts=self.max_attempts
                    )

                    element = Integer(number)

                if abs(element) < min_abs:
                    raise ValueError("Matrix contains elements less than the minimum absolute value")

                row.append(element)
            matrix_elements.append(row)

        return Matrix(matrix_elements)

    def _generate_invertible_matrix(self, size: int, entropy: float) -> Matrix:
        """Generate an invertible matrix with retry logic."""
        for _ in range(self.max_attempts):
            matrix = self._generate_matrix(size, size, entropy)

            try:
                det = matrix.det()
                if det != 0:
                    return matrix
            except Exception:  # noqa: S112
                continue

        raise ValueError(f"Failed to generate invertible matrix after {self.max_attempts} attempts")

    def _get_matrix_with_constraints(
        self,
        context: ProblemContext,
        added_constraints: GenerationConstraints | None = None,
        entropy: float | None = None,
    ) -> Matrix:
        """Generate matrix based on constructor constraints."""
        gen_constraints = self.gen_constraints.merge(added_constraints) if added_constraints else self.gen_constraints

        # Determine dimensions
        if gen_constraints.rows is not None and gen_constraints.cols is not None:
            rows, cols = gen_constraints.rows, gen_constraints.cols
        elif gen_constraints.square:
            size = gen_constraints.size if gen_constraints.size is not None else self.config.get_random_matrix_size()
            rows = cols = size
        else:
            rows = self.config.get_random_matrix_size()
            cols = self.config.get_random_matrix_size()

        # Determine entropy allocation via context allocator (centralized budget management)
        matrix_entropy = context.allocate_entropy(entropy=entropy)

        # Generate matrix based on special properties
        if gen_constraints.invertible:
            if rows != cols:
                raise ValueError("Invertible matrices must be square")
            matrix_A = self._generate_invertible_matrix(rows, matrix_entropy)
        else:
            matrix_A = self._generate_matrix(rows, cols, matrix_entropy)

        return matrix_A

    def _get_vector_with_constraints(self, context: ProblemContext, size: int, entropy: float | None = None) -> Matrix:
        """Generate a vector using centralized entropy allocation.

        Dimensions are provided explicitly via `size`. Constraints are used only
        to resolve entropy (fixed or sampled); shape-related fields are ignored
        for vectors.
        """
        vector_entropy = context.allocate_entropy(entropy=entropy)

        # Generate as column vector (size x 1 matrix)
        vector = self._generate_matrix(size, 1, vector_entropy)
        return vector
