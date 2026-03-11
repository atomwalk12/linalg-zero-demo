from typing import Any

from sympy import Float, Integer, Matrix, Rational
from typing_extensions import override

from linalg_zero.generator.difficulty_config import (
    Precision,
    validate_tool_calls,
)
from linalg_zero.generator.generation_constraints import GenerationConstraints
from linalg_zero.generator.models import DifficultyCategory, Task
from linalg_zero.generator.sympy.base import (
    ProblemContext,
    ProblemTemplate,
)
from linalg_zero.generator.sympy.generators.base_generator import (
    MatrixVectorBaseGenerator,
)
from linalg_zero.generator.sympy.template_engine import MathFormatter
from linalg_zero.shared.lib import determinant


class DeterminantGenerator(MatrixVectorBaseGenerator):
    """
    This generator creates problems asking to compute the determinant of a square matrix.
    """

    def __init__(self, difficulty_level: DifficultyCategory, **kwargs: Any) -> None:
        """Initialize determinant generator."""
        super().__init__(difficulty_level=difficulty_level, **kwargs)
        assert self.problem_type == Task.ONE_DETERMINANT

        validate_tool_calls(expected=self.config.target_tool_calls, actual=1, problem_type=self.problem_type)

    @property
    def precision(self) -> Precision:
        """The precision of the problem."""
        return Precision.DETERMINANT

    def _get_matrix(self, context: ProblemContext) -> Matrix:
        """Generate or retrieve the matrix for determinant calculation."""
        # Get matrix size and entropy
        mandatory = GenerationConstraints(square=True)
        matrix_A = self._get_matrix_with_constraints(context, added_constraints=mandatory)
        return matrix_A

    @override
    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """
        Generate determinant calculation problem content.

        This method creates problems asking to compute det(A) for a square matrix A.
        Uses difficulty configuration to determine matrix size and complexity.
        """
        # Get matrix using overrideable method
        matrix_A = self._get_matrix(context)

        sympy_det, lib_result = self._calculate_determinant_sympy(matrix_A)

        # Record tool call with input data
        input_data = self._prepare_tool_call_input_data(matrix=matrix_A)
        context.record_tool_call(determinant.__name__, lib_result, input_data, is_final=True)

        # Generate question templates
        problem_expression = matrix_A

        return ProblemTemplate(
            expression=problem_expression,
            variables={"matrix": matrix_A},
            sympy_solution=sympy_det,
            lib_result=lib_result,
            context_info={
                "matrix": matrix_A,
            },
            difficulty_markers=self.build_difficulty_markers(context),
            difficulty=self.difficulty_level,
        )

    @override
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        """Return the variables dictionary to pass to the template engine."""
        input_variables = {"matrix": (template.context_info["matrix"], self.local_index)}
        self.sources.update({"input_matrix": "local"})
        return self.get_dependent_template_variables(input_variables, self.sources)

    def _calculate_determinant_sympy(self, matrix_a: Matrix) -> tuple[Float | Integer | Rational, float]:
        """Calculate determinant using both SymPy and lib.py function."""
        # Convert to primitives (this applies precision constraints)
        matrix_a_primitive = MathFormatter.sympy_to_primitive(matrix_a, precision=self.precision)
        assert isinstance(matrix_a_primitive, list)

        # Calculate using lib.py with the primitives
        lib_result = self.lib["determinant"](matrix_a_primitive)

        # Convert primitives back to SymPy Matrix at the same precision level
        # This ensures both calculations work with the same precision
        matrix_a_precision_matched = Matrix(matrix_a_primitive)
        sympy_result = matrix_a_precision_matched.det()
        assert isinstance(sympy_result, Float | Integer | Rational)

        return sympy_result, lib_result


class DeterminantGeneratorDependent(DeterminantGenerator):
    """Dependent variant: uses provided input matrix and reports dependency index in difficulty markers."""

    def __init__(
        self,
        difficulty_level: DifficultyCategory,
        input_matrix: Matrix,
        input_matrix_index: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(difficulty_level=difficulty_level, **kwargs)

        assert self.problem_type == Task.ONE_DETERMINANT

        self.input_matrix = input_matrix
        self.input_matrix_index = input_matrix_index

    def _get_matrix(self, context: ProblemContext) -> Matrix:
        """Return the provided input matrix without consuming entropy."""
        return self.input_matrix

    def _prepare_tool_call_input_data(self, **kwargs: dict[str, Any]) -> dict[str, Any]:
        """Prepare input data for dependent generator including dependency info."""
        base_data = super()._prepare_tool_call_input_data(**kwargs)
        assert self.input_matrix == kwargs["matrix"]
        base_data.update({
            "dependent_on": {"input_matrix": self.input_matrix_index},
            "input_matrix": MathFormatter.sympy_to_primitive(self.input_matrix, precision=self.precision),
        })
        return base_data

    @override
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        """Use the mixin's generic logic for consistent result/value handling."""
        input_variables = {"matrix": (self.input_matrix, self.input_matrix_index)}
        return self.get_dependent_template_variables(input_variables, self.sources)
