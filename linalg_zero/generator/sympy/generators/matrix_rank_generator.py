from typing import Any

from sympy import Integer, Matrix
from typing_extensions import override

from linalg_zero.generator.difficulty_config import (
    Precision,
    validate_tool_calls,
)
from linalg_zero.generator.models import DifficultyCategory, Task
from linalg_zero.generator.sympy.base import (
    ProblemContext,
    ProblemTemplate,
)
from linalg_zero.generator.sympy.generators.base_generator import (
    MatrixVectorBaseGenerator,
)
from linalg_zero.generator.sympy.template_engine import MathFormatter
from linalg_zero.shared.lib import matrix_rank


class MatrixRankGenerator(MatrixVectorBaseGenerator):
    """
    This generator creates problems asking to compute the rank of a matrix.
    """

    def __init__(self, difficulty_level: DifficultyCategory, **kwargs: Any) -> None:
        """Initialize matrix rank generator."""
        super().__init__(difficulty_level=difficulty_level, **kwargs)
        assert self.problem_type == Task.ONE_RANK

        validate_tool_calls(expected=self.config.target_tool_calls, actual=1, problem_type=self.problem_type)

    @property
    def precision(self) -> Precision:
        """The precision of the problem."""
        return Precision.MATRIX_RANK

    def _get_matrix(self, context: ProblemContext) -> Matrix:
        """Generate or retrieve the matrix for rank calculation."""
        return self._get_matrix_with_constraints(context)

    @override
    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """
        Generate matrix rank calculation problem content.

        This method creates problems asking to compute rank(A) for a matrix A.
        Uses difficulty configuration to determine matrix size and complexity.
        """
        # Get matrix using overrideable method
        matrix_A = self._get_matrix(context)

        sympy_rank, lib_result = self._calculate_rank_sympy(matrix_A)

        # Record tool call with input data
        input_data = self._prepare_tool_call_input_data(matrix=matrix_A)
        context.record_tool_call(matrix_rank.__name__, lib_result, input_data, is_final=True)

        # Generate question templates
        problem_expression = matrix_A

        return ProblemTemplate(
            expression=problem_expression,
            variables={"matrix": matrix_A},
            sympy_solution=sympy_rank,
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

    def _calculate_rank_sympy(self, matrix_a: Matrix) -> tuple[Integer, int]:
        """Calculate matrix rank using both SymPy and lib.py function."""
        # Convert to primitives for lib.py calculation

        matrix_a_primitive = MathFormatter.sympy_to_primitive(matrix_a, precision=self.precision)
        assert isinstance(matrix_a_primitive, list)

        # Calculate using lib.py with the primitives
        lib_result = self.lib["matrix_rank"](matrix_a_primitive)

        # Convert primitives back to SymPy Matrix at the same precision level
        # This ensures both calculations work with the same precision
        matrix_a_precision_matched = Matrix(matrix_a_primitive)
        sympy_result = matrix_a_precision_matched.rank()
        assert isinstance(sympy_result, int)

        return Integer(sympy_result), lib_result


class MatrixRankGeneratorDependent(MatrixRankGenerator):
    """Dependent variant: uses provided input matrix and reports dependency index in difficulty markers."""

    def __init__(
        self,
        difficulty_level: DifficultyCategory,
        input_matrix: Matrix,
        input_matrix_index: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(difficulty_level=difficulty_level, **kwargs)

        assert self.problem_type == Task.ONE_RANK
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
        """Return template variables for dependent matrix rank generator."""
        input_variables = {"matrix": (self.input_matrix, self.input_matrix_index)}
        return self.get_dependent_template_variables(input_variables, self.sources)
