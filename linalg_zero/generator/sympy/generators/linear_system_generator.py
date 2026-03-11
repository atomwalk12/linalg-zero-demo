from typing import Any

import sympy
from sympy.matrices import Matrix
from typing_extensions import override

from linalg_zero.generator.difficulty_config import (
    Precision,
    validate_tool_calls,
)
from linalg_zero.generator.entropy_control import SampleArgs
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
from linalg_zero.shared.lib_extra import solve_linear_system


class LinearSystemGenerator(MatrixVectorBaseGenerator):
    """
    Generator for linear system solving problems (independent variant).

    Creates "Solve Ax = b for x" problems using backwards construction:
    generate matrix A and solution vector x first, then compute b = Ax.
    """

    def __init__(
        self,
        difficulty_level: DifficultyCategory,
        **kwargs: Any,
    ) -> None:
        """Initialize independent linear system solver generator.

        Args:
            difficulty_level: The difficulty category for the problem
            **kwargs: Additional keyword arguments
        """
        super().__init__(difficulty_level=difficulty_level, **kwargs)
        assert self.problem_type == Task.ONE_LINEAR_SYSTEM_SOLVER

        # Validate that this problem type uses exactly 1 tool call
        validate_tool_calls(expected=self.config.target_tool_calls, actual=1, problem_type=self.problem_type)

    @property
    def precision(self) -> Precision:
        """The precision of the problem."""
        return Precision.LINEAR_SYSTEM_SOLVER

    @override
    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """
        Generate linear system solving problem content.

        Independent variant: generate A and x, compute b = Ax, then ask to solve for x.
        """
        # Set constraint for matrix invertibility
        context.constraints["matrix_invertible"] = True

        matrix_entropy, vector_entropy = self._split_entropy(context)

        size = self._determine_size(context)

        matrix_A = self._generate_matrix_A(size, matrix_entropy, context)
        vector_b = self._generate_vector_b(matrix_A, size, vector_entropy, context)

        sympy_sol, lib_result = self._solve_linear_system_sympy(matrix_A, vector_b)

        # Record tool call with input data
        input_data = self._prepare_tool_call_input_data(matrix_a=matrix_A, vector_b=vector_b)
        context.record_tool_call(solve_linear_system.__name__, lib_result, input_data, is_final=True)

        # Create symbolic variables for rendering the equation
        x_symbols = sympy.Matrix([sympy.Symbol(f"x_{i + 1}") for i in range(size)])

        # Problem: "Solve Ax = b for x"
        problem_expression = sympy.Eq(matrix_A * x_symbols, vector_b)

        context_info = self._prepare_context_info(matrix_A, vector_b, size)

        return ProblemTemplate(
            expression=problem_expression,
            variables={"matrix_A": matrix_A, "target_b": vector_b},
            sympy_solution=sympy_sol,
            lib_result=lib_result,
            context_info=context_info,
            difficulty_markers=self.build_difficulty_markers(context),
            difficulty=self.difficulty_level,
        )

    @override
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        """Return the variables dictionary to pass to the template engine."""
        input_variables = {"matrix_A": (template.context_info["matrix_A"], self.local_index)}
        input_variables["target_b"] = (template.context_info["target_b"], self.local_index)
        self.sources.update({"input_matrix_A": "local"})
        self.sources.update({"input_target_b": "local"})
        return self.get_dependent_template_variables(input_variables, self.sources)

    def _solve_linear_system_sympy(
        self, matrix_a: sympy.Matrix, vector_b: sympy.Matrix
    ) -> tuple[sympy.Matrix, list[list[float | int]]]:
        """Solve linear system Ax = b using lib.py function."""
        # Convert to primitives (this applies precision constraints)
        matrix_a_sympy = MathFormatter.sympy_to_primitive(matrix_a, precision=self.precision)
        vector_b_sympy = MathFormatter.sympy_to_primitive(vector_b, precision=self.precision)
        assert isinstance(matrix_a_sympy, list) and isinstance(vector_b_sympy, list)

        # Calculate using lib.py
        lib_result = self.lib["solve_linear_system"](matrix_a_sympy, vector_b_sympy)

        # This ensures there is no precision loss during verification
        matrix_a_precision_matched = Matrix(matrix_a_sympy)
        vector_b_precision_matched = Matrix(vector_b_sympy)
        sympy_result = matrix_a_precision_matched.LUsolve(vector_b_precision_matched)

        return sympy_result, lib_result

    def _split_entropy(self, context: ProblemContext) -> tuple[float, float]:
        """Split entropy between matrix and vector generation."""
        sample_args = SampleArgs(num_modules=2, entropy=context.entropy)
        split_fraction = (
            self.gen_constraints.split_fraction if self.gen_constraints.split_fraction is not None else 0.2
        )
        matrix_sample_args, vector_sample_args = sample_args.split(
            count=2, min_fraction=split_fraction, concentration_scale=3.0
        )
        return matrix_sample_args.entropy, vector_sample_args.entropy

    def _determine_size(self, context: ProblemContext) -> int:
        """Determine problem dimension (independent: random from config)."""
        return self.config.get_random_matrix_size()

    def _generate_matrix_A(
        self,
        size: int,
        matrix_entropy: float,
        context: ProblemContext,
    ) -> sympy.Matrix:
        # Use constraint-based generation for square invertible matrix
        # Temporarily set constraints for this specific call
        additional = GenerationConstraints(square=True, invertible=True, size=size)

        matrix_A = self._get_matrix_with_constraints(context, added_constraints=additional, entropy=matrix_entropy)
        return matrix_A

    def _generate_vector_b(
        self,
        matrix_A: sympy.Matrix,
        size: int,
        vector_entropy: float,
        context: ProblemContext,
    ) -> sympy.Matrix:
        # Generate solution vector using centralized entropy allocation, with fixed amount
        solution_x = self._get_vector_with_constraints(context, size=size, entropy=vector_entropy)
        return matrix_A * solution_x

    def _prepare_context_info(
        self,
        matrix_A: sympy.Matrix,
        vector_b: sympy.Matrix,
        size: int,
    ) -> dict[str, Any]:
        return {
            "matrix_dimensions": (size, size),
            "problem_type": self.problem_type,
            "matrix_A": matrix_A,
            "target_b": vector_b,
        }


class LinearSystemGeneratorDependent(LinearSystemGenerator):
    """Dependent variant: uses provided b vector from previous component."""

    def __init__(
        self,
        difficulty_level: DifficultyCategory,
        input_vector_b: sympy.Matrix,
        input_vector_b_index: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            difficulty_level=difficulty_level,
            **kwargs,
        )

        assert self.problem_type == Task.ONE_LINEAR_SYSTEM_SOLVER

        # Keep instance variables for other methods that need them
        self.input_vector_b = input_vector_b
        self.input_vector_b_index = input_vector_b_index

    def _split_entropy(self, context: ProblemContext) -> tuple[float, float]:
        sample_args = SampleArgs(num_modules=1, entropy=context.entropy)
        return sample_args.entropy, 0.0

    def _determine_size(self, context: ProblemContext) -> int:
        return int(self.input_vector_b.shape[0])

    def _generate_vector_b(
        self,
        matrix_A: sympy.Matrix,
        size: int,
        vector_entropy: float,
        context: ProblemContext,
    ) -> sympy.Matrix:
        # No entropy usage for provided vector b
        return self.input_vector_b

    def _prepare_context_info(
        self,
        matrix_A: sympy.Matrix,
        vector_b: sympy.Matrix,
        size: int,
    ) -> dict[str, Any]:
        context_info = super()._prepare_context_info(matrix_A, vector_b, size)
        context_info["input_variable_name"] = "b"
        context_info["input_indices"] = self.input_vector_b_index
        return context_info

    def _prepare_tool_call_input_data(self, **kwargs: dict[str, Any]) -> dict[str, Any]:
        """Prepare input data for dependent generator including dependency info."""
        base_data = super()._prepare_tool_call_input_data(**kwargs)
        assert self.input_vector_b == kwargs["vector_b"]
        base_data.update({
            "dependent_on": {"input_vector_b": self.input_vector_b_index},
            "input_vector_b": MathFormatter.sympy_to_primitive(self.input_vector_b, precision=self.precision),
        })
        return base_data

    @override
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        """Use the mixin's generic logic for consistent result/value handling."""
        input_variables = {}
        input_variables["target_b"] = (self.input_vector_b, self.input_vector_b_index)
        input_variables["matrix_A"] = (template.context_info["matrix_A"], self.local_index)

        base_vars = self.get_dependent_template_variables(input_variables, self.sources)

        return base_vars
