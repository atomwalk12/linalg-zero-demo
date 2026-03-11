from typing import Any

import sympy
from sympy import Matrix
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
from linalg_zero.generator.sympy.generators.base_generator import MatrixVectorBaseGenerator
from linalg_zero.generator.sympy.template_engine import MathFormatter
from linalg_zero.shared.lib_extra import multiply_matrices


class MatrixMatrixMultiplicationGenerator(MatrixVectorBaseGenerator):
    def __init__(
        self,
        difficulty_level: DifficultyCategory,
        **kwargs: Any,
    ) -> None:
        """Initialize independent matrix-matrix multiplication generator."""
        super().__init__(difficulty_level=difficulty_level, **kwargs)
        assert self.problem_type == Task.ONE_MATRIX_MATRIX_MULTIPLICATION

        # Validate that this problem type uses exactly 1 tool call
        validate_tool_calls(expected=self.config.target_tool_calls, actual=1, problem_type=self.problem_type)

    @property
    def precision(self) -> Precision:
        return Precision.MATRIX_MATRIX_MULTIPLICATION

    @override
    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """Generate matrix-matrix multiplication problem content (independent)."""

        matrix_A_entropy, matrix_B_entropy = self._split_entropy(context)
        rows, inner_dim, cols = self._determine_dimensions(context)

        for _ in range(self.max_attempts):
            matrix_A = self._generate_matrix_A(rows, inner_dim, matrix_A_entropy, context)
            matrix_B = self._generate_matrix_B(inner_dim, cols, matrix_B_entropy, context)
            sympy_sol, lib_result = self._multiply_matrices_sympy(matrix_a=matrix_A, matrix_b=matrix_B)

            # Accept if the product is not an all-zero matrix; allow zeros otherwise
            if not all(value == 0 for value in sympy_sol):
                break

            # Refund the entropy consumed by this attempt to avoid exhausting the budget
            used_entropy = matrix_A_entropy + matrix_B_entropy
            context.used_entropy -= used_entropy
            if context.used_entropy < 0:
                context.used_entropy = 0.0
        else:
            # If every attempt produced an all-zero product, raise an error to avoid
            # silently emitting degenerate data.
            raise ValueError(f"Failed to generate non-degenerate product after {self.max_attempts} attempts")

        # Record tool call with input data
        input_data = self._prepare_tool_call_input_data(matrix_a=matrix_A, matrix_b=matrix_B)
        context.record_tool_call(multiply_matrices.__name__, lib_result, input_data, is_final=True)

        problem_expression = matrix_A * matrix_B

        context_info = {
            "matrix_dimensions": (rows, inner_dim, cols),
            "problem_type": self.problem_type,
            "matrix_A": matrix_A,
            "matrix_B": matrix_B,
        }

        return ProblemTemplate(
            expression=problem_expression,
            variables={"matrix_A": matrix_A, "matrix_B": matrix_B},
            sympy_solution=sympy_sol,
            lib_result=lib_result,
            context_info={**context_info},
            difficulty_markers=self.build_difficulty_markers(
                context, matrix_size=(matrix_A.rows, matrix_A.cols), matrix_size_B=(matrix_B.rows, matrix_B.cols)
            ),
            difficulty=self.difficulty_level,
        )

    @override
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        """Return the variables dictionary to pass to the template engine."""
        input_variables = {"matrix_A": (template.context_info["matrix_A"], self.local_index)}
        input_variables["matrix_B"] = (template.context_info["matrix_B"], self.local_index)
        self.sources.update({"input_matrix_A": "local"})
        self.sources.update({"input_matrix_B": "local"})
        return self.get_dependent_template_variables(input_variables, self.sources)

    def _multiply_matrices_sympy(self, matrix_a: Matrix, matrix_b: Matrix) -> tuple[Matrix, list[list[float]]]:
        """Multiply two sympy matrices using lib.py function."""
        # Convert to primitives (this applies precision constraints)
        a_list = self.formatter.sympy_to_primitive(matrix_a, precision=self.precision)
        b_list = self.formatter.sympy_to_primitive(matrix_b, precision=self.precision)
        assert isinstance(a_list, list) and isinstance(b_list, list)

        # Calculate using lib.py with the primitives
        lib_result = self.lib["multiply_matrices"](a_list, b_list)

        # Convert primitives back to SymPy matrices at the same precision level
        # This ensures both calculations work with the same precision
        matrix_a_precision_matched = Matrix(a_list)
        matrix_b_precision_matched = Matrix(b_list)
        sympy_result = matrix_a_precision_matched * matrix_b_precision_matched

        return sympy_result, lib_result

    def _split_entropy(self, context: ProblemContext) -> tuple[float, float]:
        """Split entropy between matrix A and matrix B generation (independent)."""
        sample_args = SampleArgs(num_modules=2, entropy=context.entropy)
        split_fraction = (
            self.gen_constraints.split_fraction if self.gen_constraints.split_fraction is not None else 0.3
        )
        matrix_A_sample_args, matrix_B_sample_args = sample_args.split(
            count=2, min_fraction=split_fraction, concentration_scale=10.0
        )
        return matrix_A_sample_args.entropy, matrix_B_sample_args.entropy

    def _determine_dimensions(self, context: ProblemContext) -> tuple[int, int, int]:
        """Select matrix dimensions (independent): A(rows x inner_dim), B(inner_dim x cols)."""
        rows = self.config.get_random_matrix_size()
        inner_dim = self.config.get_random_matrix_size()
        cols = self.config.get_random_matrix_size()
        return rows, inner_dim, cols

    def _generate_matrix_A(
        self,
        rows: int,
        cols: int,
        matrix_entropy: float,
        context: ProblemContext,
    ) -> Matrix:
        # Use constraint-based generation with specific dimensions
        # Temporarily set constraints for this specific call
        mandatory = GenerationConstraints(rows=rows, cols=cols, min_element_abs=1)

        matrix_A = self._get_matrix_with_constraints(context, added_constraints=mandatory, entropy=matrix_entropy)

        return matrix_A

    def _generate_matrix_B(
        self,
        rows: int,
        cols: int,
        matrix_entropy: float,
        context: ProblemContext,
    ) -> Matrix:
        mandatory = GenerationConstraints(rows=rows, cols=cols, min_element_abs=1)

        matrix_B = self._get_matrix_with_constraints(context, added_constraints=mandatory, entropy=matrix_entropy)

        return matrix_B


class MatrixMatrixMultiplicationGeneratorDependent(MatrixMatrixMultiplicationGenerator):
    def __init__(
        self,
        difficulty_level: DifficultyCategory,
        input_matrix_A: sympy.Matrix,
        input_matrix_A_index: int,
        input_matrix_B: sympy.Matrix | None = None,
        input_matrix_B_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            difficulty_level=difficulty_level,
            **kwargs,
        )

        assert self.problem_type == Task.ONE_MATRIX_MATRIX_MULTIPLICATION
        self.input_matrix_A = input_matrix_A
        self.input_matrix_B = input_matrix_B
        self.input_matrix_A_index = input_matrix_A_index
        self.input_matrix_B_index = input_matrix_B_index

    @override
    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """Generate content for dependent case without retry loop.

        If any input matrix is provided by an upstream component, accept the
        product as-is (including the all-zero case). Only matrices we generate
        locally are constrained to have non-zero entries via min_element_abs.
        """

        matrix_A_entropy, matrix_B_entropy = self._split_entropy(context)
        rows, inner_dim, cols = self._determine_dimensions(context)

        matrix_A = self._generate_matrix_A(rows, inner_dim, matrix_A_entropy, context)
        matrix_B = self._generate_matrix_B(inner_dim, cols, matrix_B_entropy, context)

        sympy_sol, lib_result = self._multiply_matrices_sympy(matrix_a=matrix_A, matrix_b=matrix_B)

        # Record tool call with input data
        input_data = self._prepare_tool_call_input_data(matrix_a=matrix_A, matrix_b=matrix_B)
        context.record_tool_call(multiply_matrices.__name__, lib_result, input_data, is_final=True)

        problem_expression = matrix_A * matrix_B

        context_info = {
            "matrix_dimensions": (rows, inner_dim, cols),
            "problem_type": self.problem_type,
            "matrix_A": matrix_A,
            "matrix_B": matrix_B,
        }

        return ProblemTemplate(
            expression=problem_expression,
            variables={"matrix_A": matrix_A, "matrix_B": matrix_B},
            sympy_solution=sympy_sol,
            lib_result=lib_result,
            context_info={**context_info},
            difficulty_markers=self.build_difficulty_markers(
                context, matrix_size=(matrix_A.rows, matrix_A.cols), matrix_size_B=(matrix_B.rows, matrix_B.cols)
            ),
            difficulty=self.difficulty_level,
        )

    @override
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        """Use the mixin's generic logic for consistent result/value handling."""
        input_variables: dict[str, tuple[Any, int]] = {}
        input_variables["matrix_A"] = (self.input_matrix_A, self.input_matrix_A_index)

        if self.input_matrix_B is not None:
            assert self.input_matrix_B_index is not None
            input_variables["matrix_B"] = (self.input_matrix_B, self.input_matrix_B_index)
        else:
            input_variables["matrix_B"] = (template.context_info["matrix_B"], self.local_index)
            self.sources.update({"input_matrix_B": "local"})

        base_vars = self.get_dependent_template_variables(input_variables, self.sources)

        return base_vars

    def _split_entropy(self, context: ProblemContext) -> tuple[float, float]:
        # If both inputs provided; do not consume additional entropy
        # If only matrix_A provided, allocate entropy for matrix_B generation
        if self.input_matrix_B is not None:
            return 0.0, 0.0
        else:
            # Only matrix_A provided, need entropy for matrix_B
            return 0.0, context.entropy

    def _determine_dimensions(self, context: ProblemContext) -> tuple[int, int, int]:
        a_rows, a_cols = self.input_matrix_A.rows, self.input_matrix_A.cols

        if self.input_matrix_B is not None:
            # Both matrices provided - validate compatibility
            b_rows, b_cols = self.input_matrix_B.rows, self.input_matrix_B.cols
            if a_cols != b_rows:
                raise ValueError(
                    f"Matrix multiplication incompatible: A({a_rows}x{a_cols}) * B({b_rows}x{b_cols}) - A.cols({a_cols}) ≠ B.rows({b_rows})"
                )
            return int(a_rows), int(a_cols), int(b_cols)
        else:
            # Only matrix_A provided - generate random dimensions for B
            inner_dim = a_cols  # B must have a_cols rows for compatibility
            b_cols = self.config.get_random_matrix_size()
            return int(a_rows), int(inner_dim), int(b_cols)

    def _generate_matrix_B(
        self,
        rows: int,
        cols: int,
        matrix_entropy: float,
        context: ProblemContext,
    ) -> Matrix:
        if self.input_matrix_B is not None:
            return self.input_matrix_B
        else:
            # Generate matrix_B using the parent class logic
            mandatory = GenerationConstraints(rows=rows, cols=cols, min_element_abs=1)
            return self._get_matrix_with_constraints(context, added_constraints=mandatory, entropy=matrix_entropy)

    def _generate_matrix_A(
        self,
        rows: int,
        cols: int,
        matrix_entropy: float,
        context: ProblemContext,
    ) -> Matrix:
        return self.input_matrix_A

    def _prepare_tool_call_input_data(self, **kwargs: dict[str, Any]) -> dict[str, Any]:
        """Prepare input data for dependent generator including dependency info."""
        base_data = super()._prepare_tool_call_input_data(**kwargs)
        assert self.input_matrix_A == kwargs["matrix_a"]

        dependent_on = {"input_matrix_A": self.input_matrix_A_index}
        base_data.update({
            "input_matrix_A": MathFormatter.sympy_to_primitive(self.input_matrix_A, precision=self.precision),
        })

        if self.input_matrix_B is not None:
            assert self.input_matrix_B == kwargs["matrix_b"]
            assert self.input_matrix_B_index is not None
            dependent_on["input_matrix_B"] = self.input_matrix_B_index
            base_data["input_matrix_B"] = MathFormatter.sympy_to_primitive(
                self.input_matrix_B, precision=self.precision
            )

        base_data["dependent_on"] = dependent_on

        # Remove the inputs that are not assigned to the result of the previous step
        if self.sources is not None:
            for key, value in self.sources.items():
                if value != "result" and value != "local":
                    base_data.pop(key)
                    base_data["dependent_on"].pop(key)

        return base_data
