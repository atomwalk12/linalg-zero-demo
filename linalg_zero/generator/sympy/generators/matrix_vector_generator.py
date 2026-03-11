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
from linalg_zero.generator.sympy.generators.base_generator import (
    MatrixVectorBaseGenerator,
)
from linalg_zero.generator.sympy.template_engine import MathFormatter
from linalg_zero.shared.lib_extra import multiply_matrices


class MatrixVectorMultiplicationGenerator(MatrixVectorBaseGenerator):
    def __init__(
        self,
        difficulty_level: DifficultyCategory,
        **kwargs: Any,
    ) -> None:
        """Initialize independent matrix-vector multiplication generator."""
        super().__init__(difficulty_level=difficulty_level, **kwargs)
        assert self.problem_type == Task.ONE_MATRIX_VECTOR_MULTIPLICATION

        # Validate that this problem type uses exactly 1 tool call
        validate_tool_calls(expected=self.config.target_tool_calls, actual=1, problem_type=self.problem_type)

    @property
    def precision(self) -> Precision:
        return Precision.MATRIX_VECTOR_MULTIPLICATION

    @override
    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """Generate matrix-vector multiplication problem content (independent)."""

        matrix_entropy, vector_entropy = self._split_entropy(context)
        rows, cols = self._determine_dimensions(context)

        matrix_A = self._generate_matrix_A(rows, cols, matrix_entropy, context)
        vector_b = self._generate_vector_b(cols, vector_entropy, context)
        sympy_sol, lib_result = self._multiply_matrices_sympy(matrix_a=matrix_A, vector_b=vector_b)

        # It is necessary to use matrix_b as keyword parameter because the lib.py function uses it
        input_data = self._prepare_tool_call_input_data(matrix_a=matrix_A, matrix_b=vector_b)
        context.record_tool_call(multiply_matrices.__name__, lib_result, input_data, is_final=True)

        problem_expression = matrix_A * vector_b

        context_info = {
            "matrix_dimensions": (rows, cols),
            "problem_type": self.problem_type,
            "matrix": matrix_A,
            "vector": vector_b,
        }

        return ProblemTemplate(
            expression=problem_expression,
            variables={"matrix": matrix_A, "vector": vector_b},
            sympy_solution=sympy_sol,
            lib_result=lib_result,
            context_info={**context_info},
            difficulty_markers=self.build_difficulty_markers(
                context, matrix_size=(matrix_A.rows, matrix_A.cols), vector_size=vector_b.rows
            ),
            difficulty=self.difficulty_level,
        )

    @override
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        """Return the variables dictionary to pass to the template engine."""
        input_variables = {"matrix": (template.context_info["matrix"], self.local_index)}
        input_variables["vector"] = (template.context_info["vector"], self.local_index)
        self.sources.update({"input_matrix": "local"})
        self.sources.update({"input_vector": "local"})
        return self.get_dependent_template_variables(input_variables, self.sources)

    def _multiply_matrices_sympy(self, matrix_a: Matrix, vector_b: Matrix) -> tuple[Matrix, list[list[float]]]:
        """Multiply two sympy matrices using lib.py function."""
        # Convert to primitives (this applies precision constraints)
        a_list = self.formatter.sympy_to_primitive(matrix_a, precision=self.precision)
        b_list = self.formatter.sympy_to_primitive(vector_b, precision=self.precision)
        assert isinstance(a_list, list) and isinstance(b_list, list)

        # Calculate using lib.py with the primitives
        lib_result = self.lib["multiply_matrices"](a_list, b_list)

        # Convert primitives back to SymPy matrices at the same precision level
        # This ensures both calculations work with the same precision
        matrix_a_precision_matched = Matrix(a_list)
        vector_b_precision_matched = Matrix(b_list)
        sympy_result = matrix_a_precision_matched * vector_b_precision_matched

        return sympy_result, lib_result

    def _split_entropy(self, context: ProblemContext) -> tuple[float, float]:
        """Split entropy between matrix and vector generation (independent)."""
        sample_args = SampleArgs(num_modules=2, entropy=context.entropy)
        split_fraction = (
            self.gen_constraints.split_fraction if self.gen_constraints.split_fraction is not None else 0.3
        )
        matrix_sample_args, vector_sample_args = sample_args.split(
            count=2, min_fraction=split_fraction, concentration_scale=10.0
        )
        return matrix_sample_args.entropy, vector_sample_args.entropy

    def _determine_dimensions(self, context: ProblemContext) -> tuple[int, int]:
        """Select matrix dimensions (independent): both from config."""
        rows = self.config.get_random_matrix_size()
        cols = self.config.get_random_matrix_size()
        return rows, cols

    def _generate_matrix_A(
        self,
        rows: int,
        cols: int,
        matrix_entropy: float,
        context: ProblemContext,
    ) -> Matrix:
        # Use constraint-based generation with specific dimensions
        # Temporarily set constraints for this specific call
        mandatory = GenerationConstraints(rows=rows, cols=cols)

        matrix_A = self._get_matrix_with_constraints(context, added_constraints=mandatory, entropy=matrix_entropy)

        return matrix_A

    def _generate_vector_b(
        self,
        cols: int,
        vector_entropy: float,
        context: ProblemContext,
    ) -> Matrix:
        # Use centralized vector generation. Provide fixed entropy via constraints
        # so the allocator records exactly this amount.
        return self._get_vector_with_constraints(context, size=cols, entropy=vector_entropy)


class MatrixVectorMultiplicationGeneratorDependent(MatrixVectorMultiplicationGenerator):
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

        assert self.problem_type == Task.ONE_MATRIX_VECTOR_MULTIPLICATION
        self.input_vector_b = input_vector_b
        self.input_vector_b_index = input_vector_b_index

    def _split_entropy(self, context: ProblemContext) -> tuple[float, float]:
        """Allocate some entropy to matrix generation even when vector is provided."""
        # Reserve all available context entropy for matrix generation; vector consumes none.
        return context.entropy, 0.0

    def _determine_dimensions(self, context: ProblemContext) -> tuple[int, int]:
        req_cols = int(self.input_vector_b.rows)
        rows = self.config.get_random_matrix_size()
        return rows, req_cols

    def _generate_vector_b(
        self,
        cols: int,
        vector_entropy: float,
        context: ProblemContext,
    ) -> Matrix:
        # No entropy usage for provided vector
        return self.input_vector_b

    def _prepare_tool_call_input_data(self, **kwargs: dict[str, Any]) -> dict[str, Any]:
        """Prepare input data for dependent generator including dependency info."""
        base_data = super()._prepare_tool_call_input_data(**kwargs)
        assert self.input_vector_b == kwargs["matrix_b"]
        base_data.update({
            "dependent_on": {"input_vector_b": self.input_vector_b_index},
            "input_vector_b": MathFormatter.sympy_to_primitive(self.input_vector_b, precision=self.precision),
        })
        return base_data

    @override
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        """Return template variables for dependent matrix-vector multiplication generator."""
        input_variables = {}
        input_variables["vector"] = (self.input_vector_b, self.input_vector_b_index)
        input_variables["matrix"] = (template.context_info["matrix"], self.local_index)
        base_vars = self.get_dependent_template_variables(input_variables, self.sources)

        return base_vars
