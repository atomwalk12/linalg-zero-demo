from abc import abstractmethod
from typing import Any

import sympy
from sympy import Float, Integer, Rational

from linalg_zero.generator.composition.composition import (
    ComponentResult,
    CompositionContext,
    ProblemComponent,
)
from linalg_zero.generator.entropy_control import EntropyConstraints
from linalg_zero.generator.generation_constraints import GenerationConstraints
from linalg_zero.generator.models import Task, Topic
from linalg_zero.generator.sympy.base import ProblemContext, ProblemTemplate, SympyProblemGenerator
from linalg_zero.generator.sympy.generators.determinant_generator import (
    DeterminantGenerator,
    DeterminantGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.frobenius_norm_generator import (
    FrobeniusNormGenerator,
    FrobeniusNormGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.linear_system_generator import (
    LinearSystemGenerator,
    LinearSystemGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.matrix_cofactor_generator import (
    MatrixCofactorGenerator,
    MatrixCofactorGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.matrix_inverse_generator import (
    MatrixInverseGenerator,
    MatrixInverseGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.matrix_matrix_generator import (
    MatrixMatrixMultiplicationGenerator,
    MatrixMatrixMultiplicationGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.matrix_rank_generator import (
    MatrixRankGenerator,
    MatrixRankGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.matrix_trace_generator import (
    MatrixTraceGenerator,
    MatrixTraceGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.matrix_transpose_generator import (
    MatrixTransposeGenerator,
    MatrixTransposeGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.matrix_vector_generator import (
    MatrixVectorMultiplicationGenerator,
    MatrixVectorMultiplicationGeneratorDependent,
)


class SympyGeneratorWrapperComponent(ProblemComponent):
    """Generic base class for wrapping sympy generators in the composition system."""

    def __init__(
        self,
        name: Task,
        generator_class: type[SympyProblemGenerator],
        component_type: Task,
        topic: Topic,
        constraints: dict[str, Any],
        entropy_constraints: EntropyConstraints,
        gen_constraints: GenerationConstraints | None = None,
        **kwargs: Any,
    ) -> None:
        is_independent = constraints.get("is_independent")
        assert isinstance(is_independent, bool)
        super().__init__(name, is_independent=is_independent, entropy_constraints=entropy_constraints, **kwargs)
        self.constraints = constraints
        self.gen_constraints = gen_constraints
        self.generator_class = generator_class
        self.component_type = component_type
        self.topic = topic

    def get_generator_params(self, context: CompositionContext, input_names: list[str]) -> dict[str, Any]:
        """Extract previous component results to use as inputs."""
        if not self.is_independent:
            params = {}
            input_indices = self.constraints["input_indices"]
            sources = self.constraints.get("sources", {})

            # Validate we have indices for all input names
            for input_name in input_names:
                if input_name not in input_indices:
                    raise ValueError(f"Missing input_index for input '{input_name}'")

                component_index = input_indices[input_name]
                source_type = sources[input_name]

                previous_result = context.component_results[component_index]

                # Get the appropriate data based on source type
                if source_type == "result":
                    # Get the computed result from the previous component
                    if not hasattr(previous_result.template, "sympy_solution"):
                        raise ValueError(f"Previous component result has no sympy_solution: {previous_result}")
                    previous_sol = previous_result.template.sympy_solution
                else:
                    # Get a specific variable from the previous component's variables
                    if not hasattr(previous_result.template, "variables"):
                        raise ValueError(f"Previous component result has no variables: {previous_result}")

                    variables = previous_result.template.variables
                    if source_type not in variables:
                        available_vars = list(variables.keys())
                        raise ValueError(
                            f"Variable '{source_type}' not found in component {component_index}. Available variables: {available_vars}"
                        )

                    previous_sol = variables[source_type]

                self._validate_dependent_input(previous_sol)

                # Add to params
                params[input_name] = previous_sol
                params[f"{input_name}_index"] = component_index

            return params
        return {}

    def _get_input_validation_spec(self) -> dict[str, bool]:
        """Subclasses may override to declare constraints for dependent input.

        Supported flags:
        - require_matrix: input must be a sympy.Matrix
        - non_empty: rows > 0 and cols > 0
        - column_vector: cols == 1
        - square: rows == cols
        - numeric_only: all elements are numeric (Integer, Float, Rational)
        """
        return {}

    def _validate_dependent_input(self, value: Any) -> None:
        """Validate dependent input according to subclass spec."""

        spec = self._get_input_validation_spec()
        if not spec:
            return

        is_matrix = isinstance(value, sympy.Matrix)
        if spec.get("require_matrix", False) and not is_matrix:
            raise TypeError(f"Expected dependent input to be a sympy Matrix, got {type(value)}")
        if not is_matrix:
            raise TypeError(f"Dependent input must be a sympy Matrix-like with shape, got {type(value)}")

        rows, cols = value.shape

        if spec.get("non_empty", False) and (rows == 0 or cols == 0):
            raise ValueError(f"Dependent input matrix cannot be empty, got shape {value.shape}")

        if spec.get("column_vector", False) and cols != 1:
            raise ValueError(f"Dependent input must be a column vector with shape (n, 1), got shape {value.shape}")

        if spec.get("square", False) and rows != cols:
            raise ValueError(f"Dependent input must be square, got shape {value.shape}")

        if spec.get("numeric_only", False) and not all(
            isinstance(element, Integer | Float | Rational) for element in value
        ):
            raise ValueError("Dependent input must contain only numeric elements")

    @abstractmethod
    def get_input_name(self) -> list[str]:
        pass

    def generate(self, context: CompositionContext) -> ComponentResult:
        # This context is used for communication and state tracking
        problem_context = ProblemContext(
            entropy=context.entropy, difficulty_level=context.difficulty_level, step_counter=context._step_counter
        )

        # Get any additional parameters for parameterized generation
        additional_params = self.get_generator_params(context, self.get_input_name())
        additional_params["constraints"] = self.constraints
        additional_params["gen_constraints"] = self.gen_constraints

        # Now, we perform the 3 key steps involved in component generation
        generator: SympyProblemGenerator = self.generator_class(
            difficulty_level=context.difficulty_level,
            problem_type=self.component_type,
            topic=self.topic,
            entropy=problem_context.entropy,
            is_independent=self.is_independent,
            template_engine=context.template_engine,
            local_index=context.local_index,
            **additional_params,
        )
        template: ProblemTemplate = generator.generate_mathematical_content(problem_context)
        generator.verify_problem(template)

        # Transfer the state of the problem context to the new problem template
        formatted_template = ProblemTemplate(
            expression=template.expression,
            variables=template.variables,
            sympy_solution=template.sympy_solution,
            lib_result=template.lib_result,
            context_info={
                **template.context_info,
            },
            difficulty_markers=template.difficulty_markers,
            difficulty=template.difficulty,
        )

        context.stepwise_results.extend(problem_context.stepwise_results)
        context.golden_result.update(problem_context.golden_result)
        context._step_counter = problem_context._step_counter

        return ComponentResult(
            template=formatted_template,
            entropy_consumed=problem_context.used_entropy,
            tool_calls_used=problem_context.tool_calls_count,
            generator=generator,
        )


class MatrixVectorMultiplicationWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the MatrixVectorMultiplicationGenerator."""

    def __init__(self, name: Task, **kwargs: Any) -> None:
        constraints = kwargs["constraints"]
        is_independent = constraints["is_independent"]
        generator_cls = (
            MatrixVectorMultiplicationGenerator if is_independent else MatrixVectorMultiplicationGeneratorDependent
        )
        super().__init__(
            name=name,
            generator_class=generator_cls,
            component_type=Task.ONE_MATRIX_VECTOR_MULTIPLICATION,
            topic=Topic.LINEAR_ALGEBRA,
            **kwargs,
        )

    def get_input_name(self) -> list[str]:
        return ["input_vector_b"]

    def entropy_weight(self) -> float:
        if self.is_independent:
            return 1.0

        # This component still needs to generate a matrix, even if a vector is
        # provided, so we provide 0.5 entropy weight.
        return 0.5

    def _get_input_validation_spec(self) -> dict[str, bool]:
        return {"require_matrix": True, "non_empty": True, "column_vector": True}


class MatrixMatrixMultiplicationWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the MatrixMatrixMultiplicationGeneratorDependent."""

    def __init__(self, name: Task, **kwargs: Any) -> None:
        self.constraints = kwargs["constraints"]
        is_independent = self.constraints["is_independent"]
        generator_cls = (
            MatrixMatrixMultiplicationGenerator if is_independent else MatrixMatrixMultiplicationGeneratorDependent
        )
        super().__init__(
            name=name,
            generator_class=generator_cls,
            component_type=Task.ONE_MATRIX_MATRIX_MULTIPLICATION,
            topic=Topic.LINEAR_ALGEBRA,
            **kwargs,
        )

    def entropy_weight(self) -> float:
        # If independent, allocate all entropy
        if self.is_independent:
            return 1.0

        input_indices = self.constraints.get("input_indices", {})
        if "input_matrix_B" not in input_indices:
            # matrix_B is not provided, so it will be generated inside the component
            # allocate half of the total entropy amount
            return 0.5
        else:
            # All components provided, allocate no entropy
            return 0.0

    def get_input_name(self) -> list[str]:
        # Check if we need both matrices or just one based on constraints
        input_indices = self.constraints["input_indices"]
        if "input_matrix_B" in input_indices:
            return ["input_matrix_A", "input_matrix_B"]
        else:
            return ["input_matrix_A"]

    def _get_input_validation_spec(self) -> dict[str, bool]:
        return {"require_matrix": True, "non_empty": True}


class LinearSystemSolverWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the LinearSystemGenerator."""

    def __init__(self, name: Task, **kwargs: Any) -> None:
        constraints = kwargs["constraints"]
        is_independent = constraints["is_independent"]
        generator_cls = LinearSystemGenerator if is_independent else LinearSystemGeneratorDependent
        super().__init__(
            name=name,
            generator_class=generator_cls,
            component_type=Task.ONE_LINEAR_SYSTEM_SOLVER,
            topic=Topic.LINEAR_ALGEBRA,
            **kwargs,
        )

    def entropy_weight(self) -> float:
        if self.is_independent:
            return 1.0

        # This component still needs to generate a matrix, even if vector b is
        # provided, so we provide 0.5 entropy weight.
        return 0.5

    def get_input_name(self) -> list[str]:
        return ["input_vector_b"]

    def _get_input_validation_spec(self) -> dict[str, bool]:
        return {"require_matrix": True, "non_empty": True, "column_vector": True}


class FrobeniusNormWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the FrobeniusNormGenerator."""

    def __init__(self, name: Task, **kwargs: Any) -> None:
        constraints = kwargs["constraints"]
        is_independent = constraints["is_independent"]
        generator_cls = FrobeniusNormGenerator if is_independent else FrobeniusNormGeneratorDependent
        super().__init__(
            name=name,
            generator_class=generator_cls,
            component_type=Task.ONE_FROBENIUS_NORM,
            topic=Topic.LINEAR_ALGEBRA,
            **kwargs,
        )

    def get_input_name(self) -> list[str]:
        return ["input_matrix"]

    def _get_input_validation_spec(self) -> dict[str, bool]:
        return {"require_matrix": True, "non_empty": True}

    def entropy_weight(self) -> float:
        if self.is_independent:
            return 1.0
        return 0.0


class DeterminantWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the DeterminantGenerator."""

    def __init__(self, name: Task, **kwargs: Any) -> None:
        constraints = kwargs["constraints"]
        is_independent = constraints["is_independent"]
        generator_cls = DeterminantGenerator if is_independent else DeterminantGeneratorDependent
        super().__init__(
            name=name,
            generator_class=generator_cls,
            component_type=Task.ONE_DETERMINANT,
            topic=Topic.LINEAR_ALGEBRA,
            **kwargs,
        )

    def entropy_weight(self) -> float:
        if self.is_independent:
            return 1.0
        return 0.0

    def get_input_name(self) -> list[str]:
        return ["input_matrix"]

    def _get_input_validation_spec(self) -> dict[str, bool]:
        return {"require_matrix": True, "non_empty": True, "square": True}


class RankWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the MatrixRankGenerator."""

    def __init__(self, name: Task, **kwargs: Any) -> None:
        constraints = kwargs["constraints"]
        is_independent = constraints["is_independent"]
        generator_cls = MatrixRankGenerator if is_independent else MatrixRankGeneratorDependent
        super().__init__(
            name=name,
            generator_class=generator_cls,
            component_type=Task.ONE_RANK,
            topic=Topic.LINEAR_ALGEBRA,
            **kwargs,
        )

    def entropy_weight(self) -> float:
        if self.is_independent:
            return 1.0
        return 0.0

    def get_input_name(self) -> list[str]:
        return ["input_matrix"]

    def _get_input_validation_spec(self) -> dict[str, bool]:
        return {"require_matrix": True, "non_empty": True, "numeric_only": True}


class TransposeWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the MatrixTransposeGenerator."""

    def __init__(self, name: Task, **kwargs: Any) -> None:
        constraints = kwargs["constraints"]
        is_independent = constraints["is_independent"]
        generator_cls = MatrixTransposeGenerator if is_independent else MatrixTransposeGeneratorDependent
        super().__init__(
            name=name,
            generator_class=generator_cls,
            component_type=Task.ONE_TRANSPOSE,
            topic=Topic.LINEAR_ALGEBRA,
            **kwargs,
        )

    def entropy_weight(self) -> float:
        if self.is_independent:
            return 1.0
        return 0.0

    def get_input_name(self) -> list[str]:
        return ["input_matrix"]

    def _get_input_validation_spec(self) -> dict[str, bool]:
        return {"require_matrix": True, "non_empty": True}


class MatrixTraceWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the MatrixTraceGenerator."""

    def __init__(self, name: Task, **kwargs: Any) -> None:
        constraints = kwargs["constraints"]
        is_independent = constraints["is_independent"]
        generator_cls = MatrixTraceGenerator if is_independent else MatrixTraceGeneratorDependent
        super().__init__(
            name=name,
            generator_class=generator_cls,
            component_type=Task.ONE_TRACE,
            topic=Topic.LINEAR_ALGEBRA,
            **kwargs,
        )

    def entropy_weight(self) -> float:
        if self.is_independent:
            return 1.0
        return 0.0

    def get_input_name(self) -> list[str]:
        return ["input_matrix"]

    def _get_input_validation_spec(self) -> dict[str, bool]:
        return {"require_matrix": True, "non_empty": True, "square": True}


class MatrixInverseWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the MatrixInverseGenerator."""

    def __init__(self, name: Task, **kwargs: Any) -> None:
        constraints = kwargs["constraints"]
        is_independent = constraints["is_independent"]
        generator_cls = MatrixInverseGenerator if is_independent else MatrixInverseGeneratorDependent
        super().__init__(
            name=name,
            generator_class=generator_cls,
            component_type=Task.ONE_INVERSE,
            topic=Topic.LINEAR_ALGEBRA,
            **kwargs,
        )

    def entropy_weight(self) -> float:
        if self.is_independent:
            return 1.0
        return 0.0

    def get_input_name(self) -> list[str]:
        return ["input_matrix"]

    def _get_input_validation_spec(self) -> dict[str, bool]:
        return {"require_matrix": True, "non_empty": True, "square": True, "invertible": True}


class MatrixCofactorWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the MatrixCofactorGenerator."""

    def __init__(self, name: Task, **kwargs: Any) -> None:
        constraints = kwargs["constraints"]
        is_independent = constraints["is_independent"]
        generator_cls = MatrixCofactorGenerator if is_independent else MatrixCofactorGeneratorDependent
        super().__init__(
            name=name,
            generator_class=generator_cls,
            component_type=Task.ONE_COFACTOR,
            topic=Topic.LINEAR_ALGEBRA,
            **kwargs,
        )

    def entropy_weight(self) -> float:
        if self.is_independent:
            return 1.0
        return 0.0

    def get_input_name(self) -> list[str]:
        return ["input_matrix"]

    def _get_input_validation_spec(self) -> dict[str, bool]:
        return {"require_matrix": True, "non_empty": True, "square": True}
