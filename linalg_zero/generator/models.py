from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from sympy import Expr

from linalg_zero.shared.types import LibTypes

if TYPE_CHECKING:
    from linalg_zero.generator.context import CompositionContext
    from linalg_zero.generator.sympy.base import CompositionStrategy, SympyProblemGenerator


class Topic(Enum):
    """Enum for topics used in problem generation."""

    LINEAR_ALGEBRA = "linear_algebra"


class Task(Enum):
    """Enum for problem types used in problem generation."""

    SEQUENTIAL_PROBLEM = "sequential_problem"

    # Single tool call problems
    ONE_DETERMINANT = "one_determinant"
    ONE_LINEAR_SYSTEM_SOLVER = "one_linear_system_solver"
    ONE_MATRIX_VECTOR_MULTIPLICATION = "one_matrix_vector_multiplication"
    ONE_MATRIX_MATRIX_MULTIPLICATION = "one_matrix_matrix_multiplication"
    ONE_FROBENIUS_NORM = "one_frobenius_norm"
    ONE_RANK = "one_matrix_rank"
    ONE_TRANSPOSE = "one_matrix_transpose"
    ONE_INVERSE = "one_matrix_inverse"
    ONE_TRACE = "one_matrix_trace"
    ONE_COFACTOR = "one_matrix_cofactor"

    # Two tool call problems
    TWO_TRANSPOSE_DETERMINANT = "two_transpose_determinant"
    TWO_COFACTOR_RANK = "two_cofactor_rank"
    TWO_TRANSPOSE_FROBENIUS = "two_transpose_frobenius"
    TWO_COFACTOR_TRACE = "two_cofactor_trace"

    # Not used
    TWO_COFACTOR_FROBENIUS = "two_cofactor_frobenius"

    # Cause value explosion due to matrix multiplications
    THREE_COFACTOR_MATRIXMULT_RANK = "three_cofactor_matrixmult_rank"
    THREE_SYSTEM_MATRIXMULT_FROBENIUS = "three_system_matrixmult_frobenius"
    THREE_MATRIXVECTOR_SYSTEM_FROBENIUS = "three_matrixvector_system_frobenius"
    THREE_TRANSPOSE_DETERMINANT_TRACE = "three_transpose_determinant_trace"

    # Stable output values
    THREE_TRANSPOSE_COFACTOR_RANK = "three_transpose_cofactor_rank"
    THREE_COFACTOR_TRANSPOSE_TRACE = "three_cofactor_transpose_trace"
    THREE_TRANSPOSE_COFACTOR_FROBENIUS = "three_transpose_cofactor_frobenius"


class DifficultyCategory(Enum):
    """Enum for difficulty categories used in problem generation."""

    ONE_TOOL_CALL = 1
    TWO_TOOL_CALLS = 2
    THREE_TOOL_CALLS = 3

    def __str__(self) -> str:
        """Return the string value for compatibility with existing code."""
        if self == DifficultyCategory.ONE_TOOL_CALL:
            return "easy (1 tool call)"
        elif self == DifficultyCategory.TWO_TOOL_CALLS:
            return "medium (2 tool calls)"
        elif self == DifficultyCategory.THREE_TOOL_CALLS:
            return "hard (3 tool calls)"
        else:
            raise ValueError(f"Invalid difficulty category: {self}")


@dataclass
class QuestionTemplate:
    """
    Data class template for generating natural language questions.
    """

    template_string: str
    required_variables: list[str]
    difficulty_level: DifficultyCategory
    question_type: Task
    context_info: dict[str, Any] | None = None


@dataclass
class Question:
    """Represents a generated question with its answer."""

    question: str
    answer: str
    difficulty: DifficultyCategory
    topic: Topic
    problem_type: Task
    is_valid: bool = True
    entropy_used: float = 0.0
    tool_calls_required: int = 0
    stepwise: list[dict[str, str]] = field(default_factory=list)
    golden: dict[str, str] = field(default_factory=dict)


@dataclass
class ProblemTemplate:
    """
    Data class with the main components for a problem.
    """

    expression: Expr
    variables: dict[str, Expr]
    sympy_solution: Expr | list[Expr] | str
    lib_result: LibTypes
    context_info: dict[str, Any]
    difficulty_markers: dict[str, float | tuple]
    difficulty: DifficultyCategory | None = None


class CompositionType(Enum):
    """
    Types of problem composition strategies

    The mathematics_dataset package contains the following composition types:
    - Sequential composition feeds the output of one component into the next
    - Hierarchical composition with peel() method for parent-child relationships
    - Parallel composition for independent sub-problems
    - Conditional composition that adapts based on intermediate results
    """

    # NOTE[Future]: Implement other composition types here
    SEQUENTIAL = "sequential"


@dataclass
class ComponentResult:
    """Result from executing a problem component."""

    template: ProblemTemplate
    generator: "SympyProblemGenerator"
    entropy_consumed: float = 0.0
    tool_calls_used: int = 0


@dataclass
class CompositeResultBuilder:
    """Builder for combining component results into a unified template."""

    def __init__(self, composition_strategy: "CompositionStrategy"):
        self.composition_strategy = composition_strategy
        self.expressions: list = []
        self.solutions: list = []
        self.lib_results: list = []
        self.context_info: dict[str, Any] = {}
        self.component_templates: list[ProblemTemplate] = []

    def add_component_result(self, result: ComponentResult) -> None:
        """Add a component result to the builder."""
        template = result.template

        self.expressions.append(template.expression)
        self.component_templates.append(template)
        # Variables are accessed directly from component results via sources system
        # No need to aggregate here as it would cause naming conflicts

        self.solutions.append(template.sympy_solution)
        self.lib_results.append(template.lib_result)

        self.context_info.update(template.context_info)

    def build_template(
        self, comp_context: "CompositionContext", component_results: list[ComponentResult]
    ) -> ProblemTemplate:
        """Build the final composite template."""
        return ProblemTemplate(
            expression=self._build_main_expression(),
            variables=self._deduplicate_variables(),
            sympy_solution=self.solutions,
            lib_result=self.lib_results,
            context_info=self._build_context_info(comp_context, component_results),
            difficulty_markers=self._build_difficulty_markers(comp_context),
        )

    def _build_main_expression(self) -> Expr | list[Expr]:
        """Build the main expression (single vs list)."""
        return self.expressions[0] if len(self.expressions) == 1 else self.expressions

    def _deduplicate_variables(self) -> dict[str, Expr]:
        """Return empty dict since composite problems don't aggregate variables."""
        # Variables are accessed directly from individual component results
        # via the sources system in composition constraints
        return {}

    def _build_context_info(
        self, comp_context: "CompositionContext", component_results: list[ComponentResult]
    ) -> dict[str, Any]:
        """Build combined context info with composition metadata."""
        return {
            **self.context_info,
            "composition_type": self.composition_strategy.__class__.__name__,
            "component_count": len(self.component_templates),
            "total_entropy_used": comp_context.used_entropy,
            "total_tool_calls": comp_context.tool_calls_count,
            "component_templates": self.component_templates,
            "component_results": component_results,
        }

    def _build_difficulty_markers(self, comp_context: "CompositionContext") -> dict[str, Any]:
        """Build difficulty markers for the composite problem."""
        return {
            "composition_complexity": len(self.component_templates),  # the number of components
            "entropy_per_component": comp_context.used_entropy / len(self.component_templates),
            "variable_count": len(self._deduplicate_variables()),
        }
