from abc import ABC, abstractmethod
from typing import Any

from linalg_zero.generator.context import CompositionContext, ProblemContext
from linalg_zero.generator.difficulty_config import Precision, get_problem_config
from linalg_zero.generator.entropy_control import EntropyConstraints
from linalg_zero.generator.models import (
    ComponentResult,
    DifficultyCategory,
    ProblemTemplate,
    Question,
    Task,
    Topic,
)
from linalg_zero.generator.sympy.template_engine import MathFormatter, TemplateEngine
from linalg_zero.grpo.verify import verify_answers
from linalg_zero.shared.lib import get_lib
from linalg_zero.shared.types import LibTypes


class ProblemComponent(ABC):
    """
    Abstract base class for composable problem components.

    A component represents an atomic piece of a mathematical problem that can
    be combined with other components to create more complex problems.
    """

    def __init__(self, name: Task, is_independent: bool, entropy_constraints: EntropyConstraints):
        self.name = name
        self.is_independent = is_independent
        self.entropy_constraints = entropy_constraints

    @abstractmethod
    def generate(self, context: CompositionContext) -> ComponentResult:
        """
        Generate the component's mathematical content.
        """
        pass

    def can_execute(self, context: CompositionContext) -> bool:
        """
        Check if this component can execute given the current context.
        """
        return True

    @abstractmethod
    def entropy_weight(self) -> float:
        """Relative weight for entropy allocation in compositions.

        Override in subclasses that truly perform pure transformations to return 0.0.
        Defaults to 1.0, meaning the component participates in entropy allocation.
        """
        pass


class CompositionStrategy(ABC):
    """Abstract base class for problem composition strategies."""

    @abstractmethod
    def compose(self, components: list[ProblemComponent], base_context: CompositionContext) -> list[ComponentResult]:
        """
        Execute composition strategy on the given components.
        """
        pass


class SympyProblemGenerator(ABC):
    """
    Abstract base class for SymPy-based mathematical problem generators.

    It orchestrates the interactions around the problem resolution process,
    including content generation, query/answer formatting and verification.
    """

    def __init__(
        self,
        difficulty_level: DifficultyCategory,
        problem_type: Task,
        topic: Topic,
        template_engine: TemplateEngine,
        local_index: int,
        constraints: dict[str, Any],
        entropy: float,
        is_independent: bool = True,
    ):
        self.difficulty_level = difficulty_level
        self.problem_type = problem_type
        self.topic = topic
        self.local_index = local_index
        self.config = get_problem_config(difficulty_level)
        self.lib = get_lib()
        self.entropy = entropy
        self.sources = constraints.get("sources", {})

        self.template_engine = template_engine
        self.formatter = MathFormatter()
        self.is_independent = is_independent
        self.max_attempts = 4000

    @property
    def precision(self) -> Precision:
        """The precision of the problem."""
        raise NotImplementedError("Implemented by subclasses.")

    @abstractmethod
    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """
        Generate the core mathematical content for the problem.
        """
        pass

    @abstractmethod
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        """Return the variables dictionary to pass to the template engine."""
        pass

    def format_question(self, template: ProblemTemplate) -> str:
        """
        Convert SymPy content into a natural language query.
        """
        # Get problem-specific information from subclass
        problem_type = self.problem_type
        variables = self.get_template_variables(template)

        # Get templates for the problem type
        templates = self.template_engine.create_default_templates(
            question_type=problem_type,
            difficulty=self.difficulty_level,
            variables=variables,
            is_independent=self.is_independent,
        )
        if templates:
            selected_template = self.template_engine.select_template(
                templates, problem_type, self.difficulty_level, variables
            )
            question_text = self.template_engine.generate_question(
                template=selected_template, variables=variables, precision=self.precision
            )
        else:
            raise ValueError(f"No templates available for {problem_type}")

        return question_text

    def format_solution(self, template: ProblemTemplate) -> str:
        """The solution string used as the ground truth in the final dataset entry."""
        return self.template_engine.format_answer(template.sympy_solution, precision=self.precision)

    def verify_problem(self, template: ProblemTemplate) -> bool:
        """
        Verify the mathematical correctness using end-to-end verification.
        This ensures sympy and lib.py results match.
        """
        lib_result = template.lib_result
        sympy_solution = template.sympy_solution

        ground_truth = self.formatter.sympy_to_primitive(sympy_solution, precision=self.precision)
        assert isinstance(ground_truth, LibTypes)

        if not verify_answers(ground_truth, lib_result):
            raise ValueError(f"Verification failed: sympy={ground_truth} vs lib={lib_result}")

        return True

    def generate(self) -> Question:
        """
        Orchestrates the problem generation process by generating a SymPy
        problem template, formatting it and verifying it.
        """
        with ProblemContext(self.entropy, self.difficulty_level, 0) as context:
            # Generate mathematical content
            template = self.generate_mathematical_content(context)

            # Format natural language components
            question_text = self.format_question(template)
            answer_text = self.format_solution(template)

            # Verify correctness
            is_valid = self.verify_problem(template)

            return Question(
                question=question_text,
                answer=answer_text,
                difficulty=self.difficulty_level,
                topic=self.topic,
                problem_type=self.problem_type,
                is_valid=is_valid,
                entropy_used=context.used_entropy,
                tool_calls_required=context.tool_calls_count,
                stepwise=context.stepwise_results,
                golden=context.golden_result,
            )

    def build_difficulty_markers(self, context: ProblemContext, **kwargs: Any) -> dict[str, Any]:
        return {
            "entropy_used": context.used_entropy,
            "available_entropy": context.entropy,
            "tool_calls": self.config.target_tool_calls,
            **kwargs,
        }

    def _prepare_tool_call_input_data(self, **kwargs: dict[str, Any]) -> dict[str, Any]:
        converted_kwargs = {
            key: self.formatter.sympy_to_primitive(value, precision=self.precision) for key, value in kwargs.items()
        }
        return {"generator_type": self.__class__.__name__, **converted_kwargs}

    def get_dependent_template_variables(
        self, input_variables: dict[str, tuple[Any, int]], sources: dict[str, str]
    ) -> dict[str, Any]:
        """Generic implementation for result/value reference handling."""
        base_vars: dict[str, Any] = {}
        base_vars["context_info"] = []

        for var_name, (value, gen_source) in input_variables.items():
            source_var_name = sources[f"input_{var_name}"]

            if source_var_name == "result":
                # This hides the result of the previous component in the template
                base_vars[var_name] = f"the result from step {gen_source + 1}"
                base_vars["context_info"].append({
                    "source_index": gen_source,
                    "generate_new": True,
                    "source_var": self.sources[f"input_{var_name}"],
                    "local_index": self.local_index,
                    "template_var": var_name,
                })
            elif source_var_name == "local":
                # This shows the exact value from a previous component
                base_vars[var_name] = value
                base_vars["context_info"].append({
                    "source_index": gen_source,
                    "generate_new": True,
                    "source_var": var_name,
                    "local_index": self.local_index,
                    "template_var": var_name,
                })
            else:
                raise ValueError(f"Invalid source variable: {source_var_name}")
                # This shows the exact value from a previous component
                base_vars[var_name] = value
                base_vars["context_info"].append({
                    "source_index": gen_source,
                    "generate_new": False,
                    "source_var": self.sources[f"input_{var_name}"],
                    "local_index": self.local_index,
                    "template_var": var_name,
                })

        return base_vars
