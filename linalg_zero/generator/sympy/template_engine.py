import json
import random
from typing import Any, ClassVar

from sympy import Float, Integer, Mul, Number, Pow, Rational, Symbol
from sympy.core import Expr
from sympy.matrices import MutableDenseMatrix

from linalg_zero.generator.difficulty_config import Precision
from linalg_zero.generator.models import ComponentResult, DifficultyCategory, QuestionTemplate, Task
from linalg_zero.generator.sympy.templates import (
    get_static_templates,
)
from linalg_zero.shared.types import LibTypes
from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)


class MathFormatter:
    """
    Utilities for formatting mathematical expressions in text format.
    """

    @staticmethod
    def round_sympy_element(element: LibTypes, precision: Precision) -> LibTypes | str:
        """
        Round a SymPy element to a precision of 2. The string return type is used for
        symbol variables that may be part of specific lists.
        """
        if isinstance(element, int | float):
            if precision != Precision.FULL:
                return round(element, precision.value)
            else:
                return element
        elif isinstance(element, list):
            result = []
            for e in element:
                if isinstance(e, int | float | list):
                    result.append(MathFormatter.round_sympy_element(e, precision))
                elif isinstance(e, str):
                    # Symbol elements are appended as they are. These are used for instance
                    # for the linear_system_solver problem type.
                    result.append(e)
                else:
                    raise TypeError(f"Unsupported element type in list: {type(e)} (value: {e})")
            return result
        else:
            raise TypeError(f"Unsupported element type: {type(element)}")

    @staticmethod
    def sympy_to_primitive(sympy_result: Expr, precision: Precision) -> LibTypes | str:
        """Convert sympy result to primitive type for verification."""
        result: LibTypes | str | None = None
        if isinstance(sympy_result, MutableDenseMatrix):
            list_of_lists = sympy_result.tolist()
            result = [[MathFormatter._sympy_element_to_python(element) for element in _] for _ in list_of_lists]
        elif isinstance(sympy_result, Number | Integer | Float):
            result = MathFormatter._sympy_element_to_python(sympy_result)
        elif isinstance(sympy_result, Mul | Pow):
            # Frobenius norm requires Pow and Mul expressions
            result = float(sympy_result.evalf())
        else:
            raise TypeError(f"Unsupported element type: {type(sympy_result)}")

        if precision != Precision.FULL and not isinstance(result, str):
            return MathFormatter.round_sympy_element(result, precision)
        else:
            return result

    @staticmethod
    def _sympy_element_to_python(element: Integer | Float | Number | Symbol) -> float | int | str:
        """Convert SymPy element to Python primitive, following quantum matrixutils pattern."""
        if hasattr(element, "is_Integer") and element.is_Integer:
            value = element.__int__()
            if isinstance(value, int):
                return value
            else:
                raise ValueError(f"Expected int, got {type(value)}")
        elif (hasattr(element, "is_Float") and element.is_Float) or (
            hasattr(element, "is_Number") and element.is_Number
        ):
            value = element.__float__()
            if isinstance(value, float):
                return value
            else:
                raise ValueError(f"Expected float, got {type(value)}")
        elif isinstance(element, Symbol):
            # This is used by the inverse solver because of variables that are not numbers
            return str(element)
        raise ValueError(f"Unsupported element type: {type(element)}")


class TemplateEngine:
    """
    Main engine for generating natural language questions from mathematical templates.

    This class coordinates the process of converting SymPy content into
    human-readable questions using templates and formatters.
    """

    VERBS: ClassVar[list[str]] = ["Find", "Calculate", "Compute", "Determine", "Evaluate"]

    def __init__(self) -> None:
        self.math_formatter = MathFormatter()
        self.available_matrices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        self.context_info: dict[str, Any] = {}

    def generate_question(self, template: QuestionTemplate, variables: dict[str, Any], precision: Precision) -> str:
        """
        Generate natural language question text that will be included as the
        "query" field in the final dataset entry.

        This method validates variable types upfront, then formats expressions
        before performing template substitution.
        """
        # 1. Validation checks
        missing_vars = set(template.required_variables) - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        # 2. Format mathematical expressions stored within variables
        formatted_variables = {}
        for var_name, var_value in variables.items():
            if isinstance(var_value, MutableDenseMatrix):
                formatted_variables[var_name] = self.math_formatter.sympy_to_primitive(var_value, precision)
            elif isinstance(var_value, str):
                formatted_variables[var_name] = var_value
            else:
                raise TypeError(f"Variable '{var_name}' has unsupported type {type(var_value).__name__}.")

        try:
            # 3. Apply template substitution
            question_text = template.template_string.format(**formatted_variables)
        except KeyError as e:
            raise ValueError(f"Template substitution failed: missing variable {e}") from e

        return question_text

    def format_answer(self, answer: Any, precision: Precision) -> str:
        """
        Format a SymPy matrix (can also be a vector), to be displayed in question answer.
        """
        if isinstance(answer, MutableDenseMatrix | Integer | Float | Rational | Pow | Mul):
            result = self.math_formatter.sympy_to_primitive(answer, precision)
            return json.dumps(result)
        else:
            raise TypeError(f"Variable '{answer}' has unsupported type {type(answer).__name__}.")

    def format_composite_answer(self, sympy_solutions: list[Expr], component_results: list[ComponentResult]) -> str:
        """
        Format a composite answer to be displayed in question as a JSON string.
        """
        result_dict = {}
        for i, (sol, component_result) in enumerate(zip(sympy_solutions, component_results, strict=True), 1):
            precision = component_result.generator.precision
            formatted_sol = self.math_formatter.sympy_to_primitive(sol, precision=precision)
            result_dict[f"tool_{i}"] = formatted_sol
        return json.dumps(result_dict)

    def create_default_templates(
        self,
        question_type: Task,
        difficulty: DifficultyCategory,
        variables: dict[str, Any],
        is_independent: bool,
        *,
        deterministic: bool = False,
        verb_index: int = 0,
    ) -> list[QuestionTemplate]:
        """
        Create default question templates for common problem types.
        This simplifies the creation of question/answer pairs.
        """

        context_info = variables.pop("context_info", None)

        # Derive index of previous step (if any) from context info
        previous_step: int | None = None
        if isinstance(context_info, list) and context_info:
            for record in context_info:
                # Use the referenced component index when we generate a new label for a prior result
                if record.get("generate_new") and record.get("source_var") == "result":
                    previous_step = record.get("source_index")
                    break

        # Use the variables when creating the templates
        return get_static_templates(question_type, difficulty, previous_step)

        # NOTE: Uncomment to use dynamic templates
        # variables = self.customise_templates(context_info)
        # return get_dynamic_templates(question_type, difficulty, variables)

    def customise_templates(self, context_info: list[dict[str, Any]]) -> dict[str, Any]:
        allocated_vars = {}

        for record in context_info:
            source_var = record["source_var"]
            source_idx = record["source_index"]
            local_idx = record["local_index"]
            template_var = record["template_var"]
            generate_new = record["generate_new"]

            if generate_new:
                # Allocate a fresh variable
                new_var = self.available_matrices.pop(0)
                allocated_vars[template_var] = new_var

                # Initialize component context if needed
                if local_idx not in self.context_info:
                    self.context_info[local_idx] = {}
                self.context_info[local_idx][source_var] = new_var
            else:
                # Reuse variable from previous component
                if source_idx in self.context_info and source_var in self.context_info[source_idx]:
                    previous_var = self.context_info[source_idx][source_var]
                    allocated_vars[template_var] = previous_var
                else:
                    raise ValueError(f"Cannot find variable '{source_var}' for component '{source_idx}'")

        return allocated_vars

    def select_template(
        self,
        templates: list[QuestionTemplate],
        question_type: Task,
        difficulty: DifficultyCategory,
        available_variables: dict[str, Any],
        *,
        template_index: int | None = None,
    ) -> QuestionTemplate:
        """
        Select an appropriate template from a list based on specified criteria.
        The selection checks the question type and problem difficulty.
        """
        if not templates:
            raise ValueError("No templates available")

        # Filter by both criteria simultaneously. We look for templates that
        # match both in terms of question type as well as difficulty level
        candidates = [t for t in templates if t.question_type == question_type and t.difficulty_level == difficulty]

        # Filter by available variables
        if available_variables:
            available_vars = set(available_variables.keys())
            variable_compatible = [t for t in candidates if set(t.required_variables) == available_vars]
            if variable_compatible:
                candidates = variable_compatible
            else:
                raise ValueError(
                    f"No variable compatible templates found for {available_variables}, candidates: {candidates}"
                )

        # Deterministic selection by index if provided
        if template_index is not None:
            pool = candidates if candidates else templates
            return pool[template_index % len(pool)]

        return random.choice(candidates if candidates else templates)
