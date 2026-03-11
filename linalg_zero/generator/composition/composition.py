from typing import Any

from typing_extensions import override

from linalg_zero.generator.context import CompositionContext
from linalg_zero.generator.entropy_control import EntropyConstraints
from linalg_zero.generator.models import (
    ComponentResult,
    CompositeResultBuilder,
    DifficultyCategory,
    Task,
    Topic,
)
from linalg_zero.generator.sympy.base import (
    CompositionStrategy,
    ProblemComponent,
    ProblemContext,
    ProblemTemplate,
    SympyProblemGenerator,
)
from linalg_zero.generator.sympy.template_engine import TemplateEngine
from linalg_zero.grpo.verify import verify_answers
from linalg_zero.shared.types import LibTypes
from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)


class SequentialComposition(CompositionStrategy):
    """
    Sequential composition strategy.

    Executes components in order, where each component can use results
    from previous components. Useful for multi-step problems.
    """

    def compose(self, components: list[ProblemComponent], base_context: CompositionContext) -> list[ComponentResult]:
        """Execute components using DeepMind-style entropy distribution."""
        results = []

        # Allocate entropy proportionally to integer module counts (simple, DM-style).
        def component_modules(c: ProblemComponent) -> float:
            return max(0, c.entropy_weight())

        weights = [component_modules(c) for c in components]
        total_modules = sum(weights)

        if total_modules <= 0:
            raise ValueError("Total modules must be > 0 for composite problems")
            # Alternative use:
            # component_sample_args = sample_args.split(len(components))
        else:
            # Instead of a uniform distribution, we sample the provided values component-wise
            # This allows to provide a range of entropy values or fixed values for each component.
            allocations: list[float] = []
            for comp in components:
                override: EntropyConstraints = comp.entropy_constraints
                entropy = override.sample_entropy()
                assert entropy is not None
                allocations.append(float(entropy))

            for alloc, weight, comp in zip(allocations, weights, components, strict=True):
                if weight == 0 and alloc != 0:
                    raise ValueError(f"Weight is 0 but allocation is {alloc} for component {comp.name}")
                if weight != 0 and alloc == 0:
                    raise ValueError(f"Weight is {weight} but allocation is 0 for component {comp.name}")

            allocations = [0.0 if weight == 0 else alloc for alloc, weight in zip(allocations, weights, strict=True)]

            # Set the composite budget to the sum of per-component allocations
            base_context.entropy = float(sum(allocations))

        for local_index, (component_wrapper, component_entropy) in enumerate(
            zip(components, allocations, strict=True)
        ):
            if not component_wrapper.can_execute(base_context):
                continue

            # Create a context copy with the allocated entropy for this component
            component_context = CompositionContext(
                component_entropy,
                base_context.difficulty_level,
                base_context._step_counter,
                template_engine=base_context.template_engine,
                local_index=local_index,
            )

            # NOTE[atom]: these variables can be useful to share state between components
            component_context.constraints = base_context.constraints.copy()
            # Pass previous component results to enable sequential data flow
            component_context.component_results = base_context.component_results.copy()

            result = component_wrapper.generate(component_context)
            base_context.record_component_result(result)

            base_context.stepwise_results.extend(component_context.stepwise_results)
            base_context.golden_result.update(component_context.golden_result)
            base_context._step_counter = component_context._step_counter

            results.append(result)

        return results


class CompositeProblem(SympyProblemGenerator):
    """
    Generator for composite mathematical problems.

    Combines multiple ProblemComponent instances using a CompositionStrategy
    to create complex, multi-part mathematical problems.
    """

    def __init__(
        self,
        components: list[ProblemComponent],
        composition_strategy: CompositionStrategy,
        template_engine: TemplateEngine,
        difficulty_level: DifficultyCategory,
        problem_type: Task,
        topic: Topic,
    ):
        super().__init__(
            entropy=0.0,
            difficulty_level=difficulty_level,
            problem_type=problem_type,
            topic=topic,
            template_engine=template_engine,
            local_index=-1,
            constraints={},
        )

        self.components = components
        self.composition_strategy = composition_strategy
        # No global sample_args: composition strategy allocates per-component entropy

    @override
    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """Generate composed mathematical content."""
        # Convert to CompositionContext with a temporary zero budget; the strategy sets the proper budget
        comp_context = CompositionContext(
            0.0,
            context.difficulty_level,
            context._step_counter,
            self.template_engine,
            local_index=-1,
        )
        comp_context.constraints = context.constraints.copy()

        # Execute all components and store their results
        component_results = self.composition_strategy.compose(self.components, comp_context)

        if not component_results:
            raise ValueError("No components could be executed")

        # This is a helper class to aggregate component results and create a composite template
        builder = CompositeResultBuilder(self.composition_strategy)
        for result in component_results:
            builder.add_component_result(result)

        template = builder.build_template(comp_context, component_results)

        # Transfer state back to original context
        self._transfer_context_state(comp_context, context)

        return template

    def _transfer_context_state(self, comp_context: CompositionContext, original_context: ProblemContext) -> None:
        """Transfer entropy and tool call tracking back to original context."""
        original_context.entropy = comp_context.entropy
        original_context.used_entropy = comp_context.used_entropy
        original_context.tool_calls_count = comp_context.tool_calls_count
        original_context.stepwise_results = comp_context.stepwise_results
        original_context.golden_result = comp_context.golden_result
        original_context._step_counter = comp_context._step_counter

    @override
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        """Not used for composite problems."""
        raise NotImplementedError("Not used for composite problems.")

    @override
    def format_question(self, template: ProblemTemplate) -> str:
        """Format composite problem as natural language multi-step question."""
        context_info = template.context_info
        composition_type = context_info["composition_type"]

        if isinstance(template.expression, list) and len(template.expression) > 1:
            if composition_type == SequentialComposition.__name__:
                return self._format_sequential_question(template)
            else:
                raise ValueError(f"Unknown composition type: {composition_type}")
        else:
            raise ValueError("Composite problem should have multiple expressions.")

    def _format_sequential_question(self, template: ProblemTemplate) -> str:
        """Format sequential composition data with the results produced by each component."""
        component_results: list[ComponentResult] = template.context_info.get("component_results", [])

        if not component_results:
            raise ValueError("Sequential composition requires component results with generators")

        step_descriptions = []
        for i, result in enumerate(component_results, 1):
            formatted_question = result.generator.format_question(result.template)
            formatted_question = formatted_question[0].lower() + formatted_question[1:]
            step_descriptions.append(f"Step {i}: {formatted_question}")

        return "\n".join(step_descriptions)

    @override
    def format_solution(self, template: ProblemTemplate) -> str:
        """Format composite problem solution using MathFormatter for clean output."""

        component_results: list[ComponentResult] = template.context_info.get("component_results", [])

        if not isinstance(template.sympy_solution, list):
            raise TypeError("The sympy solution should be a list because the number of provided components is a list.")

        if len(template.sympy_solution) == 1:
            raise ValueError("Composite problem should have multiple solutions.")

        return self.template_engine.format_composite_answer(template.sympy_solution, component_results)

    @override
    def verify_problem(self, template: ProblemTemplate) -> bool:
        """Verify the problem is mathematically correct."""
        lib_results = template.lib_result
        sympy_solutions = template.sympy_solution
        assert isinstance(sympy_solutions, list)
        assert isinstance(lib_results, list)

        component_results: list[ComponentResult] = template.context_info["component_results"]

        for sympy_solution, lib_result, result in zip(sympy_solutions, lib_results, component_results, strict=True):
            precision = result.generator.precision
            sympy_solution = self.formatter.sympy_to_primitive(sympy_solution, precision=precision)

            assert isinstance(lib_result, LibTypes)
            assert isinstance(sympy_solution, LibTypes)

            if not verify_answers(sympy_solution, lib_result):
                raise ValueError(f"Verification failed: sympy={sympy_solution} vs lib={lib_result}")

        return True
