"""
Systematic template inspection for all problem types in the registry.

This test file generates deterministic samples for every problem type and template
to allow manual inspection of question quality and correctness.
"""

import contextlib
import random
from collections.abc import Iterator
from itertools import product
from typing import Any
from unittest.mock import patch

import numpy as np
from sympy.core.random import seed as sympy_seed

from linalg_zero.generator.models import DifficultyCategory, QuestionTemplate, Task, Topic
from linalg_zero.generator.registry import create_default_registry
from linalg_zero.generator.sympy.template_engine import TemplateEngine


class TemplateInspector:
    """Systematic inspector for all problem templates."""

    def __init__(self, seed: int = 42):
        """Initialize with fixed seed for deterministic results."""
        random.seed(seed)
        np.random.seed(seed)
        sympy_seed(seed)
        self.registry = create_default_registry()
        self.template_engine = TemplateEngine()

    def get_all_problem_types(self) -> list[tuple[Topic, Task, DifficultyCategory]]:
        """Extract all registered problem types and their difficulties."""
        problem_types = []

        for topic in self.registry.list_topics():
            for task in self.registry.list_problem_types(topic):
                # Get the difficulty from the factory difficulties mapping
                if topic in self.registry._factory_difficulties and task in self.registry._factory_difficulties[topic]:
                    difficulty = self.registry._factory_difficulties[topic][task]
                    problem_types.append((topic, task, difficulty))

        return problem_types

    @contextlib.contextmanager
    def _deterministic_templates(
        self, verb_index: int, template_index: int | None = None, template_index_map: dict[Task, int] | None = None
    ) -> Iterator[None]:
        """Force deterministic verbs and template selection during generation.

        If template_index_map is provided, it overrides selection per Task.
        """
        from linalg_zero.generator.sympy.template_engine import TemplateEngine as _TE

        original_create_default = _TE.create_default_templates
        original_select_template = _TE.select_template

        def create_default_deterministic(
            self: "TemplateEngine",
            question_type: Task,
            difficulty: DifficultyCategory,
            is_independent: bool = True,
            **kwargs: Any,
        ) -> list[QuestionTemplate]:
            return original_create_default(
                self,
                question_type,
                difficulty,
                is_independent=is_independent,
                variables={},
                deterministic=True,
                verb_index=verb_index,
            )

        def select_template_deterministic(
            self: "TemplateEngine",
            templates: list[QuestionTemplate],
            question_type: Task,
            difficulty: DifficultyCategory,
            available_variables: dict[str, Any],
            **kwargs: Any,
        ) -> QuestionTemplate:
            if template_index_map is not None:
                idx = template_index_map.get(question_type, 0)
            else:
                assert template_index is not None
                idx = template_index
            context_info = available_variables.pop("context_info", None)
            result = original_select_template(
                self,
                templates,
                question_type,
                difficulty,
                available_variables,
                template_index=idx,
            )
            result.context_info = context_info
            return result

        with (
            patch.object(_TE, "create_default_templates", create_default_deterministic),
            patch.object(_TE, "select_template", select_template_deterministic),
        ):
            yield

    def _composite_steps_for_task(self, topic: Topic, task: Task) -> list[tuple[Task, bool]]:
        """Return (step_task, is_independent) using registry metadata."""
        return self.registry.get_composite_components(topic, task)

    def _composite_template_lengths(self, difficulty: DifficultyCategory, steps: list[tuple[Task, bool]]) -> list[int]:
        lengths: list[int] = []
        for step_task, is_ind in steps:
            step_templates = self.template_engine.create_default_templates(
                question_type=step_task,
                difficulty=difficulty,
                variables={},
                is_independent=is_ind,
                deterministic=True,
                verb_index=0,
            )
            lengths.append(len(step_templates))
        return lengths

    def generate_sample_for_problem_type(
        self,
        topic: Topic,
        task: Task,
        difficulty: DifficultyCategory,
        template_index: int = 0,
        template_index_map: dict[Task, int] | None = None,
    ) -> dict[str, Any]:
        """Generate a sample question for a specific problem type and template index."""

        # Set seed for deterministic generation
        random.seed(42 + template_index)
        np.random.seed(42 + template_index)
        sympy_seed(42 + template_index)

        # Get factory for this problem type
        factory = self.registry.get_factory(topic, task)

        # Check if this is a composite problem
        steps = self._composite_steps_for_task(topic, task)
        is_composite = len(steps) > 0

        if is_composite:
            # Composite problems: stabilize wording; selection handled in outer generator
            steps = self._composite_steps_for_task(topic, task)
            step_lengths = self._composite_template_lengths(difficulty, steps)
            # Use provided per-step selection if available; otherwise default to modulo mapping
            index_map = template_index_map
            if index_map is None:
                index_map = {}
                for (step_task, _), ln in zip(steps, step_lengths, strict=True):
                    index_map[step_task] = template_index % max(1, ln)
            with self._deterministic_templates(verb_index=0, template_index_map=index_map):
                question = factory()
            return {
                "topic": topic,
                "task": task,
                "difficulty": difficulty,
                "template_index": template_index,
                "total_templates": int(np.prod(step_lengths)) if step_lengths else 1,
                "selected_template": None,
                "question": question,
                "is_independent": getattr(question, "is_independent", False),
                "is_composite": True,
                "generated_question_text": question.question,
            }

        # For simple problems, use deterministic template generation
        is_independent = True
        with self._deterministic_templates(verb_index=0, template_index=template_index):
            question = factory()

        # Get all templates deterministically
        templates = self.template_engine.create_default_templates(
            question_type=task,
            difficulty=difficulty,
            is_independent=is_independent,
            variables={},
            deterministic=True,
            verb_index=0,
        )

        if len(templates) == 0:
            return {
                "topic": topic,
                "task": task,
                "difficulty": difficulty,
                "template_index": template_index,
                "total_templates": 0,
                "selected_template": None,
                "question": question,
                "is_independent": is_independent,
                "is_composite": False,
                "error": "No templates found",
            }

        # Select specific template by index (deterministic)
        selected_template = templates[template_index % len(templates)]

        return {
            "topic": topic,
            "task": task,
            "difficulty": difficulty,
            "template_index": template_index,
            "total_templates": len(templates),
            "selected_template": selected_template,
            "question": question,
            "is_independent": is_independent,
            "is_composite": False,
        }

    def generate_all_samples(self) -> dict[str, list[dict[str, Any]]]:
        """Generate samples for all problem types and all their templates."""

        results: dict[str, list[dict[str, Any]]] = {}
        problem_types = self.get_all_problem_types()

        for topic, task, difficulty in problem_types:
            key = f"{topic.value}_{task.value}_{difficulty.value}"
            results[key] = []

            # First, determine how many templates exist for this problem type
            steps = self._composite_steps_for_task(topic, task)
            if steps:  # If composite components exist, it's a composite problem
                total_templates = int(np.prod(self._composite_template_lengths(difficulty, steps))) if steps else 1
            else:
                templates = self.template_engine.create_default_templates(
                    question_type=task,
                    difficulty=difficulty,
                    is_independent=True,
                    variables={},
                    deterministic=True,
                    verb_index=0,
                )
                total_templates = len(templates)

            # Generate a sample for each template
            if steps:  # If composite components exist, it's a composite problem
                # Cross-product of step templates
                steps = self._composite_steps_for_task(topic, task)
                lengths = self._composite_template_lengths(difficulty, steps)
                index_spaces = [range(ln) for ln in lengths]
                for combo in product(*index_spaces) if index_spaces else [()]:
                    # Map per-step selection
                    index_map: dict[Task, int] = {t: i for (t, _), i in zip(steps, combo, strict=True)}
                    # Use a representative flat index for reporting order
                    flat_idx = 0
                    if lengths:
                        # simple mixed radix flattening
                        mul = 1
                        for ln, i in zip(reversed(lengths), reversed(combo), strict=True):
                            flat_idx += i * mul
                            mul *= ln
                    sample = self.generate_sample_for_problem_type(
                        topic, task, difficulty, flat_idx, template_index_map=index_map
                    )
                    results[key].append(sample)
            else:
                for template_idx in range(total_templates):
                    sample = self.generate_sample_for_problem_type(topic, task, difficulty, template_idx)
                    results[key].append(sample)

        return results


def test_inspect_all_templates() -> None:
    """Generate and display all template samples for manual inspection."""

    inspector = TemplateInspector()
    all_samples = inspector.generate_all_samples()

    print("\n" + "=" * 80)
    print("TEMPLATE INSPECTION REPORT")
    print("=" * 80)

    for problem_key, samples in all_samples.items():
        print(f"\n\n--- {problem_key} ---")
        print(f"Total templates: {len(samples)}")

        for i, sample in enumerate(samples):
            print(f"\n  Template {i + 1}/{len(samples)}:")

            if sample.get("is_composite", False):
                print("    Type: COMPOSITE PROBLEM")
                print(f"    Generated Question: {sample['generated_question_text']}")
                print(f"    Answer: {sample['question'].answer}")
            elif "error" in sample:
                print(f"    ERROR: {sample['error']}")
            else:
                print("    Type: TEMPLATE-BASED PROBLEM")
                print(f"    Template String: {sample['selected_template'].template_string}")
                print(f"    Required Variables: {sample['selected_template'].required_variables}")
                print(f"    Generated Question: {sample['question'].question}")
                print(f"    Answer: {sample['question'].answer}")

            print(f"    Is Independent: {sample['is_independent']}")


if __name__ == "__main__":
    # Run the inspection when called directly
    test_inspect_all_templates()
