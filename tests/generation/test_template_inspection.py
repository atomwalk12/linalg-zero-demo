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

from linalg_zero.generator.models import DifficultyCategory, Task, Topic
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

        def create_default_deterministic(self, question_type, difficulty, variables, is_independent=True, **kwargs):
            return original_create_default(
                self,
                question_type,
                difficulty,
                variables,
                is_independent,
                deterministic=True,
                verb_index=verb_index,
            )

        def select_template_deterministic(self, templates, question_type, difficulty, available_variables, **kwargs):
            if template_index_map is not None:
                idx = template_index_map.get(question_type, 0)
            else:
                assert template_index is not None  # for type-checkers
                idx = template_index
            result = original_select_template(
                self,
                templates,
                question_type,
                difficulty,
                available_variables,
                template_index=idx,
            )
            return result

        with (
            patch.object(_TE, "create_default_templates", create_default_deterministic),
            patch.object(_TE, "select_template", select_template_deterministic),
        ):
            yield

    def _composite_steps_for_task(self, topic: Topic, task: Task) -> list[tuple[Task, bool]]:
        """Return (step_task, is_independent) using registry metadata."""
        try:
            return self.registry.get_composite_components(topic, task)
        except Exception:
            return []

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
        is_composite = task.name.startswith("COMPOSITE_")

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
            variables={},
            is_independent=is_independent,
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

        results = {}
        problem_types = self.get_all_problem_types()

        for topic, task, difficulty in problem_types:
            key = f"{topic.value}_{task.value}_{difficulty.value}"
            results[key] = []

            # First, determine how many templates exist for this problem type
            if task.name.startswith("COMPOSITE_"):
                steps = self._composite_steps_for_task(topic, task)
                total_templates = int(np.prod(self._composite_template_lengths(difficulty, steps))) if steps else 1
            else:
                templates = self.template_engine.create_default_templates(
                    question_type=task,
                    difficulty=difficulty,
                    is_independent=True,
                    deterministic=True,
                    verb_index=0,
                    variables={},
                )
                total_templates = len(templates)

            # Generate a sample for each template
            if task.name.startswith("COMPOSITE_"):
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


def test_inspect_specific_problem_type():
    """Example of inspecting a specific problem type in detail."""

    inspector = TemplateInspector()

    # Example: Inspect DETERMINANT problem
    samples = []
    try:
        for template_idx in range(10):  # Try up to 10 templates
            sample = inspector.generate_sample_for_problem_type(
                Topic.LINEAR_ALGEBRA, Task.ONE_DETERMINANT, DifficultyCategory.ONE_TOOL_CALL, template_idx
            )
            samples.append(sample)
    except Exception:  # noqa: S110
        # Stop when we run out of templates
        pass

    print("\n\nDETERMINANT PROBLEM INSPECTION:")
    print(f"Found {len(samples)} template variations")

    for i, sample in enumerate(samples):
        print(f"\nTemplate {i + 1}:")
        print(f"  String: {sample['selected_template'].template_string}")
        print(f"  Variables: {sample['selected_template'].required_variables}")


def test_generate_sample_questions():
    """Generate actual question text for inspection."""

    inspector = TemplateInspector()

    # Generate a few actual questions to see the full pipeline
    problem_types = [
        (Topic.LINEAR_ALGEBRA, Task.ONE_DETERMINANT, DifficultyCategory.ONE_TOOL_CALL),
        (Topic.LINEAR_ALGEBRA, Task.ONE_TRANSPOSE, DifficultyCategory.ONE_TOOL_CALL),
        (Topic.LINEAR_ALGEBRA, Task.ONE_FROBENIUS_NORM, DifficultyCategory.ONE_TOOL_CALL),
    ]

    print("\n\nSAMPLE QUESTION GENERATION:")

    for topic, task, _ in problem_types:
        print(f"\n--- {task.value} ---")

        try:
            # Generate question using factory
            factory = inspector.registry.get_factory(topic, task)
            question = factory()

            print(f"Generated Question Text: {question.question}")
            print(f"Answer: {question.answer}")
            print(f"Topic: {question.topic}")
            print(f"Task: {question.problem_type}")
            print(f"Difficulty: {question.difficulty}")

        except Exception as e:
            print(f"Error generating question: {e}")
