import hashlib
from collections.abc import Callable
from typing import Any

from linalg_zero.generator.composition.composition import (
    CompositeProblem,
    CompositionStrategy,
    ProblemComponent,
)
from linalg_zero.generator.difficulty_config import DETERMINISTIC_BASE_SEED, DETERMINISTIC_MODE
from linalg_zero.generator.entropy_control import EntropyConstraints
from linalg_zero.generator.generation_constraints import GenerationConstraints
from linalg_zero.generator.models import DifficultyCategory, Question, Task, Topic
from linalg_zero.generator.sympy.base import SympyProblemGenerator
from linalg_zero.generator.sympy.template_engine import TemplateEngine
from linalg_zero.generator.utils import set_seed


def create_composite_factory(
    components: list[ProblemComponent],
    composition_strategy: CompositionStrategy,
    difficulty_level: DifficultyCategory,
    problem_type: Task,
    topic: Topic,
) -> Callable[[], Question]:
    """
    Factory function for creating composite problem generators.
    """

    # Per-factory question counter for deterministic seeding
    counter = {"i": 0}

    def factory() -> Question:
        # Ensure deterministic generation across different scenarios (e.g. analysis vs generation)
        if DETERMINISTIC_MODE:
            base = DETERMINISTIC_BASE_SEED
            key = f"{problem_type.value}|{topic.value}|{counter['i']}".encode()
            h = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
            seed_value = ((base & 0xFFFFFFFF) << 16) ^ (h & 0xFFFFFFFF)
            set_seed(seed_value & 0x7FFFFFFF)
        generator = CompositeProblem(
            components=components,
            composition_strategy=composition_strategy,
            template_engine=TemplateEngine(),
            difficulty_level=difficulty_level,
            problem_type=problem_type,
            topic=topic,
        )
        try:
            return generator.generate()
        finally:
            counter["i"] += 1

    return factory


def create_sympy_factory(
    generator_class: type,
    difficulty_level: DifficultyCategory,
    problem_type: Task,
    topic: Topic,
    entropy: EntropyConstraints,
    gen_constraints: GenerationConstraints | None = None,
    **kwargs: Any,
) -> Callable[[], Question]:
    """
    Convenience function for generating a factory function for registry registration.
    """
    # Per-factory question counter for deterministic seeding
    counter = {"i": 0}

    def factory() -> Question:
        # Ensure deterministic generation across different scenarios (e.g. analysis vs generation)
        if DETERMINISTIC_MODE:
            base = DETERMINISTIC_BASE_SEED
            key = f"{problem_type.value}|{topic.value}|{counter['i']}".encode()
            h = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
            seed_value = ((base & 0xFFFFFFFF) << 12) ^ (h & 0xFFFFFFFF)
            set_seed(seed_value & 0x7FFFFFFF)
        value = entropy.sample_entropy()
        generator: SympyProblemGenerator = generator_class(
            difficulty_level=difficulty_level,
            problem_type=problem_type,
            topic=topic,
            template_engine=TemplateEngine(),
            entropy=value,
            local_index=0,
            gen_constraints=gen_constraints,
            constraints={},
            **kwargs,
        )
        try:
            return generator.generate()
        finally:
            counter["i"] += 1

    return factory
