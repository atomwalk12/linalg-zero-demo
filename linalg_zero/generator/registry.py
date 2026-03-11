import random
from collections.abc import Callable
from typing import Any

from linalg_zero.generator.composition.components import (
    DeterminantWrapperComponent,
    FrobeniusNormWrapperComponent,
    MatrixCofactorWrapperComponent,
    MatrixTraceWrapperComponent,
    RankWrapperComponent,
    TransposeWrapperComponent,
)
from linalg_zero.generator.composition.composition import (
    SequentialComposition,
)
from linalg_zero.generator.entropy_control import EntropyConstraints
from linalg_zero.generator.generation_constraints import GenerationConstraints
from linalg_zero.generator.generator_factories import (
    create_composite_factory,
    create_sympy_factory,
)
from linalg_zero.generator.models import DifficultyCategory, Question, Task, Topic
from linalg_zero.generator.sympy.base import (
    CompositionStrategy,
    ProblemComponent,
)
from linalg_zero.generator.sympy.generators.determinant_generator import DeterminantGenerator
from linalg_zero.generator.sympy.generators.frobenius_norm_generator import FrobeniusNormGenerator
from linalg_zero.generator.sympy.generators.matrix_cofactor_generator import MatrixCofactorGenerator
from linalg_zero.generator.sympy.generators.matrix_rank_generator import MatrixRankGenerator
from linalg_zero.generator.sympy.generators.matrix_trace_generator import MatrixTraceGenerator
from linalg_zero.generator.sympy.generators.matrix_transpose_generator import (
    MatrixTransposeGenerator,
)
from linalg_zero.generator.utils import load_entropy_settings, print_entropy_settings


def register_one_determinant_factory(
    registry: "FactoryRegistry", entropy: tuple[float, float] | float, gen_constraints: dict[str, Any] | None = None
) -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.ONE_DETERMINANT,
        create_sympy_factory(
            generator_class=DeterminantGenerator,
            topic=Topic.LINEAR_ALGEBRA,
            problem_type=Task.ONE_DETERMINANT,
            difficulty_level=DifficultyCategory.ONE_TOOL_CALL,
            gen_constraints=_merge_gen_constraints({"square": True}, gen_constraints),
            entropy=EntropyConstraints(entropy),
        ),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_one_frobenius_norm_factory(
    registry: "FactoryRegistry", entropy: tuple[float, float] | float, gen_constraints: dict[str, Any] | None = None
) -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.ONE_FROBENIUS_NORM,
        create_sympy_factory(
            generator_class=FrobeniusNormGenerator,
            topic=Topic.LINEAR_ALGEBRA,
            problem_type=Task.ONE_FROBENIUS_NORM,
            difficulty_level=DifficultyCategory.ONE_TOOL_CALL,
            entropy=EntropyConstraints(entropy),
            gen_constraints=_merge_gen_constraints({"square": True}, gen_constraints),
        ),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_one_matrix_rank_factory(
    registry: "FactoryRegistry", entropy: tuple[float, float] | float, gen_constraints: dict[str, Any] | None = None
) -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.ONE_RANK,
        create_sympy_factory(
            generator_class=MatrixRankGenerator,
            topic=Topic.LINEAR_ALGEBRA,
            problem_type=Task.ONE_RANK,
            difficulty_level=DifficultyCategory.ONE_TOOL_CALL,
            entropy=EntropyConstraints(entropy),
            gen_constraints=_merge_gen_constraints({"square": True}, gen_constraints),
        ),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_one_matrix_transpose_factory(
    registry: "FactoryRegistry", entropy: tuple[float, float] | float, gen_constraints: dict[str, Any] | None = None
) -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.ONE_TRANSPOSE,
        create_sympy_factory(
            generator_class=MatrixTransposeGenerator,
            topic=Topic.LINEAR_ALGEBRA,
            problem_type=Task.ONE_TRANSPOSE,
            difficulty_level=DifficultyCategory.ONE_TOOL_CALL,
            entropy=EntropyConstraints(entropy),
            gen_constraints=_merge_gen_constraints({"square": True}, gen_constraints),
        ),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_one_matrix_cofactor_factory(
    registry: "FactoryRegistry", entropy: tuple[float, float] | float, gen_constraints: dict[str, Any] | None = None
) -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.ONE_COFACTOR,
        create_sympy_factory(
            generator_class=MatrixCofactorGenerator,
            topic=Topic.LINEAR_ALGEBRA,
            problem_type=Task.ONE_COFACTOR,
            difficulty_level=DifficultyCategory.ONE_TOOL_CALL,
            entropy=EntropyConstraints(entropy),
            gen_constraints=_merge_gen_constraints({"square": True}, gen_constraints),
        ),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_one_trace_factory(
    registry: "FactoryRegistry", entropy: tuple[float, float] | float, gen_constraints: dict[str, Any] | None = None
) -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.ONE_TRACE,
        create_sympy_factory(
            generator_class=MatrixTraceGenerator,
            topic=Topic.LINEAR_ALGEBRA,
            problem_type=Task.ONE_TRACE,
            difficulty_level=DifficultyCategory.ONE_TOOL_CALL,
            entropy=EntropyConstraints(entropy),
            gen_constraints=_merge_gen_constraints({"square": True}, gen_constraints),
        ),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def _merge_gen_constraints(mandatory: dict[str, Any], optional: dict[str, Any] | None) -> GenerationConstraints:
    """Merge optional gen_constraints with mandatory ones, giving priority to mandatory."""
    if optional is None:
        return GenerationConstraints(**mandatory)

    merged = {**optional, **mandatory}  # mandatory overwrites optional
    return GenerationConstraints(**merged)


def register_two_transpose_determinant(
    registry: "FactoryRegistry",
    entropy: dict[Task, tuple[float, float] | float],
    gen_constraints: dict[Task, dict[str, Any]] | None = None,
) -> None:
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.TWO_TRANSPOSE_DETERMINANT,
        components=[
            TransposeWrapperComponent(
                name=Task.ONE_TRANSPOSE,
                constraints={"is_independent": True},
                gen_constraints=_merge_gen_constraints(
                    {"square": True}, gen_constraints.get(Task.ONE_TRANSPOSE) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_TRANSPOSE]),
            ),
            DeterminantWrapperComponent(
                name=Task.ONE_DETERMINANT,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 0},
                    "sources": {"input_matrix": "result"},
                },
                gen_constraints=_merge_gen_constraints(
                    {"square": True}, gen_constraints.get(Task.ONE_DETERMINANT) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_DETERMINANT]),
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.TWO_TOOL_CALLS,
    )


def register_two_cofactor_trace(
    registry: "FactoryRegistry",
    entropy: dict[Task, tuple[float, float] | float],
    gen_constraints: dict[Task, dict[str, Any]] | None = None,
) -> None:
    """Register cofactor + trace composition."""
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.TWO_COFACTOR_TRACE,
        components=[
            MatrixCofactorWrapperComponent(
                name=Task.ONE_COFACTOR,
                constraints={"is_independent": True},
                gen_constraints=_merge_gen_constraints(
                    {"square": True}, gen_constraints.get(Task.ONE_COFACTOR) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_COFACTOR]),
            ),
            MatrixTraceWrapperComponent(
                name=Task.ONE_TRACE,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 0},
                    "sources": {"input_matrix": "result"},
                },
                gen_constraints=_merge_gen_constraints(
                    {"square": True}, gen_constraints.get(Task.ONE_TRACE) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_TRACE]),
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.TWO_TOOL_CALLS,
    )


def register_two_cofactor_rank(
    registry: "FactoryRegistry",
    entropy: dict[Task, tuple[float, float] | float],
    gen_constraints: dict[Task, dict[str, Any]] | None = None,
) -> None:
    """Register cofactor + matrix_rank composition."""
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.TWO_COFACTOR_RANK,
        components=[
            MatrixCofactorWrapperComponent(
                name=Task.ONE_COFACTOR,
                constraints={"is_independent": True},
                gen_constraints=_merge_gen_constraints(
                    {"square": True}, gen_constraints.get(Task.ONE_COFACTOR) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_COFACTOR]),
            ),
            RankWrapperComponent(
                name=Task.ONE_RANK,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 0},
                    "sources": {"input_matrix": "result"},
                },
                gen_constraints=_merge_gen_constraints(
                    {"square": True}, gen_constraints.get(Task.ONE_RANK) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_RANK]),
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.TWO_TOOL_CALLS,
    )


def register_two_transpose_frobenius(
    registry: "FactoryRegistry",
    entropy: dict[Task, tuple[float, float] | float],
    gen_constraints: dict[Task, dict[str, Any]] | None = None,
) -> None:
    """Register transpose + frobenius_norm composition."""
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.TWO_TRANSPOSE_FROBENIUS,
        components=[
            TransposeWrapperComponent(
                name=Task.ONE_TRANSPOSE,
                constraints={"is_independent": True},
                gen_constraints=_merge_gen_constraints(
                    {"square": True}, gen_constraints.get(Task.ONE_TRANSPOSE) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_TRANSPOSE]),
            ),
            FrobeniusNormWrapperComponent(
                name=Task.ONE_FROBENIUS_NORM,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 0},
                    "sources": {"input_matrix": "result"},
                },
                gen_constraints=_merge_gen_constraints(
                    {"square": True}, gen_constraints.get(Task.ONE_FROBENIUS_NORM) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_FROBENIUS_NORM]),
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.TWO_TOOL_CALLS,
    )


def register_three_transpose_cofactor_rank(
    registry: "FactoryRegistry",
    entropy: dict[Task, tuple[float, float] | float],
    gen_constraints: dict[Task, dict[str, Any]] | None = None,
) -> None:
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.THREE_TRANSPOSE_COFACTOR_RANK,
        components=[
            TransposeWrapperComponent(
                name=Task.ONE_TRANSPOSE,
                constraints={"is_independent": True},
                gen_constraints=_merge_gen_constraints(
                    {"square": True}, gen_constraints.get(Task.ONE_TRANSPOSE) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_TRANSPOSE]),
            ),
            MatrixCofactorWrapperComponent(
                name=Task.ONE_COFACTOR,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 0},
                    "sources": {"input_matrix": "result"},
                },
                gen_constraints=_merge_gen_constraints(
                    {"square": True}, gen_constraints.get(Task.ONE_COFACTOR) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_COFACTOR]),
            ),
            RankWrapperComponent(
                name=Task.ONE_RANK,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 1},
                    "sources": {"input_matrix": "result"},
                },
                gen_constraints=_merge_gen_constraints(
                    {"square": True}, gen_constraints.get(Task.ONE_RANK) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_RANK]),
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.THREE_TOOL_CALLS,
    )


def register_three_cofactor_transpose_trace(
    registry: "FactoryRegistry",
    entropy: dict[Task, tuple[float, float] | float],
    gen_constraints: dict[Task, dict[str, Any]] | None = None,
) -> None:
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.THREE_COFACTOR_TRANSPOSE_TRACE,
        components=[
            MatrixCofactorWrapperComponent(
                name=Task.ONE_COFACTOR,
                constraints={"is_independent": True},
                gen_constraints=_merge_gen_constraints(
                    {"square": True}, gen_constraints.get(Task.ONE_COFACTOR) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_COFACTOR]),
            ),
            TransposeWrapperComponent(
                name=Task.ONE_TRANSPOSE,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 0},
                    "sources": {"input_matrix": "result"},
                },
                gen_constraints=_merge_gen_constraints(
                    {"square": True}, gen_constraints.get(Task.ONE_TRANSPOSE) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints((0, 0)),
            ),
            MatrixTraceWrapperComponent(
                name=Task.ONE_TRACE,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 1},
                    "sources": {"input_matrix": "result"},
                },
                gen_constraints=_merge_gen_constraints(
                    {"square": True}, gen_constraints.get(Task.ONE_TRACE) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints((0, 0)),
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.THREE_TOOL_CALLS,
    )


def register_three_transpose_cofactor_frobenius(
    registry: "FactoryRegistry",
    entropy: dict[Task, tuple[float, float] | float],
    gen_constraints: dict[Task, dict[str, Any]] | None = None,
) -> None:
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.THREE_TRANSPOSE_COFACTOR_FROBENIUS,
        components=[
            TransposeWrapperComponent(
                name=Task.ONE_TRANSPOSE,
                constraints={"is_independent": True},
                gen_constraints=_merge_gen_constraints(
                    {"square": True}, gen_constraints.get(Task.ONE_TRANSPOSE) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_TRANSPOSE]),
            ),
            MatrixCofactorWrapperComponent(
                name=Task.ONE_COFACTOR,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 0},
                    "sources": {"input_matrix": "result"},
                },
                gen_constraints=_merge_gen_constraints(
                    {"square": True}, gen_constraints.get(Task.ONE_COFACTOR) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_COFACTOR]),
            ),
            FrobeniusNormWrapperComponent(
                name=Task.ONE_FROBENIUS_NORM,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 1},
                    "sources": {"input_matrix": "result"},
                },
                gen_constraints=_merge_gen_constraints(
                    {"square": True}, gen_constraints.get(Task.ONE_FROBENIUS_NORM) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_FROBENIUS_NORM]),
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.THREE_TOOL_CALLS,
    )


class FactoryRegistry:
    """Registry for managing different question factories."""

    def __init__(self) -> None:
        self._factories: dict[Topic, dict[Task, Callable[[], Question]]] = {}
        self._factory_difficulties: dict[Topic, dict[Task, DifficultyCategory]] = {}
        self._composite_components: dict[Topic, dict[Task, list[tuple[Task, bool]]]] = {}

    def register_factory(
        self,
        topic: Topic,
        problem_type: Task,
        factory: Callable[[], Question],
        difficulty: DifficultyCategory | None = None,
    ) -> None:
        """Register a factory function."""
        if topic not in self._factories:
            self._factories[topic] = {}
            self._factory_difficulties[topic] = {}
        self._factories[topic][problem_type] = factory
        if difficulty is not None:
            self._factory_difficulties[topic][problem_type] = difficulty

    def get_factory(self, topic: Topic, problem_type: Task) -> Callable[[], Question]:
        """Get a specific factory by topic and problem type."""
        if topic not in self._factories:
            raise ValueError(f"Unknown topic: {topic}")
        if problem_type not in self._factories[topic]:
            raise ValueError(f"Unknown problem type: {problem_type}")
        return self._factories[topic][problem_type]

    def get_random_factory(self, topic: Topic) -> Callable[[], Question]:
        """Get a random factory from the specified topic."""
        if topic not in self._factories:
            raise ValueError(f"Unknown topic: {topic}")
        problem_types = list(self._factories[topic].keys())
        random_type = random.choice(problem_types)
        return self._factories[topic][random_type]

    def list_topics(self) -> list[Topic]:
        """List all available topics."""
        return list(self._factories.keys())

    def list_problem_types(self, topic: Topic) -> list[Task]:
        """List all problem types for a given topic."""
        if topic not in self._factories:
            raise ValueError(f"Unknown topic: {topic}")
        return list(self._factories[topic].keys())

    def get_factories_by_difficulty(
        self, topic: Topic, difficulty: DifficultyCategory
    ) -> list[Callable[[], Question]]:
        """Return all factories for the given topic and difficulty category."""
        if topic not in self._factories:
            raise ValueError(f"Unknown topic: {topic}")
        if topic not in self._factory_difficulties:
            return []
        factories: list[Callable[[], Question]] = []
        for task, task_difficulty in self._factory_difficulties[topic].items():
            if task_difficulty == difficulty:
                factories.append(self._factories[topic][task])
        return factories

    def register_composite_factory(
        self,
        topic: Topic,
        problem_type: Task,
        components: list[ProblemComponent],
        composition_strategy: CompositionStrategy,
        difficulty_level: DifficultyCategory,
    ) -> None:
        """Register a composite factory"""
        factory = create_composite_factory(
            components=components,
            composition_strategy=composition_strategy,
            difficulty_level=difficulty_level,
            problem_type=problem_type,
            topic=topic,
        )

        # Store difficulty for the composite factory
        self.register_factory(topic, problem_type, factory, difficulty=difficulty_level)

        # Store composite component metadata used in tests
        if topic not in self._composite_components:
            self._composite_components[topic] = {}
        self._composite_components[topic][problem_type] = [(c.name, c.is_independent) for c in components]

    def get_composite_components(self, topic: Topic, problem_type: Task) -> list[tuple[Task, bool]]:
        """Return (component task, is_independent) for a registered composite problem, or empty list."""
        return self._composite_components.get(topic, {}).get(problem_type, [])


def create_default_registry() -> FactoryRegistry:
    """Create and populate the default factory registry."""
    registry = FactoryRegistry()

    # ===================
    # 1-STEP COMPOSITIONS
    # ===================
    entropy = (1.2, 1.5)
    register_one_determinant_factory(registry, entropy=entropy)
    register_one_frobenius_norm_factory(registry, entropy=entropy)
    register_one_matrix_rank_factory(registry, entropy=entropy)
    register_one_matrix_transpose_factory(registry, entropy=entropy)
    register_one_matrix_cofactor_factory(registry, entropy=entropy)
    register_one_trace_factory(registry, entropy=entropy)

    # ===================
    # 2-STEP COMPOSITIONS
    # ===================
    entropy_ranges: dict[Task, dict[Task, tuple[float, float] | float]] = {
        Task.TWO_TRANSPOSE_DETERMINANT: {
            Task.ONE_TRANSPOSE: entropy,
            Task.ONE_DETERMINANT: (0.0, 0.0),
        },
        Task.TWO_COFACTOR_TRACE: {
            Task.ONE_COFACTOR: entropy,
            Task.ONE_TRACE: (0.0, 0.0),
        },
        Task.TWO_COFACTOR_RANK: {
            Task.ONE_COFACTOR: entropy,
            Task.ONE_RANK: (0.0, 0.0),
        },
        Task.TWO_TRANSPOSE_FROBENIUS: {
            Task.ONE_TRANSPOSE: entropy,
            Task.ONE_FROBENIUS_NORM: (0.0, 0.0),
        },
    }

    register_two_transpose_determinant(registry, entropy=entropy_ranges[Task.TWO_TRANSPOSE_DETERMINANT])
    register_two_cofactor_trace(registry, entropy=entropy_ranges[Task.TWO_COFACTOR_TRACE])
    register_two_cofactor_rank(registry, entropy=entropy_ranges[Task.TWO_COFACTOR_RANK])
    register_two_transpose_frobenius(registry, entropy=entropy_ranges[Task.TWO_TRANSPOSE_FROBENIUS])

    # ===================
    # 3-STEP COMPOSITIONS
    # ===================
    entropy_ranges = {
        Task.THREE_TRANSPOSE_COFACTOR_RANK: {
            Task.ONE_TRANSPOSE: (0.3, 0.6),
            Task.ONE_COFACTOR: (0.0, 0.0),
            Task.ONE_RANK: (0.0, 0.0),
        },
        Task.THREE_COFACTOR_TRANSPOSE_TRACE: {
            Task.ONE_COFACTOR: (0.3, 0.6),
            Task.ONE_TRANSPOSE: (0.0, 0.0),
            Task.ONE_TRACE: (0.0, 0.0),
        },
        Task.THREE_TRANSPOSE_COFACTOR_FROBENIUS: {
            Task.ONE_TRANSPOSE: (0.3, 0.6),
            Task.ONE_COFACTOR: (0.0, 0.0),
            Task.ONE_FROBENIUS_NORM: (0.0, 0.0),
        },
    }
    register_three_transpose_cofactor_rank(
        registry,
        entropy=entropy_ranges[Task.THREE_TRANSPOSE_COFACTOR_RANK],
    )
    register_three_cofactor_transpose_trace(registry, entropy=entropy_ranges[Task.THREE_COFACTOR_TRANSPOSE_TRACE])
    register_three_transpose_cofactor_frobenius(
        registry, entropy=entropy_ranges[Task.THREE_TRANSPOSE_COFACTOR_FROBENIUS]
    )

    return registry


def get_single_step_functions() -> dict[
    Task, Callable[[FactoryRegistry, tuple[float, float] | float, dict[str, Any]], None]
]:
    return {
        Task.ONE_DETERMINANT: register_one_determinant_factory,
        Task.ONE_FROBENIUS_NORM: register_one_frobenius_norm_factory,
        Task.ONE_RANK: register_one_matrix_rank_factory,
        Task.ONE_TRANSPOSE: register_one_matrix_transpose_factory,
        Task.ONE_COFACTOR: register_one_matrix_cofactor_factory,
        Task.ONE_TRACE: register_one_trace_factory,
    }


def get_multi_step_functions() -> dict[
    Task, Callable[[FactoryRegistry, dict[Task, tuple[float, float] | float], dict[Task, dict[str, Any]]], None]
]:
    return {
        Task.TWO_TRANSPOSE_DETERMINANT: register_two_transpose_determinant,
        Task.TWO_COFACTOR_TRACE: register_two_cofactor_trace,
        Task.TWO_COFACTOR_RANK: register_two_cofactor_rank,
        Task.TWO_TRANSPOSE_FROBENIUS: register_two_transpose_frobenius,
        Task.THREE_TRANSPOSE_COFACTOR_RANK: register_three_transpose_cofactor_rank,
        Task.THREE_COFACTOR_TRANSPOSE_TRACE: register_three_cofactor_transpose_trace,
        Task.THREE_TRANSPOSE_COFACTOR_FROBENIUS: register_three_transpose_cofactor_frobenius,
    }


def register_problem_type(
    registry: FactoryRegistry,
    problem_type: Task,
    entropy_ranges: dict[Task, float | tuple[float, float]],
    jitter: float,
    min_value_abs: float,
) -> None:
    # Get the function mappings
    single_step_functions = get_single_step_functions()
    multi_step_functions = get_multi_step_functions()

    if problem_type in single_step_functions:
        # Single-step factories take EntropyConstraints
        # Extract the entropy value (single-step has one component)
        raw_entropy_value = next(iter(entropy_ranges.values()))
        if isinstance(raw_entropy_value, tuple):
            entropy_value = raw_entropy_value
        else:
            # Convert fixed value to a small range below the fixed value
            lo = max(0.0, float(raw_entropy_value) - jitter)
            hi = float(raw_entropy_value)
            entropy_value = (lo, hi)

        single_step_functions[problem_type](registry, entropy_value, {"min_element_abs": min_value_abs})
    elif problem_type in multi_step_functions:
        # Multi-step factories take entropy_ranges directly
        assert isinstance(entropy_ranges, dict)
        # Convert fixed values to small ranges per component
        converted: dict[Task, tuple[float, float] | float] = {}
        for comp, val in entropy_ranges.items():
            if isinstance(val, tuple):
                converted[comp] = val
            else:
                lo = max(0.0, float(val) - jitter)
                hi = float(val)
                converted[comp] = (lo, hi)

        multi_step_functions[problem_type](
            registry,
            converted,
            {comp: {"min_element_abs": min_value_abs} for comp in converted},
        )
    else:
        raise ValueError(f"Unknown problem type: {problem_type}. Not found in single-step or multi-step functions.")


def create_optimized_registry(
    config_path: str,
) -> FactoryRegistry:
    """
    Create a registry with entropy values optimized from analysis results.
    Uses metadata from the JSON file to automatically reconstruct the registry configuration.
    """
    config = load_entropy_settings(path=config_path)
    print_entropy_settings(config)
    registry = FactoryRegistry()

    # Register factories with optimized entropy values
    for problem_type_str, settings in config.items():
        metadata = settings["metadata"]
        combination = settings["combination"]

        entropy_jitter = metadata["entropy_jitter"]
        min_element_abs = metadata["min_element_abs"]
        task = Task[metadata["task_enum"]]

        if metadata["is_single_step"]:
            # Single-step: use the task itself as the key
            entropy_ranges = {task: combination[0]}
        else:
            # Multi-step: map components to combination values
            components_names = metadata["components"]
            if len(combination) != len(components_names):
                raise ValueError(
                    f"Mismatch in component count for {problem_type_str}: expected {len(components_names)}, got {len(combination)}"
                )

            entropy_ranges = {}
            for component_name, entropy_value in zip(components_names, combination, strict=True):
                component_task = Task[component_name]
                entropy_ranges[component_task] = entropy_value

        register_problem_type(registry, task, entropy_ranges, entropy_jitter, min_element_abs)

    return registry
