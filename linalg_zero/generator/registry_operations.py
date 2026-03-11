from typing import Any

from linalg_zero.generator.composition.components import (
    DeterminantWrapperComponent,
    FrobeniusNormWrapperComponent,
    LinearSystemSolverWrapperComponent,
    MatrixCofactorWrapperComponent,
    MatrixMatrixMultiplicationWrapperComponent,
    MatrixTraceWrapperComponent,
    MatrixVectorMultiplicationWrapperComponent,
    RankWrapperComponent,
    TransposeWrapperComponent,
)
from linalg_zero.generator.composition.composition import (
    SequentialComposition,
)
from linalg_zero.generator.entropy_control import EntropyConstraints
from linalg_zero.generator.generator_factories import create_sympy_factory
from linalg_zero.generator.models import DifficultyCategory, Task, Topic
from linalg_zero.generator.registry import FactoryRegistry, _merge_gen_constraints
from linalg_zero.generator.sympy.generators.linear_system_generator import LinearSystemGenerator
from linalg_zero.generator.sympy.generators.matrix_matrix_generator import (
    MatrixMatrixMultiplicationGenerator,
)


def register_one_matrix_matrix_multiplication_factory(
    registry: "FactoryRegistry", entropy: tuple[float, float] | float, gen_constraints: dict[str, Any] | None = None
) -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.ONE_MATRIX_MATRIX_MULTIPLICATION,
        create_sympy_factory(
            generator_class=MatrixMatrixMultiplicationGenerator,
            topic=Topic.LINEAR_ALGEBRA,
            problem_type=Task.ONE_MATRIX_MATRIX_MULTIPLICATION,
            difficulty_level=DifficultyCategory.ONE_TOOL_CALL,
            entropy=EntropyConstraints(entropy),
            gen_constraints=_merge_gen_constraints({"square": True}, gen_constraints),
        ),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_one_linear_system_solver_factory(
    registry: "FactoryRegistry", entropy: tuple[float, float] | float, gen_constraints: dict[str, Any] | None = None
) -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.ONE_LINEAR_SYSTEM_SOLVER,
        create_sympy_factory(
            generator_class=LinearSystemGenerator,
            topic=Topic.LINEAR_ALGEBRA,
            problem_type=Task.ONE_LINEAR_SYSTEM_SOLVER,
            difficulty_level=DifficultyCategory.ONE_TOOL_CALL,
            entropy=EntropyConstraints(entropy),
            gen_constraints=_merge_gen_constraints({"square": True}, gen_constraints),
        ),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_two_cofactor_frobenius(
    registry: "FactoryRegistry",
    entropy: dict[Task, tuple[float, float] | float],
    gen_constraints: dict[Task, dict[str, Any]] | None = None,
) -> None:
    """Register cofactor + frobenius_norm composition."""
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.TWO_COFACTOR_FROBENIUS,
        components=[
            MatrixCofactorWrapperComponent(
                name=Task.ONE_COFACTOR,
                constraints={"is_independent": True},
                gen_constraints=_merge_gen_constraints(
                    {"square": True}, gen_constraints.get(Task.ONE_COFACTOR) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_COFACTOR]),
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


def register_three_matrixvector_system_frobenius(
    registry: "FactoryRegistry",
    entropy: dict[Task, tuple[float, float] | float],
    gen_constraints: dict[Task, dict[str, Any]] | None = None,
) -> None:
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.THREE_MATRIXVECTOR_SYSTEM_FROBENIUS,
        components=[
            MatrixVectorMultiplicationWrapperComponent(
                name=Task.ONE_MATRIX_VECTOR_MULTIPLICATION,
                constraints={"is_independent": True},
                gen_constraints=_merge_gen_constraints(
                    {"rows": 2, "cols": 2},
                    gen_constraints.get(Task.ONE_MATRIX_VECTOR_MULTIPLICATION) if gen_constraints else None,
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_MATRIX_VECTOR_MULTIPLICATION]),
            ),
            LinearSystemSolverWrapperComponent(
                name=Task.ONE_LINEAR_SYSTEM_SOLVER,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_vector_b": 0},
                    "sources": {"input_vector_b": "result"},
                },
                gen_constraints=_merge_gen_constraints(
                    {"square": True, "invertible": True},
                    gen_constraints.get(Task.ONE_LINEAR_SYSTEM_SOLVER) if gen_constraints else None,
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_LINEAR_SYSTEM_SOLVER]),
            ),
            FrobeniusNormWrapperComponent(
                name=Task.ONE_FROBENIUS_NORM,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 1},
                    "sources": {"input_matrix": "matrix_A"},
                },
                gen_constraints=_merge_gen_constraints(
                    {}, gen_constraints.get(Task.ONE_FROBENIUS_NORM) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_FROBENIUS_NORM]),
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.THREE_TOOL_CALLS,
    )


def register_three_system_matrixmult_frobenius(
    registry: "FactoryRegistry",
    entropy: dict[Task, tuple[float, float] | float],
    gen_constraints: dict[Task, dict[str, Any]] | None = None,
) -> None:
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.THREE_SYSTEM_MATRIXMULT_FROBENIUS,
        components=[
            LinearSystemSolverWrapperComponent(
                name=Task.ONE_LINEAR_SYSTEM_SOLVER,
                constraints={"is_independent": True},
                gen_constraints=_merge_gen_constraints(
                    {"square": True, "invertible": True},
                    gen_constraints.get(Task.ONE_LINEAR_SYSTEM_SOLVER) if gen_constraints else None,
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_LINEAR_SYSTEM_SOLVER]),
            ),
            MatrixMatrixMultiplicationWrapperComponent(
                name=Task.ONE_MATRIX_MATRIX_MULTIPLICATION,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix_A": 0},
                    "sources": {"input_matrix_A": "matrix_A"},
                },
                gen_constraints=_merge_gen_constraints(
                    {}, gen_constraints.get(Task.ONE_MATRIX_MATRIX_MULTIPLICATION) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_MATRIX_MATRIX_MULTIPLICATION]),
            ),
            FrobeniusNormWrapperComponent(
                name=Task.ONE_FROBENIUS_NORM,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 1},
                    "sources": {"input_matrix": "result"},
                },
                gen_constraints=_merge_gen_constraints(
                    {}, gen_constraints.get(Task.ONE_FROBENIUS_NORM) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_FROBENIUS_NORM]),
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.THREE_TOOL_CALLS,
    )


def register_three_cofactor_matrixmult_rank(
    registry: "FactoryRegistry",
    entropy: dict[Task, tuple[float, float] | float],
    gen_constraints: dict[Task, dict[str, Any]] | None = None,
) -> None:
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.THREE_COFACTOR_MATRIXMULT_RANK,
        components=[
            MatrixCofactorWrapperComponent(
                name=Task.ONE_COFACTOR,
                constraints={"is_independent": True},
                gen_constraints=_merge_gen_constraints(
                    {"square": True}, gen_constraints.get(Task.ONE_COFACTOR) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_COFACTOR]),
            ),
            MatrixMatrixMultiplicationWrapperComponent(
                name=Task.ONE_MATRIX_MATRIX_MULTIPLICATION,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix_A": 0, "input_matrix_B": 0},
                    "sources": {"input_matrix_A": "matrix", "input_matrix_B": "result"},
                },
                gen_constraints=_merge_gen_constraints(
                    {}, gen_constraints.get(Task.ONE_MATRIX_MATRIX_MULTIPLICATION) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_MATRIX_MATRIX_MULTIPLICATION]),
            ),
            RankWrapperComponent(
                name=Task.ONE_RANK,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 1},
                    "sources": {"input_matrix": "result"},
                },
                gen_constraints=_merge_gen_constraints(
                    {}, gen_constraints.get(Task.ONE_RANK) if gen_constraints else None
                ),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_RANK]),
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.THREE_TOOL_CALLS,
    )


def register_three_transpose_determinant_trace(
    registry: "FactoryRegistry",
    entropy: dict[Task, tuple[float, float] | float],
    gen_constraints: dict[Task, dict[str, Any]] | None = None,
) -> None:
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.THREE_TRANSPOSE_DETERMINANT_TRACE,
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
        difficulty_level=DifficultyCategory.THREE_TOOL_CALLS,
    )
