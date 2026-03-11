import pytest

from linalg_zero.generator.composition.components import (
    DeterminantWrapperComponent,
    FrobeniusNormWrapperComponent,
    MatrixTraceWrapperComponent,
    RankWrapperComponent,
    TransposeWrapperComponent,
)
from linalg_zero.generator.composition.composition import (
    CompositeProblem,
    SequentialComposition,
)
from linalg_zero.generator.entropy_control import EntropyConstraints
from linalg_zero.generator.models import DifficultyCategory, Task, Topic
from linalg_zero.generator.sympy.generators.determinant_generator import (
    DeterminantGenerator,
    DeterminantGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.frobenius_norm_generator import (
    FrobeniusNormGenerator,
    FrobeniusNormGeneratorDependent,
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
from linalg_zero.generator.sympy.template_engine import TemplateEngine


def make_composite(
    components: list, difficulty: DifficultyCategory = DifficultyCategory.TWO_TOOL_CALLS
) -> CompositeProblem:
    return CompositeProblem(
        components=components,
        composition_strategy=SequentialComposition(),
        difficulty_level=difficulty,
        problem_type=Task.SEQUENTIAL_PROBLEM,
        topic=Topic.LINEAR_ALGEBRA,
        template_engine=TemplateEngine(),
    )


class TestWrapperComponentGeneratorSelectionComprehensive:
    """Comprehensive tests across all wrapper components to ensure consistency."""

    @pytest.mark.parametrize(
        "wrapper_class,task,independent_generator",
        [
            (DeterminantWrapperComponent, Task.ONE_DETERMINANT, DeterminantGenerator),
            (FrobeniusNormWrapperComponent, Task.ONE_FROBENIUS_NORM, FrobeniusNormGenerator),
            (RankWrapperComponent, Task.ONE_RANK, MatrixRankGenerator),
            (MatrixTraceWrapperComponent, Task.ONE_TRACE, MatrixTraceGenerator),
            (TransposeWrapperComponent, Task.ONE_TRANSPOSE, MatrixTransposeGenerator),
        ],
    )
    def test_all_wrappers_independent_case(self, wrapper_class, task, independent_generator):
        """Test that all wrapper components correctly select independent generator when is_independent=True."""
        component = wrapper_class(
            name=task, constraints={"is_independent": True}, entropy_constraints=EntropyConstraints(entropy=0.1)
        )

        assert component.generator_class is independent_generator
        assert component.is_independent is True
        assert component.name == task

    @pytest.mark.parametrize(
        "wrapper_class,task,dependent_generator",
        [
            (DeterminantWrapperComponent, Task.ONE_DETERMINANT, DeterminantGeneratorDependent),
            (FrobeniusNormWrapperComponent, Task.ONE_FROBENIUS_NORM, FrobeniusNormGeneratorDependent),
            (RankWrapperComponent, Task.ONE_RANK, MatrixRankGeneratorDependent),
            (MatrixTraceWrapperComponent, Task.ONE_TRACE, MatrixTraceGeneratorDependent),
            (TransposeWrapperComponent, Task.ONE_TRANSPOSE, MatrixTransposeGeneratorDependent),
        ],
    )
    def test_all_wrappers_dependent_case(self, wrapper_class, task, dependent_generator):
        """Test that all wrapper components correctly select dependent generator when is_independent=False."""
        component = wrapper_class(
            name=task,
            constraints={"is_independent": False, "input_indices": {"input_vector_b": 0}},
            entropy_constraints=EntropyConstraints(entropy=0.1),
        )

        assert component.generator_class is dependent_generator
        assert component.is_independent is False
        assert component.name == task
