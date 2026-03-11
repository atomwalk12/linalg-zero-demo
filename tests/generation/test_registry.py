import pytest

from linalg_zero.generator.models import DifficultyCategory, Question, Task
from linalg_zero.generator.registry import FactoryRegistry


def simple_test_factory() -> Question:
    """Simple factory for testing."""
    return Question(
        question="Test",
        answer="42",
        topic="test",
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
        problem_type=Task.ONE_DETERMINANT,
    )


def another_test_factory() -> Question:  # pragma: no cover
    """Another simple factory for testing."""
    return Question(
        question="Another test",
        answer="24",
        topic="test",
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
        problem_type=Task.ONE_DETERMINANT,
    )


def test_factory_registry_registration() -> None:
    """Test basic factory registration and retrieval."""
    registry = FactoryRegistry()

    # Register a factory
    registry.register_factory("test_topic", "test_problem", simple_test_factory)

    # Retrieve and test
    factory = registry.get_factory("test_topic", "test_problem")
    question = factory()

    assert question.question == "Test"
    assert question.answer == "42"


def test_factory_registry_list_topics() -> None:
    """Test listing topics."""
    registry = FactoryRegistry()

    registry.register_factory("math", "addition", simple_test_factory)
    registry.register_factory("science", "physics", another_test_factory)

    topics = registry.list_topics()
    assert "math" in topics
    assert "science" in topics
    assert len(topics) == 2


def test_factory_registry_list_problem_types() -> None:
    """Test listing problem types for a topic."""
    registry = FactoryRegistry()

    registry.register_factory("math", "addition", simple_test_factory)
    registry.register_factory("math", "subtraction", another_test_factory)

    problem_types = registry.list_problem_types("math")
    assert "addition" in problem_types
    assert "subtraction" in problem_types
    assert len(problem_types) == 2


def test_factory_registry_unknown_topic() -> None:
    """Test error handling for unknown topic."""
    registry = FactoryRegistry()

    with pytest.raises(ValueError, match="Unknown topic"):
        registry.get_factory("non-existent", "problem")


def test_factory_registry_unknown_problem_type() -> None:
    """Test error handling for unknown problem type."""
    registry = FactoryRegistry()
    registry.register_factory("math", "addition", simple_test_factory)

    with pytest.raises(ValueError, match="Unknown problem type"):
        registry.get_factory("math", "non-existent")
