from collections.abc import Callable

from linalg_zero.generator.models import DifficultyCategory, Question, Topic
from linalg_zero.generator.registry import FactoryRegistry, create_default_registry
from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)


class QuestionGenerator:
    """
    Question generator using Instance Attribute Factory pattern. Here factories are passed as
    callables (i.e. functions, lambda expressions, methods, classes or partial functions).
    """

    def __init__(
        self, question_factory: Callable[[], Question], validator_factory: Callable[[Question], bool] | None = None
    ) -> None:
        """
        Initialize with factory callables.

        Args:
            question_factory: Any callable that returns a Question
            validator_factory: Optional callable to validate questions
        """
        self.question_factory = question_factory
        self.validator_factory = validator_factory or self._default_validator

    def generate(self) -> Question:
        """Generate a single question using the configured factories."""
        question = self.question_factory()

        # Set validation status using the configured validator
        question.is_valid = self.validator_factory(question)

        return question

    @staticmethod
    def _default_validator(question: Question) -> bool:
        """Default validator - checks basic requirements."""
        return len(question.question) > 0 and len(question.answer) > 0


class DatasetGenerator:
    """
    Dataset generator using Instance Attribute Factory pattern.

    Following python-patterns.guide recommendations - instead of a function
    with many parameters, use a class that accepts configuration in __init__.
    """

    def __init__(
        self,
        topic: Topic = Topic.LINEAR_ALGEBRA,
        validator_factory: Callable[[Question], bool] | None = None,
        max_attempts: int = 999999999,
        registry: FactoryRegistry | None = None,
    ):
        """Initialize with generation configuration."""
        self.topic = topic
        self.validator_factory = validator_factory or QuestionGenerator._default_validator
        self.max_attempts = max_attempts
        self.registry = registry or create_default_registry()

    def generate_dataset(self, num_questions: int) -> list[Question]:
        """Generate a dataset with the configured parameters (randomly across all factories)."""
        generator = QuestionGenerator(
            question_factory=lambda: self.registry.get_random_factory(self.topic)(),
            validator_factory=self.validator_factory,
        )

        questions: list[Question] = []
        attempts = 0

        while len(questions) < num_questions and attempts < self.max_attempts:
            question = generator.generate()
            if question.is_valid:
                questions.append(question)
            attempts += 1

        if len(questions) < num_questions:
            logger.warning(
                "Only generated %d/%d valid questions after %d attempts",
                len(questions),
                num_questions,
                self.max_attempts,
            )

        return questions

    def generate_exact_per_factory(self, difficulty: "DifficultyCategory", num_per_factory: int) -> list[Question]:
        """Generate exactly num_per_factory valid questions per registered factory of a category.

        This guarantees: total == (number_of_factories_in_category * num_per_factory).
        """

        if not isinstance(difficulty, DifficultyCategory):
            raise TypeError("difficulty must be a DifficultyCategory")

        factories = self.registry.get_factories_by_difficulty(self.topic, difficulty)
        if not factories:
            return []

        all_questions: list[Question] = []
        for factory in factories:
            per_factory_questions: list[Question] = []
            attempts = 0
            qg = QuestionGenerator(question_factory=factory, validator_factory=self.validator_factory)
            # Tight loop to ensure exact count (subject to max_attempts)
            while len(per_factory_questions) < num_per_factory and attempts < self.max_attempts:
                q = qg.generate()
                if q.is_valid:
                    per_factory_questions.append(q)
                attempts += 1
            if len(per_factory_questions) < num_per_factory:
                logger.warning(
                    "Factory produced %d/%d valid questions (difficulty=%s)",
                    len(per_factory_questions),
                    num_per_factory,
                    difficulty,
                )
            all_questions.extend(per_factory_questions)

        return all_questions

    def generate_exact_for_categories(self, requests: dict["DifficultyCategory", int]) -> list[Question]:
        """Generate exactly N per factory for each requested category.

        Example: {ONE_TOOL_CALL: 3000} will produce 3000 per registered factory in that category.
        """

        total: list[Question] = []
        for difficulty, num_per_factory in requests.items():
            if not isinstance(difficulty, DifficultyCategory):
                raise TypeError("All keys in requests must be DifficultyCategory")
            total.extend(self.generate_exact_per_factory(difficulty, num_per_factory))
        return total
