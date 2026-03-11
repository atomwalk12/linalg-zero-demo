import json
import os
from typing import TYPE_CHECKING, Any

from distilabel.steps.tasks.base import GeneratorTask
from pydantic import Field
from tqdm import tqdm

from linalg_zero.distillation.components.multi_turn_generation_base import MultiTurnWithToolUseBase

if TYPE_CHECKING:
    from typing import Any

    from distilabel.typing import GeneratorStepOutput, StepColumns


def update_progress(
    pbar: tqdm,
    batch_results: list[dict[str, Any]],
    batch_size: int,
    successful_count: int,
    processed: int,
    dataset_size: int,
    target_successes: int | None,
) -> int:
    """Update progress bar with batch results and return updated success count."""
    batch_successes = sum(1 for conv in batch_results if conv.get("is_correct", False))
    successful_count += batch_successes

    # Update progress based on tracking mode
    if target_successes is not None:
        pbar.n = successful_count
        # Force a redraw when manually setting position
        pbar.refresh()
    else:
        pbar.update(batch_size)

    # Update status display
    pbar.set_postfix_str(f"✓ {successful_count}/{processed} ({successful_count}/{dataset_size} total)")
    return successful_count


class MultiTurnWithToolUseGenerator(GeneratorTask, MultiTurnWithToolUseBase):
    """Simplified multi-turn generator that combines planning, execution, and summarization."""

    dataset: list[dict[str, Any]] = Field(description="Linear algebra problems to process")

    def model_post_init(self, __context: "Any") -> None:
        # Ensure base initialization is executed
        super().model_post_init(__context)
        # Do not include this runtime knob in the step signature (cache key)
        # so it can be tuned between runs without invalidating cached outputs.
        self.exclude_from_signature.add("min_successful_completions")

    @property
    def outputs(self) -> "StepColumns":
        """Define what data this task produces for downstream steps."""
        return ["messages", "model_name", "final_answer", "is_correct"]

    def format_output(
        self,
        output: str | None,
        input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Does nothing."""
        return {}

    def _get_progress_path(self) -> str:
        """Get path to progress file."""
        return os.path.join(".", "distillation-cache", "progress", f"{self.name}_success_count.json")

    def _load_successful_count(self) -> int:
        """Load successful count from progress file, returns 0 if not found."""
        try:
            with open(self._get_progress_path(), encoding="utf-8") as f:
                return int(json.load(f).get("count", 0))
        except Exception:
            return 0

    def _save_successful_count(self, count: int) -> None:
        """Save successful count to progress file."""
        try:
            progress_path = self._get_progress_path()
            os.makedirs(os.path.dirname(progress_path), exist_ok=True)
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump({"count": int(count)}, f)
        except Exception:
            self._logger.warning("Failed to save successful count", exc_info=True)

    def process(self, offset: int = 0) -> "GeneratorStepOutput":
        """Generate multi-turn conversations from the source dataset.

        Args:
            offset: Starting index for generation (for resumable generation)

        Yields:
            Tuple of (batch_data, is_last_batch)
        """
        generated = offset
        dataset_size = len(self.dataset)
        track_by_success = self.min_successful_completions != -1

        # Resume successful count from previous run if needed
        successful_count = self._load_successful_count() if (track_by_success and offset > 0) else 0

        # Setup progress bar
        total = (
            dataset_size
            if self.min_successful_completions == -1
            else min(self.min_successful_completions, dataset_size)
        )
        pbar = tqdm(
            initial=offset,
            total=total,
            desc="Generation",
            disable=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        while generated < dataset_size:
            batch_size = getattr(self, "batch_size", 8)
            rows_to_generate = min(batch_size, dataset_size - generated)
            batch_problems = self.dataset[generated : generated + rows_to_generate]
            batch_conversations = self._generate_with_pre_query_template(batch_problems)

            generated += rows_to_generate
            successful_count = update_progress(
                pbar,
                batch_conversations,
                rows_to_generate,
                successful_count,
                generated,
                dataset_size,
                None if self.min_successful_completions == -1 else self.min_successful_completions,
            )
            # Persist success counter so a crash/restart resumes accurately
            if track_by_success:
                self._save_successful_count(successful_count)

            # Check if we should stop
            stop = (
                self.min_successful_completions != -1 and successful_count >= self.min_successful_completions
            ) or generated >= dataset_size

            yield (batch_conversations, stop)

            if stop:
                break

        if pbar:
            pbar.close()
