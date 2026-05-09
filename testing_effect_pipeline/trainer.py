from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import random
import statistics

from .budget import TokenBudgetTracker
from .model import ModelAdapter
from .scheduler import Scheduler
from .types import BudgetSnapshot, ItemState, QAItem, RetentionSnapshot, StepAllocation, TrainingMetrics
from .uniform_eval import run_uniform_eval

logger = logging.getLogger(__name__)

VALID_MODES = {"test_only", "test_reinforce", "retrieval_practice", "scheduled_restudy"}


@dataclass
class TrainConfig:
    total_steps: int = 25_000
    batch_size: int = 16
    mastery_k: int = 2
    eval_every_steps: int = 1_000
    min_study_fraction: float = 0.3
    max_test_fraction: float = 0.5
    max_training_tokens: int | None = None
    loss_threshold: float | None = None
    uniform_eval_items: list[QAItem] | None = None
    periodic_held_out_sets: list[tuple[str, list[QAItem]]] | None = None


class TestingEffectTrainer:
    def __init__(
        self,
        items: list[QAItem],
        model: ModelAdapter,
        scheduler: Scheduler,
        config: TrainConfig,
        mode: str = "test_only",
        seed: int = 7,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}")

        self.items = items
        self.item_by_id = {i.item_id: i for i in items}
        self.model = model
        self.scheduler = scheduler
        self.config = config
        self.mode = mode
        self.rng = random.Random(seed)

        self.state = {i.item_id: ItemState(item_id=i.item_id) for i in items}
        self.metrics = TrainingMetrics()
        self.budget = TokenBudgetTracker(max_training_tokens=config.max_training_tokens)
        self._study_cursor = 0
        self._recent_losses: deque[float] = deque(maxlen=200)

    def _next_study_items(self, n: int) -> list[QAItem]:
        out: list[QAItem] = []
        for _ in range(n):
            item = self.items[self._study_cursor % len(self.items)]
            self._study_cursor += 1
            out.append(item)
        return out

    def _due_items(self, step: int) -> list[QAItem]:
        due = [self.item_by_id[item_id] for item_id, st in self.state.items() if st.next_due_step <= step]
        self.rng.shuffle(due)
        return due

    def _loss_to_correct(self, loss: float) -> bool:
        """Adaptive threshold: item is 'correct' if loss is below the running median."""
        self._recent_losses.append(loss)
        if len(self._recent_losses) < 10:
            return loss < 1.0
        return loss < statistics.median(self._recent_losses)

    def _update_item_state(self, item: QAItem, step: int, correct: bool, loss: float) -> None:
        """Update per-item scheduling and mastery state after evaluation."""
        st = self.state[item.item_id]
        st.last_test_step = step
        st.total_tests += 1
        st.total_correct += int(correct)
        alpha = 0.2
        st.test_accuracy_ema = (1 - alpha) * st.test_accuracy_ema + alpha * float(correct)
        st.test_loss_ema = (1 - alpha) * st.test_loss_ema + alpha * loss

        if correct:
            st.success_streak += 1
        else:
            if st.is_mastered:
                st.is_mastered = False
            st.success_streak = 0
            st.failure_count += 1

        self.scheduler.on_result(st, step, correct)

        if st.success_streak >= self.config.mastery_k:
            if st.mastered_at_step is not None and not st.is_mastered:
                st.remastery_count += 1
                self.metrics.total_remastery_events += 1
                self.metrics.remastery_events.append((step, item.item_id))
            st.is_mastered = True
            if st.mastered_at_step is None:
                st.mastered_at_step = step

    def _retention_at_horizon(self, step: int, delta: int) -> float:
        eligible = [
            st.item_id
            for st in self.state.values()
            if st.is_mastered and st.mastered_at_step is not None and st.mastered_at_step <= step - delta
        ]
        if not eligible:
            return 0.0
        items = [self.item_by_id[iid] for iid in eligible]
        for item in items:
            self.budget.add_test_inference(item)
        results = self.model.test_batch(items)
        return sum(int(c) for c, _ in results) / len(results)

    def _snapshot(self, step: int) -> None:
        mastered = sum(1 for st in self.state.values() if st.is_mastered)
        self.metrics.forgetting_snapshots.append(
            RetentionSnapshot(
                step=step,
                total_mastered=mastered,
                retained_after_5k=self._retention_at_horizon(step, 5_000),
                retained_after_10k=self._retention_at_horizon(step, 10_000),
                retained_after_20k=self._retention_at_horizon(step, 20_000),
            )
        )
        self.metrics.cumulative_mastered.append((step, mastered))
        self.metrics.budget_snapshots.append(
            BudgetSnapshot(
                step=step,
                training_tokens_used=self.budget.training_tokens_used,
                test_inference_tokens_used=self.budget.test_inference_tokens_used,
            )
        )

        if len(self.metrics.cumulative_mastered) > 1:
            prev_step, prev_mastered = self.metrics.cumulative_mastered[-2]
            span = max(1, step - prev_step)
            throughput = (mastered - prev_mastered) / span * 1000.0
            self.metrics.mastery_throughput.append((step, throughput))

    def _step_test_only_or_reinforce(self, step: int, due: list[QAItem]) -> None:
        """One training step for test_only / test_reinforce modes."""
        cfg = self.config
        due_pressure = min(1.0, len(due) / max(1, cfg.batch_size))

        test_fraction = min(cfg.max_test_fraction, due_pressure)
        study_fraction = max(cfg.min_study_fraction, 1.0 - test_fraction)

        n_test = int(cfg.batch_size * test_fraction)
        n_study = int(cfg.batch_size * study_fraction)

        study_items = self._next_study_items(n_study)
        for item in study_items:
            self.model.study_update(item)
            self.budget.add_study(item)
            self.state[item.item_id].last_study_step = step

        test_items = due[:n_test] if n_test > 0 else []
        failures: list[QAItem] = []

        if test_items:
            for item in test_items:
                self.budget.add_test_inference(item)
            test_results = self.model.test_batch(test_items)

            for item, (correct, loss) in zip(test_items, test_results):
                self._update_item_state(item, step, correct, loss)
                if not correct:
                    failures.append(item)

        reinforced = 0
        if self.mode == "test_reinforce" and failures:
            max_reinforce = max(1, n_study // 4)
            for item in failures[:max_reinforce]:
                self.model.reinforce_update(item)
                self.budget.add_reinforce(item)
                reinforced += 1

        self.metrics.step_allocations.append(
            StepAllocation(step=step, study_count=n_study, test_count=len(test_items), reinforce_count=reinforced)
        )

        if step % 50 == 0:
            mastered = sum(1 for st in self.state.values() if st.is_mastered)
            correct_count = sum(int(c) for c, _ in test_results) if test_items else 0
            logger.info(
                f"step {step}/{cfg.total_steps} | studied={n_study} tested={len(test_items)} correct={correct_count} "
                f"reinforced={reinforced} due={len(due)} mastered={mastered} | train_tokens={self.budget.training_tokens_used}"
            )

    def _step_retrieval_practice(self, step: int, due: list[QAItem]) -> None:
        """One training step for retrieval_practice mode.

        For each due item: generate answer -> score -> update schedule -> gradient step.
        The model attempts retrieval first, then learns from the answer.
        """
        cfg = self.config
        due_pressure = min(1.0, len(due) / max(1, cfg.batch_size))
        test_fraction = min(cfg.max_test_fraction, due_pressure)
        study_fraction = max(cfg.min_study_fraction, 1.0 - test_fraction)

        n_due = int(cfg.batch_size * test_fraction)
        n_study = int(cfg.batch_size * study_fraction)

        study_items = self._next_study_items(n_study)
        for item in study_items:
            self.model.study_update(item)
            self.budget.add_study(item)
            self.state[item.item_id].last_study_step = step

        due_batch = due[:n_due] if n_due > 0 else []

        if due_batch:
            for item in due_batch:
                self.budget.add_test_inference(item)
            test_results = self.model.test_batch(due_batch)

            for item, (correct, loss) in zip(due_batch, test_results):
                self._update_item_state(item, step, correct, loss)
                # Gradient step on every due item after retrieval attempt
                self.model.study_update(item)
                self.budget.add_study(item)

        self.metrics.step_allocations.append(
            StepAllocation(step=step, study_count=n_study + len(due_batch), test_count=len(due_batch), reinforce_count=0)
        )

        if step % 50 == 0:
            mastered = sum(1 for st in self.state.values() if st.is_mastered)
            correct_count = sum(int(c) for c, _ in test_results) if due_batch else 0
            logger.info(
                f"step {step}/{cfg.total_steps} | studied={n_study}+{len(due_batch)}due tested={len(due_batch)} "
                f"correct={correct_count} due={len(due)} mastered={mastered} | train_tokens={self.budget.training_tokens_used}"
            )

    def _step_scheduled_restudy(self, step: int, due: list[QAItem]) -> None:
        """One training step for scheduled_restudy mode.

        For each due item: compute_loss (forward-only, evaluate current knowledge) ->
        derive correctness from adaptive median threshold -> update schedule ->
        gradient step. No generation. Same items, same schedule, same gradient steps
        as retrieval_practice -- only difference is no retrieval attempt before learning.

        The compute_loss call happens before study_update intentionally: it mirrors the
        temporal structure of retrieval_practice (evaluate current knowledge -> update
        schedule -> then learn). Using loss from the training forward pass itself would
        give the scheduler a signal from *during* learning, not *before* learning.
        """
        cfg = self.config
        due_pressure = min(1.0, len(due) / max(1, cfg.batch_size))
        test_fraction = min(cfg.max_test_fraction, due_pressure)
        study_fraction = max(cfg.min_study_fraction, 1.0 - test_fraction)

        n_due = int(cfg.batch_size * test_fraction)
        n_study = int(cfg.batch_size * study_fraction)

        study_items = self._next_study_items(n_study)
        for item in study_items:
            self.model.study_update(item)
            self.budget.add_study(item)
            self.state[item.item_id].last_study_step = step

        due_batch = due[:n_due] if n_due > 0 else []

        for item in due_batch:
            loss = self.model.compute_loss(item)
            self.budget.add_test_inference(item)
            if self.config.loss_threshold is not None:
                correct = loss < self.config.loss_threshold
            else:
                correct = self._loss_to_correct(loss)
            self._update_item_state(item, step, correct, loss)
            self.model.study_update(item)
            self.budget.add_study(item)

        self.metrics.step_allocations.append(
            StepAllocation(step=step, study_count=n_study + len(due_batch), test_count=0, reinforce_count=0)
        )

        if step % 50 == 0:
            mastered = sum(1 for st in self.state.values() if st.is_mastered)
            logger.info(
                f"step {step}/{cfg.total_steps} | studied={n_study}+{len(due_batch)}due "
                f"due={len(due)} mastered={mastered} | train_tokens={self.budget.training_tokens_used}"
            )

    def train(self) -> TrainingMetrics:
        cfg = self.config

        if self.mode in ("retrieval_practice", "scheduled_restudy"):
            step_fn = self._step_retrieval_practice if self.mode == "retrieval_practice" else self._step_scheduled_restudy
        else:
            step_fn = self._step_test_only_or_reinforce

        for step in range(1, cfg.total_steps + 1):
            due = self._due_items(step)
            step_fn(step, due)

            if step % cfg.eval_every_steps == 0:
                self._snapshot(step)
                if cfg.uniform_eval_items is not None:
                    ue = run_uniform_eval(self.model, cfg.uniform_eval_items, step=step, include_per_item=False)
                    self.metrics.uniform_eval_results.append(ue)
                    logger.info("  periodic uniform eval: %d/%d (%.1f%%)", ue.correct_count, ue.total, ue.accuracy * 100)
                if cfg.periodic_held_out_sets:
                    for tag, items_h in cfg.periodic_held_out_sets:
                        he = run_uniform_eval(self.model, items_h, step=step, include_per_item=False)
                        self.metrics.held_out_eval_results.setdefault(tag, []).append(he)
                        logger.info("  periodic held-out [%s]: %d/%d (%.1f%%)",
                                    tag, he.correct_count, he.total, he.accuracy * 100)

            if self.budget.over_budget():
                self.metrics.stopped_early_budget = True
                break

        self.model.flush()
        return self.metrics
