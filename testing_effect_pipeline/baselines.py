from __future__ import annotations

from dataclasses import dataclass
import logging
import random

from .budget import TokenBudgetTracker
from .model import ModelAdapter
from .types import BudgetSnapshot, ItemState, QAItem, RetentionSnapshot, StepAllocation, TrainingMetrics
from .uniform_eval import run_uniform_eval

logger = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    total_steps: int = 25_000
    batch_size: int = 16
    replay_size: int = 8
    eval_every_steps: int = 1_000
    mastery_k: int = 2
    max_training_tokens: int | None = None
    uniform_eval_items: list[QAItem] | None = None
    periodic_held_out_sets: list[tuple[str, list[QAItem]]] | None = None


class BaselineTrainer:
    """Implements standard FT + random replay + curriculum + loss replay baselines."""

    def __init__(self, items: list[QAItem], model: ModelAdapter, cfg: BaselineConfig, policy: str, seed: int = 7):
        if policy not in {"standard_ft", "standard_ft_mastered", "random_replay", "curriculum", "loss_replay"}:
            raise ValueError("unknown baseline policy")
        self.items = items
        self.item_by_id = {i.item_id: i for i in items}
        self.model = model
        self.cfg = cfg
        self.policy = policy
        self.rng = random.Random(seed)
        self.state = {i.item_id: ItemState(item_id=i.item_id) for i in items}
        self.metrics = TrainingMetrics()
        self.budget = TokenBudgetTracker(max_training_tokens=cfg.max_training_tokens)
        self.cursor = 0

        if policy == "curriculum":
            self.items = sorted(items, key=lambda x: x.difficulty if x.difficulty is not None else 1.0)

    def _study_items(self, n: int) -> list[QAItem]:
        if self.policy == "standard_ft_mastered":
            available = [it for it in self.items if not self.state[it.item_id].is_mastered]
            if not available:
                return []
            out: list[QAItem] = []
            for _ in range(n):
                out.append(available[self.cursor % len(available)])
                self.cursor += 1
            return out
        out: list[QAItem] = []
        for _ in range(n):
            out.append(self.items[self.cursor % len(self.items)])
            self.cursor += 1
        return out

    def _probe(self, item: QAItem, step: int) -> bool:
        self.budget.add_test_inference(item)
        correct, loss = self.model.test(item)
        st = self.state[item.item_id]
        st.last_test_step = step
        st.total_tests += 1
        st.total_correct += int(correct)
        st.test_loss_ema = 0.8 * st.test_loss_ema + 0.2 * loss
        if correct:
            st.success_streak += 1
            if st.success_streak >= self.cfg.mastery_k:
                if st.mastered_at_step is not None and not st.is_mastered:
                    st.remastery_count += 1
                    self.metrics.total_remastery_events += 1
                    self.metrics.remastery_events.append((step, item.item_id))
                st.is_mastered = True
                if st.mastered_at_step is None:
                    st.mastered_at_step = step
        else:
            if st.is_mastered:
                st.is_mastered = False
            st.success_streak = 0
            st.failure_count += 1
        return correct

    def _retention(self, step: int, delta: int) -> float:
        eligible = [
            iid
            for iid, st in self.state.items()
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
        mastered = sum(1 for s in self.state.values() if s.is_mastered)
        self.metrics.forgetting_snapshots.append(
            RetentionSnapshot(
                step=step,
                total_mastered=mastered,
                retained_after_5k=self._retention(step, 5_000),
                retained_after_10k=self._retention(step, 10_000),
                retained_after_20k=self._retention(step, 20_000),
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

    def train(self) -> TrainingMetrics:
        for step in range(1, self.cfg.total_steps + 1):
            study_batch = self._study_items(self.cfg.batch_size)
            for item in study_batch:
                self.model.study_update(item)
                self.budget.add_study(item)
                self.state[item.item_id].last_study_step = step

            test_count = 0
            reinforce_count = 0
            if self.policy in {"random_replay", "loss_replay"}:
                if self.policy == "random_replay":
                    replay_items = self.rng.sample(self.items, k=min(self.cfg.replay_size, len(self.items)))
                else:
                    ranked = sorted(self.items, key=lambda x: self.state[x.item_id].test_loss_ema, reverse=True)
                    replay_items = ranked[: min(self.cfg.replay_size, len(ranked))]
                for item in replay_items:
                    self.model.reinforce_update(item)
                    self.budget.add_reinforce(item)
                    reinforce_count += 1
                    self._probe(item, step)
                    test_count += 1
            elif self.policy == "standard_ft":
                probe_items = self.rng.sample(self.items, k=min(self.cfg.replay_size, len(self.items)))
                for item in probe_items:
                    self._probe(item, step)
                    test_count += 1
            elif self.policy == "standard_ft_mastered":
                probe_items = self.rng.sample(self.items, k=min(self.cfg.replay_size, len(self.items)))
                for item in probe_items:
                    self._probe(item, step)
                    test_count += 1
            elif self.policy == "curriculum":
                half = max(10, len(self.items) // 2)
                probe_pool = self.items[:half]
                probe_items = self.rng.sample(probe_pool, k=min(self.cfg.replay_size, len(probe_pool)))
                for item in probe_items:
                    self._probe(item, step)
                    test_count += 1

            self.metrics.step_allocations.append(
                StepAllocation(step=step, study_count=len(study_batch), test_count=test_count, reinforce_count=reinforce_count)
            )

            # Log progress every 50 steps
            if step % 50 == 0:
                mastered = sum(1 for st in self.state.values() if st.is_mastered)
                logger.info(
                    f"step {step}/{self.cfg.total_steps} | studied={len(study_batch)} tested={test_count} reinforced={reinforce_count} "
                    f"mastered={mastered} | train_tokens={self.budget.training_tokens_used}"
                )

            if step % self.cfg.eval_every_steps == 0:
                self._snapshot(step)
                if self.cfg.uniform_eval_items is not None:
                    ue = run_uniform_eval(self.model, self.cfg.uniform_eval_items, step=step, include_per_item=False)
                    self.metrics.uniform_eval_results.append(ue)
                    logger.info("  periodic uniform eval: %d/%d (%.1f%%)", ue.correct_count, ue.total, ue.accuracy * 100)
                if self.cfg.periodic_held_out_sets:
                    for tag, items_h in self.cfg.periodic_held_out_sets:
                        he = run_uniform_eval(self.model, items_h, step=step, include_per_item=False)
                        self.metrics.held_out_eval_results.setdefault(tag, []).append(he)
                        logger.info("  periodic held-out [%s]: %d/%d (%.1f%%)",
                                    tag, he.correct_count, he.total, he.accuracy * 100)

            if self.budget.over_budget():
                self.metrics.stopped_early_budget = True
                break

        self.model.flush()
        return self.metrics
