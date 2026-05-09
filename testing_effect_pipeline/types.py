from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class QAItem:
    """Single closed-book QA item with stable identifier."""

    item_id: str
    prompt: str
    target: str
    domain_tag: str = "squad"
    difficulty: Optional[float] = None


@dataclass
class ItemState:
    """Per-item memory/scheduling state used by retrieval-gated replay."""

    item_id: str
    last_study_step: int = -1
    last_test_step: int = -1
    success_streak: int = 0
    failure_count: int = 0
    estimated_half_life: float = 1.0
    next_due_step: int = 0
    is_mastered: bool = False
    mastered_at_step: Optional[int] = None
    test_accuracy_ema: float = 0.0
    test_loss_ema: float = 1.0
    total_tests: int = 0
    total_correct: int = 0
    remastery_count: int = 0


@dataclass
class RetentionSnapshot:
    step: int
    total_mastered: int
    retained_after_5k: float
    retained_after_10k: float
    retained_after_20k: float


@dataclass
class StepAllocation:
    step: int
    study_count: int
    test_count: int
    reinforce_count: int


@dataclass
class BudgetSnapshot:
    step: int
    training_tokens_used: int
    test_inference_tokens_used: int


@dataclass
class UniformEvalResult:
    """Result of running exact-match generation on all items after training."""

    step: int
    correct_count: int
    total: int
    accuracy: float
    mean_loss: float
    per_item: list[tuple[str, bool, float]] = field(default_factory=list)


@dataclass
class TrainingMetrics:
    forgetting_snapshots: list[RetentionSnapshot] = field(default_factory=list)
    cumulative_mastered: list[tuple[int, int]] = field(default_factory=list)
    mastery_throughput: list[tuple[int, float]] = field(default_factory=list)
    step_allocations: list[StepAllocation] = field(default_factory=list)
    budget_snapshots: list[BudgetSnapshot] = field(default_factory=list)
    remastery_events: list[tuple[int, str]] = field(default_factory=list)
    total_remastery_events: int = 0
    stopped_early_budget: bool = False
    uniform_eval_results: list[UniformEvalResult] = field(default_factory=list)
    held_out_eval_results: dict[str, list[UniformEvalResult]] = field(default_factory=dict)
