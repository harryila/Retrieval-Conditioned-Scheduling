"""Real LLM ModelAdapter: Qwen2.5-1.5B-Instruct with LoRA fine-tuning via PEFT.

Implements the ModelAdapter interface for closed-book QA:
- study_update / reinforce_update: LoRA gradient step (loss only on answer tokens)
- test: greedy decode + NQ-style exact-match scoring
- Internal gradient accumulation with configurable batch size
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model import ModelAdapter
from .nq_eval import normalize_nq_answer
from .types import QAItem

logger = logging.getLogger(__name__)


@dataclass
class RealModelConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    lr: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_len: int = 256
    max_new_tokens: int = 32
    grad_accum_steps: int = 4
    dtype: str = "bfloat16"
    hf_token: Optional[str] = None
    gen_batch_size: int = 16
    system_prompt: str = "Answer the question with a short factual answer."
    deterministic: bool = False


class RealModelAdapter(ModelAdapter):
    """Qwen2.5-1.5B-Instruct + LoRA adapter for closed-book QA.

    Gradient accumulation is handled internally: items are buffered during
    study_update / reinforce_update, and an optimizer step fires every
    ``grad_accum_steps`` items.  Calling ``flush()`` (or ``test()``) forces
    any pending gradients to be applied immediately.
    """

    def __init__(self, config: RealModelConfig, device: Optional[str] = None) -> None:
        self.config = config
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if config.deterministic:
            import os
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            logger.info("Deterministic mode enabled (CUBLAS_WORKSPACE_CONFIG, cudnn.deterministic=True)")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self._dtype = dtype_map.get(config.dtype, torch.bfloat16)

        logger.info("Loading tokenizer: %s", config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            token=config.hf_token,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info("Loading model: %s  dtype=%s  device=%s", config.model_name, config.dtype, self._device)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=self._dtype,
            token=config.hf_token,
            trust_remote_code=True,
        )
        if self._device != "auto":
            self.base_model = self.base_model.to(self._device)

        self._lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
        )
        self.model = get_peft_model(self.base_model, self._lora_config)
        trainable, total = self.model.get_nb_trainable_parameters()
        logger.info(
            "LoRA applied: %s trainable / %s total params (%.2f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / total,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.optimizer.zero_grad()

        self._accumulated = 0
        self._total_optimizer_steps = 0
        self._total_train_items = 0

        self._prompt_len_cache: dict[str, int] = {}

    # ------------------------------------------------------------------
    # LoRA reset (for multi-seed runs without reloading base weights)
    # ------------------------------------------------------------------

    def reset_adapter(self) -> None:
        """Re-initialise LoRA parameters + optimizer for a fresh seed run."""
        for name, param in self.model.named_parameters():
            if "lora_A" in name:
                torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            elif "lora_B" in name:
                torch.nn.init.zeros_(param)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.optimizer.zero_grad()
        self._accumulated = 0
        self._total_optimizer_steps = 0
        self._total_train_items = 0
        self._prompt_len_cache.clear()
        logger.info("LoRA adapter reset for new run")

    # ------------------------------------------------------------------
    # Chat-template helpers
    # ------------------------------------------------------------------

    def _build_messages(self, question: str, answer: str | None = None) -> list[dict]:
        msgs: list[dict] = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": question},
        ]
        if answer is not None:
            msgs.append({"role": "assistant", "content": answer})
        return msgs

    def _prompt_token_length(self, question: str) -> int:
        """Number of tokens for the prompt (system + user + generation prefix)."""
        if question in self._prompt_len_cache:
            return self._prompt_len_cache[question]

        prompt_text = self.tokenizer.apply_chat_template(
            self._build_messages(question),
            tokenize=False,
            add_generation_prompt=True,
        )
        length = len(
            self.tokenizer(prompt_text, truncation=True, max_length=self.config.max_seq_len).input_ids
        )
        self._prompt_len_cache[question] = length
        return length

    # ------------------------------------------------------------------
    # Training internals
    # ------------------------------------------------------------------

    def _train_on_item(self, item: QAItem) -> None:
        target = item.target.split("|||")[0].strip()
        messages = self._build_messages(item.prompt, target)

        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_len,
        )
        input_ids = enc.input_ids.to(self.model.device)
        attention_mask = enc.attention_mask.to(self.model.device)

        prompt_len = self._prompt_token_length(item.prompt)
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / self.config.grad_accum_steps
        loss.backward()

        self._accumulated += 1
        self._total_train_items += 1

        if self._accumulated >= self.config.grad_accum_steps:
            self._step_optimizer()

    def _step_optimizer(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._accumulated = 0
        self._total_optimizer_steps += 1

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    def study_update(self, item: QAItem) -> None:
        self.model.train()
        self._train_on_item(item)

    def reinforce_update(self, item: QAItem) -> None:
        self.model.train()
        self._train_on_item(item)

    def flush(self) -> None:
        """Force pending accumulated gradients through the optimizer."""
        if self._accumulated > 0:
            self._step_optimizer()

    @torch.no_grad()
    def test(self, item: QAItem) -> tuple[bool, float]:
        self.flush()
        self.model.eval()

        # --- compute loss on primary target ---
        target = item.target.split("|||")[0].strip()
        full_messages = self._build_messages(item.prompt, target)
        full_text = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        full_enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_len,
        )
        full_ids = full_enc.input_ids.to(self.model.device)
        full_mask = full_enc.attention_mask.to(self.model.device)

        prompt_len = self._prompt_token_length(item.prompt)
        labels = full_ids.clone()
        labels[0, :prompt_len] = -100

        outputs = self.model(input_ids=full_ids, attention_mask=full_mask, labels=labels)
        loss = outputs.loss.item()

        # --- greedy generation ---
        prompt_messages = self._build_messages(item.prompt)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_enc = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_len,
        )
        prompt_ids = prompt_enc.input_ids.to(self.model.device)
        prompt_mask = prompt_enc.attention_mask.to(self.model.device)

        gen_ids = self.model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_tokens = gen_ids[0, prompt_ids.shape[1] :]
        prediction = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # --- NQ exact-match against all acceptable answers ---
        all_targets = [t.strip() for t in item.target.split("|||")]
        norm_pred = normalize_nq_answer(prediction)
        correct = any(normalize_nq_answer(t) == norm_pred for t in all_targets)

        return correct, loss

    # ------------------------------------------------------------------
    # Batched test (generation is the bottleneck — batch it)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def test_batch(self, items: list[QAItem]) -> list[tuple[bool, float]]:
        """Batched test: sequential per-item loss + left-padded batched generation."""
        if not items:
            return []
        self.flush()
        self.model.eval()

        losses = [self._forward_loss(it) for it in items]
        predictions = self._generate_batch([it.prompt for it in items])

        results: list[tuple[bool, float]] = []
        for item, loss, pred in zip(items, losses, predictions):
            all_targets = [t.strip() for t in item.target.split("|||")]
            norm_pred = normalize_nq_answer(pred)
            correct = any(normalize_nq_answer(t) == norm_pred for t in all_targets)
            results.append((correct, loss))
        return results

    def _forward_loss(self, item: QAItem) -> float:
        """Single-item forward pass returning scalar loss (no generation)."""
        target = item.target.split("|||")[0].strip()
        messages = self._build_messages(item.prompt, target)
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_len,
        )
        input_ids = enc.input_ids.to(self.model.device)
        attention_mask = enc.attention_mask.to(self.model.device)

        prompt_len = self._prompt_token_length(item.prompt)
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss.item()

    def _generate_batch(self, questions: list[str]) -> list[str]:
        """Batched greedy decode with left-padding. Returns one prediction per question."""
        bs = self.config.gen_batch_size
        all_preds: list[str] = []

        orig_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        try:
            for start in range(0, len(questions), bs):
                chunk = questions[start : start + bs]
                prompt_texts = [
                    self.tokenizer.apply_chat_template(
                        self._build_messages(q), tokenize=False, add_generation_prompt=True
                    )
                    for q in chunk
                ]

                enc = self.tokenizer(
                    prompt_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_len,
                )
                input_ids = enc.input_ids.to(self.model.device)
                attention_mask = enc.attention_mask.to(self.model.device)

                gen_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                prompt_len = input_ids.shape[1]
                for i in range(len(chunk)):
                    tokens = gen_ids[i, prompt_len:]
                    pred = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    all_preds.append(pred)
        finally:
            self.tokenizer.padding_side = orig_side

        return all_preds

    # ------------------------------------------------------------------
    # Difficulty calibration (run once on base model before training)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_loss(self, item: QAItem) -> float:
        """Forward-only loss on an item (no generation). Used for difficulty calibration."""
        self.model.eval()

        target = item.target.split("|||")[0].strip()
        messages = self._build_messages(item.prompt, target)
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_len,
        )
        input_ids = enc.input_ids.to(self.model.device)
        attention_mask = enc.attention_mask.to(self.model.device)

        prompt_len = self._prompt_token_length(item.prompt)
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss.item()
