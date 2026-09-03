"""Exact token replay over Qwen's mutable hybrid cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .cache_state import (
    CacheAudit,
    assert_cache_unchanged,
    audit_cache,
    clone_hybrid_cache,
)
from .gradient_intervention import (
    InterventionSchedule,
    hidden_tensor,
    replace_hidden,
)
from .protocol import hash_token_ids


def render_chat(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError as exc:
        raise RuntimeError("Qwen chat template must support enable_thinking=False") from exc


def encode_rendered(tokenizer: Any, rendered: str, *, offsets: bool = False) -> Any:
    return tokenizer(
        rendered,
        add_special_tokens=False,
        return_tensors="pt",
        return_offsets_mapping=offsets,
    )


def encode_chat(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> tuple[str, list[int]]:
    rendered = render_chat(tokenizer, messages, add_generation_prompt=add_generation_prompt)
    encoded = encode_rendered(tokenizer, rendered)
    return rendered, [int(value) for value in encoded["input_ids"][0].tolist()]


def eos_token_ids(tokenizer: Any, model: Any | None = None) -> set[int]:
    result: set[int] = set()
    for value in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    ):
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            result.update(int(item) for item in value)
        else:
            result.add(int(value))
    return result


class QwenReplayAdapter:
    """Thin adapter that preserves the actual Transformers DynamicCache."""

    def __init__(self, hf_model: Any, lens_model: Any) -> None:
        self.hf_model = hf_model
        self.lens_model = lens_model
        self.text_module = lens_model._text_module
        self.layers = lens_model.layers
        self.lm_head = lens_model._lm_head
        self.input_device = lens_model.input_device
        self.text_config = hf_model.config.get_text_config()
        self.intervention_hook_registrations = 0
        if len(self.layers) != int(self.text_config.num_hidden_layers):
            raise RuntimeError("J-Lens adapter/model layer count mismatch")

    def new_cache(self) -> Any:
        from transformers import DynamicCache

        return DynamicCache(config=self.hf_model.config)

    def cache_length(self, cache: Any) -> int:
        return int(cache.get_seq_length())

    def step(
        self,
        token_id: int,
        cache: Any,
        *,
        expected_position: int,
        intervention: InterventionSchedule | None = None,
        capture_layers: Sequence[int] = (),
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        observed_position = self.cache_length(cache)
        if observed_position != int(expected_position):
            raise AssertionError(
                f"cache position mismatch: expected {expected_position}, observed {observed_position}"
            )
        captures: dict[int, torch.Tensor] = {}
        handles = []
        process_layer = None if intervention is None else intervention.process_layer
        if intervention is not None:
            self.intervention_hook_registrations += 1
        hook_layers = sorted(set(int(layer) for layer in capture_layers) | ({process_layer} if process_layer is not None else set()))

        def make_hook(layer: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = hidden_tensor(output)
                if layer == process_layer:
                    delta = intervention.delta_for(expected_position) if intervention is not None else None
                    if delta is not None:
                        changed = hidden.clone()
                        changed[:, -1, :] += delta.to(device=hidden.device, dtype=hidden.dtype)
                        output = replace_hidden(output, changed)
                        hidden = changed
                if layer in capture_layers:
                    captures[layer] = hidden[:, -1, :].detach().clone()
                return output

            return hook

        try:
            for layer in hook_layers:
                handles.append(self.layers[layer].register_forward_hook(make_hook(layer)))
            inputs = torch.tensor([[int(token_id)]], dtype=torch.long, device=self.input_device)
            with torch.no_grad():
                outputs = self.text_module(
                    input_ids=inputs,
                    past_key_values=cache,
                    use_cache=True,
                )
                hidden = outputs.last_hidden_state[:, -1, :].to(self.lm_head.weight.device)
                logits = self.lm_head(hidden).float().detach()
            if not torch.isfinite(logits).all():
                raise AssertionError("model step produced non-finite logits")
            if self.cache_length(cache) != expected_position + 1:
                raise AssertionError("cache failed to advance by exactly one token")
            return logits[0], captures
        finally:
            for handle in handles:
                handle.remove()


@dataclass
class ReplayOutcome:
    cache: Any
    cache_audit: CacheAudit
    answer_sequence_logp: float
    token_logprobs: tuple[float, ...]
    transcript_hash: str
    question_token_hash: str
    answer_token_hash: str
    teacher_forced: bool
    intervention_positions: tuple[int, ...]


def replay_teacher_forced(
    adapter: QwenReplayAdapter,
    *,
    post_answer_token_ids: Sequence[int],
    question_prefix_token_ids: Sequence[int],
    answer_token_ids: Sequence[int],
    intervention: InterventionSchedule | None = None,
) -> ReplayOutcome:
    complete = [int(value) for value in post_answer_token_ids]
    prefix = [int(value) for value in question_prefix_token_ids]
    answer = [int(value) for value in answer_token_ids]
    if complete[: len(prefix)] != prefix:
        raise AssertionError("post-answer transcript does not start with question prefix IDs")
    if complete[len(prefix) : len(prefix) + len(answer)] != answer:
        raise AssertionError("post-answer transcript does not contain canonical answer IDs")
    cache = adapter.new_cache()
    token_logprobs: list[float] = []
    targets_by_predictor = {
        len(prefix) - 1 + index: token_id for index, token_id in enumerate(answer)
    }
    for position, token_id in enumerate(complete):
        logits, _ = adapter.step(
            token_id,
            cache,
            expected_position=position,
            intervention=intervention,
        )
        if position in targets_by_predictor:
            target = targets_by_predictor[position]
            token_logprobs.append(float(logits.log_softmax(dim=-1)[target].item()))
    if len(token_logprobs) != len(answer):
        raise AssertionError("did not score every teacher-forced answer token")
    if not all(torch.isfinite(torch.tensor(value)).item() for value in token_logprobs):
        raise AssertionError("teacher-forced answer support is non-finite")
    if intervention is not None:
        intervention.assert_complete()
    return ReplayOutcome(
        cache=cache,
        cache_audit=audit_cache(cache),
        answer_sequence_logp=float(sum(token_logprobs)),
        token_logprobs=tuple(token_logprobs),
        transcript_hash=hash_token_ids(complete),
        question_token_hash=hash_token_ids(prefix),
        answer_token_hash=hash_token_ids(answer),
        teacher_forced=True,
        intervention_positions=tuple(() if intervention is None else sorted(intervention.positions)),
    )


def locate_meta_question_and_suffix(
    tokenizer: Any,
    *,
    question: str,
    answer: str,
    post_answer_token_ids: Sequence[int],
    meta_prompt: str,
) -> tuple[list[int], int, str]:
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
        {"role": "user", "content": meta_prompt},
    ]
    rendered = render_chat(tokenizer, messages, add_generation_prompt=True)
    encoded = encode_rendered(tokenizer, rendered, offsets=True)
    ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    prefix = [int(value) for value in post_answer_token_ids]
    if ids[: len(prefix)] != prefix:
        raise AssertionError("Turn-3 rendering changed the canonical post-answer token prefix")
    content_start = rendered.rfind(meta_prompt)
    question_offset = meta_prompt.find("?")
    if content_start < 0 or question_offset < 0:
        raise RuntimeError("could not locate Turn-3 question mark")
    character = content_start + question_offset
    offsets = encoded["offset_mapping"][0].tolist()
    question_position = next(
        (
            index for index, (start, end) in enumerate(offsets)
            if int(start) <= character < int(end) and int(end) > int(start)
        ),
        None,
    )
    if question_position is None or ids[question_position] != int(tokenizer("?", add_special_tokens=False)["input_ids"][0]):
        raise AssertionError("Turn-3 '?' token alignment failed")
    return ids[len(prefix) :], int(question_position), rendered


def append_meta_prompt(
    adapter: QwenReplayAdapter,
    source_cache: Any,
    *,
    suffix_token_ids: Sequence[int],
    question_position: int,
    capture_layers: Sequence[int],
) -> tuple[Any, torch.Tensor, dict[int, torch.Tensor], CacheAudit, int]:
    source_before = audit_cache(source_cache)
    branch = clone_hybrid_cache(source_cache)
    hook_registrations_before = adapter.intervention_hook_registrations
    start = adapter.cache_length(branch)
    captures: dict[int, torch.Tensor] = {}
    last_logits: torch.Tensor | None = None
    for offset, token_id in enumerate(suffix_token_ids):
        position = start + offset
        requested = capture_layers if position == question_position else ()
        last_logits, current = adapter.step(
            int(token_id), branch, expected_position=position, capture_layers=requested
        )
        captures.update(current)
    assert_cache_unchanged(source_before, source_cache, "Turn-3 branch append")
    process_hook_registrations = adapter.intervention_hook_registrations - hook_registrations_before
    if last_logits is None:
        raise AssertionError("empty Turn-3 suffix")
    if set(captures) != set(int(layer) for layer in capture_layers):
        raise AssertionError("failed to capture every requested J-Lens layer at '?'")
    return branch, last_logits, captures, audit_cache(branch), process_hook_registrations


def label_token_ids(tokenizer: Any, label: str) -> list[int]:
    encoded = tokenizer(label, add_special_tokens=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    if not encoded:
        raise RuntimeError(f"label {label!r} tokenized to no tokens")
    return [int(value) for value in encoded]


def score_label_from_cache(
    adapter: QwenReplayAdapter,
    boundary_cache: Any,
    boundary_logits: torch.Tensor,
    tokenizer: Any,
    label: str,
) -> dict[str, Any]:
    source_before = audit_cache(boundary_cache)
    cache = clone_hybrid_cache(boundary_cache)
    ids = label_token_ids(tokenizer, label)
    logits = boundary_logits
    values: list[float] = []
    for index, token_id in enumerate(ids):
        values.append(float(logits.log_softmax(dim=-1)[token_id].item()))
        if index + 1 < len(ids):
            logits, _ = adapter.step(
                token_id,
                cache,
                expected_position=adapter.cache_length(cache),
            )
    assert_cache_unchanged(source_before, boundary_cache, f"label scoring {label}")
    return {"label": label, "token_ids": ids, "token_logprobs": values, "sequence_logprob": sum(values)}


def score_label_pair_from_cache(
    adapter: QwenReplayAdapter,
    boundary_cache: Any,
    boundary_logits: torch.Tensor,
    tokenizer: Any,
    labels: Sequence[str],
) -> dict[str, Any]:
    if len(labels) != 2:
        raise ValueError("exactly two meta labels are required")
    first = score_label_from_cache(adapter, boundary_cache, boundary_logits, tokenizer, labels[0])
    second = score_label_from_cache(adapter, boundary_cache, boundary_logits, tokenizer, labels[1])
    return {
        labels[0]: first,
        labels[1]: second,
        "margin": float(first["sequence_logprob"] - second["sequence_logprob"]),
        "margin_definition": f"logP({labels[0]}) - logP({labels[1]})",
    }


def generate_from_cache(
    adapter: QwenReplayAdapter,
    boundary_cache: Any,
    boundary_logits: torch.Tensor,
    tokenizer: Any,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    source_before = audit_cache(boundary_cache)
    cache = clone_hybrid_cache(boundary_cache)
    logits = boundary_logits
    generated: list[int] = []
    values: list[float] = []
    eos = eos_token_ids(tokenizer, adapter.hf_model)
    for _ in range(int(max_new_tokens)):
        log_probs = logits.log_softmax(dim=-1)
        token_id = int(logits.argmax().item())
        generated.append(token_id)
        values.append(float(log_probs[token_id].item()))
        if token_id in eos:
            break
        logits, _ = adapter.step(
            token_id,
            cache,
            expected_position=adapter.cache_length(cache),
        )
    assert_cache_unchanged(source_before, boundary_cache, "meta generation")
    return {
        "raw": tokenizer.decode(generated, skip_special_tokens=True),
        "token_ids": generated,
        "token_logprobs": values,
        "sequence_logprob": float(sum(values)),
    }
