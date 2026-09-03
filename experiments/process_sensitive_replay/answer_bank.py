"""Clean answer discovery used only to freeze the visible factual answer X."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from experiments.higher_v_readout_global.protocol import (
    is_invalid_factual_response,
    score_factual_answer,
)

from .protocol import direct_factual_question, hash_token_ids
from .replay import QwenReplayAdapter, encode_chat, eos_token_ids


def discover_answer(
    adapter: QwenReplayAdapter,
    tokenizer: Any,
    row: Mapping[str, str],
    *,
    max_answer_tokens: int,
) -> dict[str, Any]:
    question = direct_factual_question(row)
    rendered, prefix_ids = encode_chat(
        tokenizer,
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
    )
    cache = adapter.new_cache()
    logits: torch.Tensor | None = None
    for position, token_id in enumerate(prefix_ids):
        logits, _ = adapter.step(token_id, cache, expected_position=position)
    if logits is None:
        raise AssertionError("factual question rendered to no tokens")
    eos = eos_token_ids(tokenizer, adapter.hf_model)
    generated: list[int] = []
    token_logprobs: list[float] = []
    terminated = False
    for _ in range(int(max_answer_tokens)):
        log_probs = logits.log_softmax(dim=-1)
        token_id = int(logits.argmax().item())
        generated.append(token_id)
        token_logprobs.append(float(log_probs[token_id].item()))
        if token_id in eos:
            terminated = True
            break
        logits, _ = adapter.step(
            token_id,
            cache,
            expected_position=adapter.cache_length(cache),
        )
    content_ids = list(generated)
    while content_ids and content_ids[-1] in eos:
        content_ids.pop()
    answer = tokenizer.decode(
        content_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    canonical_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if canonical_ids and isinstance(canonical_ids[0], list):
        canonical_ids = canonical_ids[0]
    canonical_ids = [int(value) for value in canonical_ids]
    stable = canonical_ids == content_ids
    post_answer_ids = [*prefix_ids, *generated]
    _, rendered_turn_ids = encode_chat(
        tokenizer,
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        add_generation_prompt=False,
    )
    rendered_prefix_stable = rendered_turn_ids[: len(post_answer_ids)] == post_answer_ids
    invalid = (
        not terminated
        or not content_ids
        or not stable
        or not rendered_prefix_stable
        or is_invalid_factual_response(str(row["item_type"]), answer)
    )
    scoring = score_factual_answer(answer, str(row["answer_key"]))
    content_logprobs = token_logprobs[: len(content_ids)]
    return {
        "item_id": str(row["item_id"]),
        "item_type": str(row["item_type"]),
        "question": question,
        "raw_question": str(row["prompt"]),
        "answer": answer,
        "answer_key": str(row["answer_key"]),
        "factual_correct": bool(scoring["factual_correct"]),
        "factual_scoring": scoring,
        "invalid": invalid,
        "invalid_reasons": {
            "missing_eos": not terminated,
            "empty_answer": not content_ids,
            "decode_retokenize_unstable": not stable,
            "chat_reconstruction_unstable": not rendered_prefix_stable,
            "malformed_factual_answer": is_invalid_factual_response(str(row["item_type"]), answer),
        },
        "question_rendered": rendered,
        "question_prefix_token_ids": prefix_ids,
        "answer_token_ids": content_ids,
        "turn_end_token_ids": generated[len(content_ids) :],
        "post_answer_token_ids": post_answer_ids,
        "question_token_hash": hash_token_ids(prefix_ids),
        "answer_token_hash": hash_token_ids(content_ids),
        "transcript_hash": hash_token_ids(post_answer_ids),
        "answer_sequence_logp": float(sum(content_logprobs)),
        "generation": {
            "method": "cached_greedy_argmax",
            "all_generated_token_ids": generated,
            "token_logprobs": token_logprobs,
            "terminated_on_eos": terminated,
            "max_answer_tokens": int(max_answer_tokens),
        },
    }
