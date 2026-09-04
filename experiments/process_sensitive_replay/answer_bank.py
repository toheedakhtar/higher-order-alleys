"""Clean answer discovery used only to freeze the visible factual answer X."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from experiments.higher_v_readout_global.protocol import (
    is_invalid_factual_response,
    score_factual_answer,
)

from .protocol import direct_factual_question, hash_token_ids
from .cache_state import release_cache_storage
from .replay import (
    CANONICAL_ASSISTANT_TURN_TERMINATOR,
    QwenReplayAdapter,
    canonical_assistant_turn_end_ids,
    encode_chat,
    encode_rendered,
    eos_token_ids,
)


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
    canonical_terminal_ids = canonical_assistant_turn_end_ids(tokenizer)
    if canonical_terminal_ids[0] not in eos:
        raise RuntimeError("canonical <|im_end|> token is not configured as a valid model terminator")
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
    original_terminal_ids = [generated[-1]] if terminated else []
    content_ids = list(generated[:-1] if terminated else generated)
    canonical_turn_end_ids = canonical_terminal_ids if terminated else []
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
    # Preserve answer-content IDs exactly. Only the invisible assistant-turn
    # delimiter is normalized to the chat template's canonical <|im_end|>.
    post_answer_ids = [*prefix_ids, *content_ids, *canonical_turn_end_ids]
    rendered_turn, rendered_turn_ids = encode_chat(
        tokenizer,
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        add_generation_prompt=False,
    )
    canonical_factual_rendered = (
        rendered + answer + CANONICAL_ASSISTANT_TURN_TERMINATOR
    )
    canonical_factual_ids = [
        int(value)
        for value in encode_rendered(tokenizer, canonical_factual_rendered)[
            "input_ids"
        ][0].tolist()
    ]
    separator_ids = [
        int(value)
        for value in encode_rendered(tokenizer, "\n")["input_ids"][0].tolist()
    ]
    canonical_transcript_exact = (
        terminated and canonical_factual_ids == post_answer_ids
    )
    completed_template_exact = (
        terminated
        and rendered_turn == canonical_factual_rendered + "\n"
        and rendered_turn_ids == [*post_answer_ids, *separator_ids]
    )
    chat_reconstruction_exact = (
        canonical_transcript_exact and completed_template_exact
    )
    invalid = (
        not terminated
        or not content_ids
        or not stable
        or not chat_reconstruction_exact
        or is_invalid_factual_response(str(row["item_type"]), answer)
    )
    scoring = score_factual_answer(answer, str(row["answer_key"]))
    content_logprobs = token_logprobs[: len(content_ids)]
    result = {
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
            "reached_token_cap_without_valid_turn_termination": not terminated,
            "empty_answer": not content_ids,
            "decode_retokenize_unstable": not stable,
            "chat_reconstruction_unstable": not chat_reconstruction_exact,
            "malformed_factual_answer": is_invalid_factual_response(str(row["item_type"]), answer),
        },
        "question_rendered": rendered,
        "question_prefix_token_ids": prefix_ids,
        "answer_token_ids": content_ids,
        "turn_end_token_ids": canonical_turn_end_ids,
        "generated_turn_end_token_ids": original_terminal_ids,
        "canonical_turn_end_token_ids": canonical_turn_end_ids,
        "generated_turn_end_token_hash": hash_token_ids(original_terminal_ids),
        "canonical_turn_end_token_hash": hash_token_ids(canonical_turn_end_ids),
        "post_answer_token_ids": post_answer_ids,
        "canonical_factual_rendered": canonical_factual_rendered,
        "canonical_transcript_exact": canonical_transcript_exact,
        "completed_template_exact": completed_template_exact,
        "completed_template_separator_token_ids": separator_ids,
        "question_token_hash": hash_token_ids(prefix_ids),
        "answer_token_hash": hash_token_ids(content_ids),
        "transcript_hash": hash_token_ids(post_answer_ids),
        "answer_sequence_logp": float(sum(content_logprobs)),
        "generation": {
            "method": "cached_greedy_argmax",
            "all_generated_token_ids": generated,
            "token_logprobs": token_logprobs,
            "terminated_on_eos": terminated,
            "terminated_on_valid_turn_end": terminated,
            "original_terminal_token_ids": original_terminal_ids,
            "canonical_terminal_token_ids": canonical_turn_end_ids,
            "terminal_was_already_canonical": original_terminal_ids == canonical_turn_end_ids,
            "max_answer_tokens": int(max_answer_tokens),
        },
    }
    release_cache_storage(cache)
    return result
