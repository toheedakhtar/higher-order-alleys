"""Dataset, conversation, scoring, and tokenizer-alignment primitives."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


SELF_TYPES = ("calibration", "prospective", "knowledge_boundary")
ALL_ITEM_TYPES = (*SELF_TYPES, "error_detection")


@dataclass(frozen=True)
class TrialProtocol:
    pre_prompt: str | None
    pre_labels: tuple[str, str] | None
    factual_prompt: str | None
    judgment_prompt: str
    judgment_labels: tuple[str, str]
    condition: str


@dataclass(frozen=True)
class SelectedPosition:
    index: int
    token_id: int
    token: str
    char_index: int
    char_span: tuple[int, int]
    selector: str
    conversation_position: str
    surrounding_tokens: list[dict[str, Any]]


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def normalize_choice(text: str, labels: Sequence[str]) -> tuple[str | None, bool]:
    normalized = normalize_text(text).upper().strip("`* .,:;\"'")
    allowed = {label.upper(): label.upper() for label in labels}
    return (allowed[normalized], True) if normalized in allowed else (None, False)


def score_factual_answer(answer: str, answer_key: str) -> dict[str, Any]:
    """Use the source notebook's case-insensitive regex-search semantics."""
    normalized = normalize_text(answer)
    try:
        match = re.search(answer_key, answer, flags=re.IGNORECASE)
    except re.error as exc:
        return {
            "normalized_answer": normalized,
            "answer_key": answer_key,
            "scoring_method": "regex_search_ignorecase_original_notebook",
            "factual_correct": False,
            "score_debug": {"regex_error": str(exc)},
        }
    return {
        "normalized_answer": normalized,
        "answer_key": answer_key,
        "scoring_method": "regex_search_ignorecase_original_notebook",
        "factual_correct": match is not None,
        "score_debug": None
        if match is None
        else {"matched_text": match.group(0), "span": list(match.span())},
    }


def is_invalid_factual_response(item_type: str, answer: str) -> bool:
    normalized = normalize_text(answer).upper()
    if not normalized:
        return True
    if item_type == "calibration":
        return not bool(
            normalized in {"TRUE", "FALSE"}
            or re.search(r"\bVERDICT\s*:\s*(TRUE|FALSE)\b", normalized)
        )
    forbidden = {
        "prospective": {"WILL_PASS", "WILL_FAIL"},
        "knowledge_boundary": {"I_KNOW", "I_WOULD_GUESS"},
    }
    return normalized in forbidden.get(item_type, set())


def load_dataset_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        "item_id",
        "prompt",
        "item_type",
        "answer_key",
        "difficulty",
        "domain",
        "detail",
    }
    actual = set(rows[0]) if rows else set()
    if not rows or actual != expected:
        raise ValueError(f"unexpected dataset columns: {sorted(actual)}")
    return rows


def _last_nonempty_line(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    last = next(
        (index for index in range(len(lines) - 1, -1, -1) if lines[index].strip()),
        None,
    )
    if last is None:
        raise ValueError("prompt contains no non-empty lines")
    return "\n".join(lines[:last]).rstrip(), lines[last].strip()


def judgment_labels(prompt: str) -> tuple[str, str]:
    upper = prompt.upper()
    for labels in (
        ("PASS", "FAIL"),
        ("CORRECT", "INCORRECT"),
        ("RIGHT", "WRONG"),
        ("YES", "NO"),
    ):
        if all(re.search(rf"\b{re.escape(label)}\b", upper) for label in labels):
            return labels
    raise ValueError(f"could not infer binary judgment labels from {prompt!r}")


def factual_turn_prompt(item_type: str, detail: str) -> str:
    if item_type == "prospective":
        prior = "your previous WILL_PASS or WILL_FAIL prediction"
    elif item_type == "knowledge_boundary":
        prior = "I_KNOW or I_WOULD_GUESS"
    else:
        return detail
    return (
        "Now answer the factual question itself.\n"
        f"Do not repeat {prior}.\n"
        "Reply only with the factual answer.\n\n"
        f"{detail}"
    )


def build_trial_protocol(row: dict[str, str]) -> TrialProtocol:
    """Split real turns while preserving the supplied final question verbatim."""
    item_type = row["item_type"]
    if item_type not in ALL_ITEM_TYPES:
        raise ValueError(f"unsupported item type: {item_type!r}")
    if item_type == "error_detection":
        return TrialProtocol(
            pre_prompt=None,
            pre_labels=None,
            factual_prompt=None,
            judgment_prompt=row["prompt"].strip(),
            judgment_labels=judgment_labels(row["prompt"]),
            condition="external",
        )

    before_judgment, judgment_prompt = _last_nonempty_line(row["prompt"])
    if item_type == "calibration":
        return TrialProtocol(
            pre_prompt=None,
            pre_labels=None,
            factual_prompt=before_judgment,
            judgment_prompt=judgment_prompt,
            judgment_labels=judgment_labels(judgment_prompt),
            condition="self",
        )

    detail = row["detail"]
    detail_start = before_judgment.rfind(detail)
    if detail_start < 0:
        raise ValueError(f"item {row['item_id']} detail is absent from prompt")
    if before_judgment[detail_start + len(detail) :].strip():
        raise ValueError(f"item {row['item_id']} has text after factual detail")
    return TrialProtocol(
        pre_prompt=before_judgment[:detail_start].rstrip(),
        pre_labels=("WILL_PASS", "WILL_FAIL")
        if item_type == "prospective"
        else ("I_KNOW", "I_WOULD_GUESS"),
        factual_prompt=factual_turn_prompt(item_type, detail),
        judgment_prompt=judgment_prompt,
        judgment_labels=judgment_labels(judgment_prompt),
        condition="self",
    )


def render_chat(tokenizer: Any, messages: Sequence[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as exc:
        raise RuntimeError("chat template must accept enable_thinking=False") from exc


def tokenize_for_generation(
    tokenizer: Any, messages: Sequence[dict[str, str]]
) -> torch.Tensor:
    return tokenizer(
        render_chat(tokenizer, messages),
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]


def _position_from_ids(
    tokenizer: Any,
    input_ids: torch.Tensor,
    offsets: Sequence[Sequence[int]],
    *,
    index: int,
    char_index: int,
    selector: str,
    conversation_position: str,
) -> SelectedPosition:
    ids = input_ids[0].tolist()
    surrounding = [
        {
            "index": position,
            "token_id": int(ids[position]),
            "token": tokenizer.decode(
                [int(ids[position])], clean_up_tokenization_spaces=False
            ),
            "offset": list(offsets[position]),
        }
        for position in range(max(0, index - 4), min(len(ids), index + 5))
    ]
    return SelectedPosition(
        index=index,
        token_id=int(ids[index]),
        token=tokenizer.decode([int(ids[index])], clean_up_tokenization_spaces=False),
        char_index=char_index,
        char_span=tuple(int(value) for value in offsets[index]),
        selector=selector,
        conversation_position=conversation_position,
        surrounding_tokens=surrounding,
    )


def locate_judgment_positions(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
) -> tuple[torch.Tensor, str, list[list[int]], dict[str, SelectedPosition]]:
    """Locate direction source and localized comparison using actual offsets."""
    rendered = render_chat(tokenizer, messages)
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    if "offset_mapping" not in encoded:
        raise RuntimeError("a fast tokenizer with offset mappings is required")
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"][0].tolist()
    content = messages[-1]["content"]
    content_start = rendered.rfind(content)
    local_char = content.rfind("?")
    if content_start < 0 or local_char < 0:
        raise RuntimeError("could not locate final judgment question mark")
    char_index = content_start + local_char
    index = next(
        (
            i
            for i, (start, end) in enumerate(offsets)
            if start <= char_index < end and end > start
        ),
        None,
    )
    if index is None:
        raise RuntimeError("question mark has no tokenizer span")
    question = _position_from_ids(
        tokenizer,
        input_ids,
        offsets,
        index=index,
        char_index=char_index,
        selector="question_mark",
        conversation_position="final_user_turn_judgment_question_mark",
    )
    previous = None
    for candidate_index in range(index - 1, -1, -1):
        start, end = offsets[candidate_index]
        token = tokenizer.decode(
            [int(input_ids[0, candidate_index])],
            clean_up_tokenization_spaces=False,
        )
        if end > start and token.strip() and any(char.isalnum() for char in token):
            previous = _position_from_ids(
                tokenizer,
                input_ids,
                offsets,
                index=candidate_index,
                char_index=start,
                selector="meaningful_token_before_question_mark",
                conversation_position="final_user_turn_token_before_question_mark",
            )
            break
    if previous is None:
        raise RuntimeError("no meaningful token exists before question mark")
    return input_ids, rendered, offsets, {
        "question_mark": question,
        "meaningful_token_before_question_mark": previous,
    }


def exported_position(
    tokenizer: Any,
    token_ids: Sequence[int],
    token_records: Sequence[dict[str, Any]],
    index: int,
) -> SelectedPosition:
    surrounding = [
        {
            "index": position,
            "token_id": int(token_ids[position]),
            "token": str(token_records[position].get("token", "")),
            "offset": [-1, -1],
        }
        for position in range(max(0, index - 4), min(len(token_ids), index + 5))
    ]
    return SelectedPosition(
        index=index,
        token_id=int(token_ids[index]),
        token=str(token_records[index].get("token", tokenizer.decode([token_ids[index]]))),
        char_index=-1,
        char_span=(-1, -1),
        selector="question_mark",
        conversation_position="exported_final_user_turn_judgment_question_mark",
        surrounding_tokens=surrounding,
    )


def load_neuronpedia_export(path: Path, parity: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "chat" or payload.get("modelId") != "qwen3.6-27b":
        raise ValueError("unexpected Neuronpedia export identity")
    prompt_len = int((payload.get("meta") or {}).get("prompt_len", -1))
    tokens = payload.get("tokens") or []
    if prompt_len != int(parity["expected_prompt_tokens"]):
        raise ValueError(f"expected {parity['expected_prompt_tokens']} prompt tokens")
    if len(tokens) < prompt_len:
        raise ValueError("export token list is shorter than prompt_len")
    prompt_tokens = tokens[:prompt_len]
    if [int(item["position"]) for item in prompt_tokens] != list(range(prompt_len)):
        raise ValueError("export positions are not contiguous")
    source = int(parity["expected_source_position"])
    if prompt_tokens[source].get("token") != "?":
        raise ValueError("configured export source position is not a question mark")
    messages = payload.get("messages") or []
    roles = [message.get("role") for message in messages]
    if roles != ["user", "assistant", "user", "assistant", "user", "assistant"]:
        raise ValueError("export must contain the complete three-turn chat")
    steer = payload.get("steer") or {}
    steer_config = steer.get("config") or {}
    expected_steer = {
        "token": "评价",
        "type": "JACOBIAN_LENS",
        "layers": [40],
        "strength": -1.7,
        "ablate": False,
        "mode": "steer",
        "steerGenerated": True,
    }
    mismatches = {
        key: {"expected": value, "observed": steer_config.get(key)}
        for key, value in expected_steer.items()
        if steer_config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"exported steering configuration mismatch: {mismatches}")
    generated = "".join(
        str(item.get("token", ""))
        for item in (steer.get("tokens") or [])[prompt_len:]
        if item.get("is_generated") and item.get("section") == "content"
    )
    return {
        "payload": payload,
        "messages_before_judgment_output": [dict(message) for message in messages[:-1]],
        "input_token_ids": [int(item["id"]) for item in prompt_tokens],
        "prompt_tokens": prompt_tokens,
        "prompt_len": prompt_len,
        "source_position": source,
        "exported_baseline": messages[-1]["content"],
        "exported_intervention": generated,
        "steer_config": steer_config,
    }


def position_record(position: SelectedPosition) -> dict[str, Any]:
    return asdict(position)
