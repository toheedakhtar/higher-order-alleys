"""Protocol construction for the matched SELF-versus-OTHER experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from experiments.higher_v_readout_global import protocol as base_protocol


SELECTED_ITEM_TYPES = ("calibration", "prospective", "knowledge_boundary")


@dataclass(frozen=True)
class PairedProtocol:
    factual_prompt: str
    labels: tuple[str, str]
    self_evaluation_prompt: str
    other_evaluation_prompt: str


def evaluation_prompt(owner: str, labels: Sequence[str]) -> str:
    """Create matched evaluation wording whose only semantic change is ownership."""
    if owner not in {"self", "other"}:
        raise ValueError(f"unsupported owner: {owner!r}")
    if len(labels) != 2:
        raise ValueError("exactly two labels are required")
    first, second = (str(label).upper() for label in labels)
    possessive = "your" if owner == "self" else "their"
    return (
        f"Is {possessive} answer {first} or {second}? "
        f"Reply exactly {first} or {second}."
    )


def build_paired_protocol(row: dict[str, str]) -> PairedProtocol:
    """Build a two-user-turn protocol without prospective/knowledge pre-turns."""
    item_type = row["item_type"]
    if item_type not in SELECTED_ITEM_TYPES:
        raise ValueError(f"item type is not eligible for paired evaluation: {item_type!r}")
    labels = base_protocol.judgment_labels(row["prompt"])
    if item_type == "calibration":
        original = base_protocol.build_trial_protocol(row)
        if original.factual_prompt is None:
            raise ValueError(f"calibration item {row['item_id']} has no factual prompt")
        factual_prompt = original.factual_prompt.strip()
    else:
        # These item families contribute only their factual question. Their
        # WILL_PASS/WILL_FAIL and I_KNOW/I_WOULD_GUESS turns are deliberately
        # absent from this experiment.
        factual_prompt = row["detail"].strip()
    if not factual_prompt:
        raise ValueError(f"item {row['item_id']} has an empty factual prompt")
    return PairedProtocol(
        factual_prompt=factual_prompt,
        labels=labels,
        self_evaluation_prompt=evaluation_prompt("self", labels),
        other_evaluation_prompt=evaluation_prompt("other", labels),
    )


def evaluation_messages(
    factual_prompt: str,
    factual_answer: str,
    evaluation_prompt_text: str,
) -> list[dict[str, str]]:
    """Reuse the exact factual answer in an otherwise matched conversation."""
    return [
        {"role": "user", "content": factual_prompt},
        {"role": "assistant", "content": factual_answer},
        {"role": "user", "content": evaluation_prompt_text},
    ]


def assert_matched_pair(
    self_messages: Sequence[dict[str, str]],
    other_messages: Sequence[dict[str, str]],
) -> None:
    """Fail closed if anything except the evaluation ownership wording differs."""
    if len(self_messages) != 3 or len(other_messages) != 3:
        raise ValueError("paired conditions must each contain exactly three messages")
    if list(self_messages[:2]) != list(other_messages[:2]):
        raise ValueError("SELF and OTHER must reuse the exact question and answer")
    self_final = self_messages[2]
    other_final = other_messages[2]
    if self_final.get("role") != "user" or other_final.get("role") != "user":
        raise ValueError("evaluation must be the second user turn")
    expected_other = self_final["content"].replace(" your ", " their ", 1)
    if other_final["content"] != expected_other:
        raise ValueError("evaluation prompts must differ only by your/their")
