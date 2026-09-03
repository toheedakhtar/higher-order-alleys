"""Run the matched SELF-versus-OTHER global J-Lens experiment.

Each factual answer is generated once. The exact question/answer prefix is then
reused for evaluation prompts that differ only in ``your`` versus ``their``.
The layer-40 candidate is measured and steered regardless of its readout rank.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from experiments.higher_v_readout_global import protocol as base_protocol
from experiments.higher_v_readout_global import runner as base_runner
from experiments.higher_v_readout_global import steering

from . import protocol


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = SCRIPT_DIR / "experiment_config.json"

TRIAL_FIELDS = (
    "item_id", "item_type", "domain", "difficulty", "condition",
    "labels", "factual_prompt", "factual_answer", "factual_answer_sha256",
    "answer_key", "factual_correct", "factual_response_invalid", "scoring_method",
    "evaluation_prompt", "question_mark_position", "question_mark_token",
    "candidate_token_id", "candidate_layer", "candidate_score", "candidate_rank",
    "candidate_word_filtered_rank", "candidate_in_top_k", "baseline_output_raw",
    "baseline_output_normalized", "baseline_valid", "baseline_judgment_correct",
    "baseline_margin", "baseline_oriented_margin",
)

PAIRED_FIELDS = (
    "item_id", "item_type", "domain", "difficulty", "labels", "candidate_token_id",
    "candidate_layer", "factual_answer",
    "factual_answer_sha256", "same_question_and_answer", "requested_strength",
    "effective_strength_self", "effective_strength_other", "candidate_score_self",
    "candidate_rank_self", "candidate_word_filtered_rank_self",
    "candidate_score_other", "candidate_rank_other",
    "candidate_word_filtered_rank_other", "self_minus_other_candidate_score",
    "candidate_rank_self_minus_other", "baseline_margin_self",
    "baseline_margin_other", "baseline_oriented_margin_self",
    "baseline_oriented_margin_other", "intervened_margin_self",
    "intervened_margin_other", "steering_delta_raw_self",
    "steering_delta_raw_other", "steering_delta_self", "steering_delta_other",
    "self_minus_other_steering_effect", "baseline_output_self",
    "baseline_output_other", "intervened_output_self", "intervened_output_other",
    "flipped_self", "flipped_other", "flip_effect_self", "flip_effect_other",
)


class Recorder:
    """Append-only recorder compatible with the existing intervention machinery."""

    def __init__(self, run_dir: Path, run_id: str, *, resume: bool = False) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.logger = logging.getLogger(f"self_v_external.{run_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        for handler in (
            logging.FileHandler(run_dir / "experiment.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.jsonl_paths = {
            name: run_dir / name for name in (
                "events.jsonl", "raw_runs.jsonl", "readouts.jsonl",
                "tokenizations.jsonl", "adaptive_paths.jsonl", "errors.jsonl",
            )
        }
        for path in self.jsonl_paths.values():
            path.touch(exist_ok=resume)
        self.csv_paths = {
            "trial_summary": run_dir / "trial_summary.csv",
            "intervention_results": run_dir / "intervention_results.csv",
            "paired_results": run_dir / "paired_results.csv",
        }
        if not resume:
            self._header(self.csv_paths["trial_summary"], TRIAL_FIELDS)
            self._header(
                self.csv_paths["intervention_results"], base_runner.INTERVENTION_FIELDS
            )
            self._header(self.csv_paths["paired_results"], PAIRED_FIELDS)

    @staticmethod
    def _header(path: Path, fields: Sequence[str]) -> None:
        with path.open("x", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()

    def jsonl(self, name: str, value: dict[str, Any]) -> None:
        with self.jsonl_paths[name].open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(
                value, ensure_ascii=False, default=base_runner.json_default
            ) + "\n")
            handle.flush()

    def event(self, event_type: str, **values: Any) -> None:
        self.jsonl("events.jsonl", {
            "timestamp": base_runner.utc_now(), "run_id": self.run_id,
            "event_type": event_type, **values,
        })

    def csv(self, name: str, fields: Sequence[str], value: dict[str, Any]) -> None:
        with self.csv_paths[name].open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore").writerow(value)
            handle.flush()

    def error(self, item_id: str | None, stage: str, exc: BaseException) -> None:
        self.jsonl("errors.jsonl", {
            "timestamp": base_runner.utc_now(), "run_id": self.run_id,
            "item_id": item_id, "stage": stage,
            "exception_type": type(exc).__name__, "exception": str(exc),
            "traceback": traceback.format_exc(),
        })
        self.event("error", item_id=item_id, stage=stage)
        self.logger.exception("item=%s stage=%s", item_id, stage)

    def close(self) -> None:
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)


def load_config(path: Path) -> dict[str, Any]:
    return base_runner.load_config(path)


def selected_rows(
    rows: Sequence[dict[str, str]], config: dict[str, Any]
) -> list[dict[str, str]]:
    allowed = set(config["dataset"]["selected_item_types"])
    return [row for row in rows if row["item_type"] in allowed]


def validate_config(
    config: dict[str, Any], rows: Sequence[dict[str, str]]
) -> dict[str, Any]:
    selected = selected_rows(rows, config)
    counts: dict[str, int] = {}
    for row in selected:
        counts[row["item_type"]] = counts.get(row["item_type"], 0) + 1
        paired = protocol.build_paired_protocol(row)
        self_messages = protocol.evaluation_messages(
            paired.factual_prompt, "EXACT_ANSWER_X", paired.self_evaluation_prompt
        )
        other_messages = protocol.evaluation_messages(
            paired.factual_prompt, "EXACT_ANSWER_X", paired.other_evaluation_prompt
        )
        protocol.assert_matched_pair(self_messages, other_messages)
        if "WILL_PASS" in paired.factual_prompt or "I_KNOW" in paired.factual_prompt:
            raise ValueError(f"item {row['item_id']} retained a prohibited pre-turn")
        re.compile(row["answer_key"], re.IGNORECASE)
    expected_counts = {"calibration": 66, "prospective": 8, "knowledge_boundary": 8}
    if len(rows) != 90 or len(selected) != 82 or counts != expected_counts:
        raise ValueError(
            f"expected 90 source rows and 82 selected rows {expected_counts}; "
            f"got {len(rows)}, {len(selected)}, {counts}"
        )
    primary = config["interventions"]["primary"]
    enabled = [item for item in config["candidates"] if item.get("enabled", True)]
    candidate_ids = {int(item["token_id"]) for item in enabled}
    if candidate_ids != {97817, 99973}:
        raise ValueError("candidate registry must contain tokens 97817 and 99973")
    selected_candidate = int(primary["candidate_token_id"])
    if selected_candidate not in candidate_ids or int(primary["layer"]) != 40:
        raise ValueError("the frozen candidate must be token 97817 or 99973 at layer 40")
    if [float(value) for value in primary["strengths"]] != [0.0, -1.7, -1.8]:
        raise ValueError("primary strengths must be [0, -1.7, -1.8]")
    if config["rank_policy"].get("visibility_gate") is not False:
        raise ValueError("candidate measurement/intervention must not use a rank gate")
    ids = {row["item_id"] for row in selected}
    missing_pilot = set(map(str, config["pilot"]["item_ids"])) - ids
    if missing_pilot:
        raise ValueError(f"pilot IDs are not eligible: {sorted(missing_pilot)}")
    return {
        "source_sample_count": len(rows), "paired_item_count": len(selected),
        "condition_count": len(selected) * 2, "counts": counts,
        "excluded_item_types": sorted({row["item_type"] for row in rows} - set(counts)),
        "candidate_token_id": selected_candidate, "candidate_layer": 40,
        "candidate_visibility_gate": False,
        "strengths": primary["strengths"],
        "pairing": "same factual question and exact generated answer; final your/their swap only",
    }


def make_run_id(model_id: str, candidate_token_id: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model_id).strip("-").lower()
    return f"{stamp}_{slug}_token{candidate_token_id}_self-v-external"


def answer_sha256(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def candidate_metric(
    prepared: base_runner.PreparedTrial,
    candidate: steering.Candidate,
) -> dict[str, Any]:
    return prepared.readouts["question_mark"]["candidate_metrics"][candidate.feature_id][
        str(candidate.layer)
    ]


def prepare_condition(
    recorder: Recorder,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    item: dict[str, str],
    paired: protocol.PairedProtocol,
    factual_answer: str,
    factual_scoring: dict[str, Any],
    condition: str,
    candidate: steering.Candidate,
    word_ids: torch.Tensor,
    config: dict[str, Any],
) -> base_runner.PreparedTrial:
    evaluation = (
        paired.self_evaluation_prompt if condition == "self"
        else paired.other_evaluation_prompt
    )
    messages = protocol.evaluation_messages(
        paired.factual_prompt, factual_answer, evaluation
    )
    input_ids, rendered, offsets, positions = base_protocol.locate_judgment_positions(
        tokenizer, messages
    )
    source = positions["question_mark"]
    # Both conditions deliberately read the same semantic location: the final
    # evaluation question mark. The full top-k table for every lens layer is
    # retained in readouts.jsonl.
    readout = steering.readout_across_layers(
        lens_model, lens, input_ids, position=source.index,
        candidates=[candidate], top_k=int(config["readout_top_k"]),
        word_ids=word_ids,
        residual_path=(recorder.run_dir / "checkpoints" / "residuals" /
                       f"{item['item_id']}_{condition}_question_mark.pt"),
    )
    recorder.jsonl("tokenizations.jsonl", {
        "timestamp": base_runner.utc_now(), "run_id": recorder.run_id,
        "item_id": item["item_id"], "condition": condition,
        "rendered_chat": rendered, "token_ids": input_ids[0].tolist(),
        "offsets": offsets, "positions": {"question_mark": asdict(source)},
    })
    recorder.jsonl("readouts.jsonl", {
        "timestamp": base_runner.utc_now(), "run_id": recorder.run_id,
        "item_id": item["item_id"], "condition": condition,
        "position": asdict(source), "readout": readout,
    })
    baseline_generation = steering.generate_greedy(
        lens_model, input_ids, tokenizer,
        max_new_tokens=int(config["generation"]["max_choice_tokens"]),
    )
    baseline_normalized, baseline_valid = base_protocol.normalize_choice(
        baseline_generation["raw"], paired.labels
    )
    baseline_scores = steering.score_label_pair(
        lens_model, input_ids, tokenizer, paired.labels
    )
    expected = base_runner.expected_judgment(
        paired.labels, bool(factual_scoring["factual_correct"])
    )
    return base_runner.PreparedTrial(
        item=item, condition=condition, messages=messages, input_ids=input_ids,
        positions={"question_mark": source}, readouts={"question_mark": readout},
        labels=paired.labels, expected_judgment=expected,
        factual_correct=bool(factual_scoring["factual_correct"]),
        factual_answer=factual_answer,
        factual_invalid=base_protocol.is_invalid_factual_response(
            item["item_type"], factual_answer
        ),
        factual_scoring=factual_scoring, pre_generation=None,
        pre_normalized=None, pre_valid=None,
        baseline_generation=baseline_generation,
        baseline_normalized=baseline_normalized,
        baseline_valid=baseline_valid, baseline_scores=baseline_scores,
    )


def trial_row(
    prepared: base_runner.PreparedTrial,
    candidate: steering.Candidate,
    config: dict[str, Any],
) -> dict[str, Any]:
    metric = candidate_metric(prepared, candidate)
    baseline_margin = float(prepared.baseline_scores["margin"])
    top_k = int(config["readout_top_k"])
    return {
        "item_id": prepared.item["item_id"],
        "item_type": prepared.item["item_type"],
        "domain": prepared.item.get("domain"),
        "difficulty": prepared.item.get("difficulty"),
        "condition": prepared.condition, "labels": "/".join(prepared.labels),
        "factual_prompt": prepared.messages[0]["content"],
        "factual_answer": prepared.factual_answer,
        "factual_answer_sha256": answer_sha256(prepared.factual_answer or ""),
        "answer_key": prepared.item["answer_key"],
        "factual_correct": prepared.factual_correct,
        "factual_response_invalid": prepared.factual_invalid,
        "scoring_method": prepared.factual_scoring["scoring_method"],
        "evaluation_prompt": prepared.messages[-1]["content"],
        "question_mark_position": prepared.positions["question_mark"].index,
        "question_mark_token": prepared.positions["question_mark"].token,
        "candidate_token_id": candidate.token_id, "candidate_layer": candidate.layer,
        "candidate_score": metric["score"], "candidate_rank": metric["raw_rank"],
        "candidate_word_filtered_rank": metric["word_filtered_rank"],
        "candidate_in_top_k": int(metric["raw_rank"]) <= top_k,
        "baseline_output_raw": prepared.baseline_generation["raw"],
        "baseline_output_normalized": prepared.baseline_normalized,
        "baseline_valid": prepared.baseline_valid,
        "baseline_judgment_correct": bool(
            prepared.baseline_valid
            and prepared.baseline_normalized == prepared.expected_judgment
        ),
        "baseline_margin": baseline_margin,
        "baseline_oriented_margin": base_runner.oriented_margin(
            baseline_margin, prepared.labels, prepared.expected_judgment
        ),
    }


def paired_row(
    self_trial: base_runner.PreparedTrial,
    other_trial: base_runner.PreparedTrial,
    self_result: dict[str, Any],
    other_result: dict[str, Any],
    candidate: steering.Candidate,
) -> dict[str, Any]:
    self_metric = candidate_metric(self_trial, candidate)
    other_metric = candidate_metric(other_trial, candidate)
    same_prefix = self_trial.messages[:2] == other_trial.messages[:2]
    if not same_prefix or self_trial.factual_answer != other_trial.factual_answer:
        raise RuntimeError("paired trial lost exact question/answer identity")
    return {
        "item_id": self_trial.item["item_id"],
        "item_type": self_trial.item["item_type"],
        "domain": self_trial.item.get("domain"),
        "difficulty": self_trial.item.get("difficulty"),
        "labels": "/".join(self_trial.labels),
        "candidate_token_id": candidate.token_id,
        "candidate_layer": candidate.layer,
        "factual_answer": self_trial.factual_answer,
        "factual_answer_sha256": answer_sha256(self_trial.factual_answer or ""),
        "same_question_and_answer": same_prefix,
        "requested_strength": self_result["requested_strength"],
        "effective_strength_self": self_result["effective_strength_after_cap"],
        "effective_strength_other": other_result["effective_strength_after_cap"],
        "candidate_score_self": self_metric["score"],
        "candidate_rank_self": self_metric["raw_rank"],
        "candidate_word_filtered_rank_self": self_metric["word_filtered_rank"],
        "candidate_score_other": other_metric["score"],
        "candidate_rank_other": other_metric["raw_rank"],
        "candidate_word_filtered_rank_other": other_metric["word_filtered_rank"],
        "self_minus_other_candidate_score": (
            float(self_metric["score"]) - float(other_metric["score"])
        ),
        "candidate_rank_self_minus_other": (
            int(self_metric["raw_rank"]) - int(other_metric["raw_rank"])
        ),
        "baseline_margin_self": self_result["baseline_margin"],
        "baseline_margin_other": other_result["baseline_margin"],
        "baseline_oriented_margin_self": self_result["baseline_oriented_margin"],
        "baseline_oriented_margin_other": other_result["baseline_oriented_margin"],
        "intervened_margin_self": self_result["intervened_margin"],
        "intervened_margin_other": other_result["intervened_margin"],
        "steering_delta_raw_self": self_result["delta_margin"],
        "steering_delta_raw_other": other_result["delta_margin"],
        # Primary causal effects use correct-oriented sequence margins so signs
        # remain comparable for correct and incorrect factual answers.
        "steering_delta_self": self_result["delta_oriented_margin"],
        "steering_delta_other": other_result["delta_oriented_margin"],
        "self_minus_other_steering_effect": (
            float(self_result["delta_oriented_margin"])
            - float(other_result["delta_oriented_margin"])
        ),
        "baseline_output_self": self_result["baseline_output_normalized"],
        "baseline_output_other": other_result["baseline_output_normalized"],
        "intervened_output_self": self_result["intervened_output_normalized"],
        "intervened_output_other": other_result["intervened_output_normalized"],
        "flipped_self": self_result["flipped"],
        "flipped_other": other_result["flipped"],
        "flip_effect_self": self_result["flip_effect"],
        "flip_effect_other": other_result["flip_effect"],
    }


def execute_item(
    recorder: Recorder,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    item: dict[str, str],
    candidate: steering.Candidate,
    word_ids: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, Any]:
    paired = protocol.build_paired_protocol(item)
    recorder.event(
        "trial_started", item_id=item["item_id"], item_type=item["item_type"],
        factual_prompt=paired.factual_prompt,
    )
    factual_messages: list[dict[str, str]] = []
    factual_generation = base_runner.run_plain_turn(
        lens_model, tokenizer, factual_messages, paired.factual_prompt,
        int(config["generation"]["max_answer_tokens"]),
    )
    factual_answer = factual_generation["raw"]
    factual_scoring = base_protocol.score_factual_answer(
        factual_answer, item["answer_key"]
    )
    prepared: dict[str, base_runner.PreparedTrial] = {}
    for condition in ("self", "other"):
        prepared[condition] = prepare_condition(
            recorder, lens_model, lens, tokenizer, item, paired,
            factual_answer, factual_scoring, condition, candidate, word_ids, config,
        )
    protocol.assert_matched_pair(
        prepared["self"].messages, prepared["other"].messages
    )
    if prepared["self"].factual_answer != factual_answer or prepared["other"].factual_answer != factual_answer:
        raise RuntimeError("a paired condition did not retain exact answer X")

    for condition in ("self", "other"):
        recorder.csv(
            "trial_summary", TRIAL_FIELDS,
            trial_row(prepared[condition], candidate, config),
        )

    results: dict[str, list[dict[str, Any]]] = {"self": [], "other": []}
    strengths = [
        float(value) for value in config["interventions"]["primary"]["strengths"]
        if float(value) != 0.0
    ]
    attempt_order = 1
    for strength in strengths:
        for condition in ("self", "other"):
            result = base_runner.run_intervention(
                recorder, lens_model, lens, tokenizer, prepared[condition], candidate,
                analysis_family="paired_primary", is_primary_estimand=True,
                attempt_order=attempt_order, mode="neuronpedia_global",
                requested_strength=strength, source_selector="question_mark",
                target_selector=None, layer_selection_reason="frozen_layer_40_no_rank_gate",
                config=config,
            )
            results[condition].append(result)
            attempt_order += 1
        pair = paired_row(
            prepared["self"], prepared["other"],
            results["self"][-1], results["other"][-1], candidate,
        )
        recorder.csv("paired_results", PAIRED_FIELDS, pair)
        recorder.event(
            "paired_result", item_id=item["item_id"], requested_strength=strength,
            self_minus_other_candidate_score=pair["self_minus_other_candidate_score"],
            self_minus_other_steering_effect=pair["self_minus_other_steering_effect"],
        )
    recorder.jsonl("raw_runs.jsonl", {
        "timestamp": base_runner.utc_now(), "run_id": recorder.run_id,
        "item": item, "paired_protocol": asdict(paired),
        "factual_generation_once": factual_generation,
        "factual_answer_sha256": answer_sha256(factual_answer),
        "self_messages": prepared["self"].messages,
        "other_messages": prepared["other"].messages,
        "same_question_and_answer": True,
    })
    recorder.event(
        "trial_finished", item_id=item["item_id"],
        factual_answer_sha256=answer_sha256(factual_answer),
        paired_rows=len(strengths), intervention_rows=len(strengths) * 2,
    )
    return {"prepared": prepared, "results": results}


def pilot_summary(run_dir: Path, expected_items: int) -> dict[str, Any]:
    import pandas as pd

    trials = pd.read_csv(run_dir / "trial_summary.csv")
    pairs = pd.read_csv(run_dir / "paired_results.csv")
    checks = {
        "expected_condition_rows": len(trials) == expected_items * 2,
        "both_conditions_per_item": bool(
            not trials.empty and (trials.groupby("item_id")["condition"].nunique() == 2).all()
        ),
        "exact_answer_hash_shared": bool(
            not trials.empty
            and (trials.groupby("item_id")["factual_answer_sha256"].nunique() == 1).all()
        ),
        "expected_paired_rows": len(pairs) == expected_items * 2,
        "pair_identity_recorded": bool(
            not pairs.empty
            and pairs["same_question_and_answer"].astype(str).str.lower().eq("true").all()
        ),
        "both_strengths_per_item": bool(
            not pairs.empty and (pairs.groupby("item_id")["requested_strength"].nunique() == 2).all()
        ),
        "errors_empty": base_runner.record_count(run_dir / "errors.jsonl") == 0,
    }
    result = {"checks": checks, "passed": all(checks.values())}
    (run_dir / "pilot_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_manifest(
    *, run_id: str, phase: str, config: dict[str, Any], dataset_path: Path,
    selected_count: int, args: argparse.Namespace, hf_model: Any | None = None,
    tokenizer: Any | None = None, lens_path: Path | None = None,
    lens_revision: str | None = None, lens_model: Any | None = None,
    candidates: Sequence[steering.Candidate] = (),
) -> dict[str, Any]:
    manifest = base_runner.build_manifest(
        run_id=run_id, phase=phase, config=config, dataset_path=dataset_path,
        selected_count=selected_count, args=args, hf_model=hf_model,
        tokenizer=tokenizer, lens_path=lens_path, lens_revision=lens_revision,
        lens_model=lens_model, candidates=candidates,
    )
    manifest["experiment_name"] = "self_v_external"
    manifest["condition_count"] = selected_count * 2
    manifest["pairing_contract"] = {
        "factual_generation_count_per_item": 1,
        "shared_prefix": "exact same factual question and generated answer text",
        "only_evaluation_difference": "your versus their",
        "excluded_pre_turns": ["WILL_PASS/WILL_FAIL", "I_KNOW/I_WOULD_GUESS"],
        "readout_position": "question mark in final evaluation user turn",
        "rank_gate": False,
    }
    manifest["analysis_contract"] = {
        "primary_unit": "paired factual item",
        "primary_rows": "paired_results.csv, one item per nonzero strength",
        "candidate_contrast": "candidate_score_self - candidate_score_other",
        "steering_contrast": (
            "correct-oriented margin delta SELF - correct-oriented margin delta OTHER"
        ),
        "prohibited_claim": "higher-order representation M(P)",
    }
    return manifest


def write_run_readme(
    run_dir: Path, run_id: str, status: str, candidate_token_id: int
) -> None:
    (run_dir / "README_run.md").write_text(
        f"""# Matched SELF-versus-OTHER run `{run_id}`

Status: {status}

Each factual answer is generated once and reused exactly in SELF and OTHER.
The evaluation prompts differ only in `your` versus `their`. Candidate token
`{candidate_token_id}` is measured and globally steered at layer 40 without a
rank/visibility gate.

The maximum supported claim is a self-evaluation-selective causal evaluator
candidate. This experiment cannot establish a higher-order representation M(P).
""",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("validate", "pilot", "full"), default="validate")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device-map", choices=("cuda", "auto", "cpu"))
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--hf-cache-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--pilot-run", type=Path)
    parser.add_argument("--skip-gate-requirement", action="store_true")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--candidate-token-id", type=int, choices=(97817, 99973))
    return parser


def configure_args(args: argparse.Namespace, config: dict[str, Any]) -> None:
    model = config["model"]
    args.model = args.model or model["id"]
    args.revision = args.revision or model.get("revision", "main")
    args.dtype = args.dtype or model.get("dtype", "bfloat16")
    args.device_map = args.device_map or model.get("device_map", "cuda")
    args.seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    config["seed"] = args.seed
    if args.candidate_token_id is not None:
        config["interventions"]["primary"]["candidate_token_id"] = int(
            args.candidate_token_id
        )


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    config = load_config(args.config.resolve())
    configure_args(args, config)
    dataset_path = base_runner.resolve_repo_path(config["dataset"]["path"])
    all_rows = base_protocol.load_dataset_rows(dataset_path)
    static = validate_config(config, all_rows)
    print(json.dumps(static, indent=2, ensure_ascii=False))
    if args.phase == "validate":
        return 0
    if args.device_map in {"cuda", "auto"} and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Qwen3.6-27B")
    if args.device_map == "cpu" and not args.allow_cpu:
        raise SystemExit("pass --allow-cpu explicitly for a small debug model")

    gate_evidence = None
    if args.phase == "full" and not args.skip_gate_requirement:
        if args.pilot_run is None:
            raise SystemExit("--phase full requires --pilot-run PATH")
        gate_evidence = base_runner.validate_gate_run(args.pilot_run, "pilot", config)

    eligible = selected_rows(all_rows, config)
    by_id = {row["item_id"]: row for row in eligible}
    selected = (
        [by_id[str(item_id)] for item_id in config["pilot"]["item_ids"]]
        if args.phase == "pilot" else eligible
    )
    resume = args.resume_run is not None
    if resume:
        if args.phase != "full":
            raise ValueError("resume is supported only for the full phase")
        run_dir = args.resume_run.resolve()
        frozen = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        old_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if frozen != config or old_manifest.get("phase") != args.phase:
            raise ValueError("resume configuration or phase mismatch")
        run_id = old_manifest["run_id"]
    else:
        run_id = make_run_id(
            args.model, int(config["interventions"]["primary"]["candidate_token_id"])
        )
        run_dir = base_runner.prepare_run_dir(args.output_root.resolve(), run_id)
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    recorder = Recorder(run_dir, run_id, resume=resume)
    initial_errors = base_runner.record_count(run_dir / "errors.jsonl")
    manifest = build_manifest(
        run_id=run_id, phase=args.phase, config=config, dataset_path=dataset_path,
        selected_count=len(selected), args=args,
    )
    manifest["gate_evidence"] = gate_evidence
    manifest["gate_requirement_bypassed"] = bool(args.skip_gate_requirement)
    manifest["resumed"] = resume
    if resume:
        manifest["experiment_started_at"] = old_manifest["experiment_started_at"]
        manifest["resume_count"] = int(old_manifest.get("resume_count", 0)) + 1
    base_runner.write_manifest(run_dir, manifest)
    selected_candidate_id = int(
        config["interventions"]["primary"]["candidate_token_id"]
    )
    write_run_readme(run_dir, run_id, manifest["status"], selected_candidate_id)

    try:
        transformers, jlens = base_runner.require_runtime()
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        hf_model, tokenizer = base_runner.load_model(args, transformers)
        lens_model = jlens.from_hf(hf_model, tokenizer, compile=False)
        lens, lens_path, lens_revision = base_runner.load_lens(
            jlens, config, args.hf_cache_dir
        )
        if lens.d_model != lens_model.d_model:
            raise RuntimeError("lens/model width mismatch")
        placement = (
            base_runner.place_lens(lens, lens_model)
            if config["lens"].get("resident_jacobians", True) else {}
        )
        primary = config["interventions"]["primary"]
        candidate_spec = next(
            item for item in config["candidates"]
            if int(item["token_id"]) == int(primary["candidate_token_id"])
        )
        candidate = steering.resolve_candidate(
            candidate_spec, int(primary["layer"]), tokenizer, lens_model, lens,
            run_dir / "checkpoints",
        )
        word_ids = steering.word_token_ids(tokenizer)
        manifest = build_manifest(
            run_id=run_id, phase=args.phase, config=config, dataset_path=dataset_path,
            selected_count=len(selected), args=args, hf_model=hf_model,
            tokenizer=tokenizer, lens_path=lens_path, lens_revision=lens_revision,
            lens_model=lens_model, candidates=[candidate],
        )
        manifest["lens"]["matrix_placement"] = placement
        manifest["rank_policy"] = {
            **config["rank_policy"], "word_filtered_vocabulary_size": len(word_ids)
        }
        manifest["gate_evidence"] = gate_evidence
        manifest["gate_requirement_bypassed"] = bool(args.skip_gate_requirement)
        manifest["resumed"] = resume
        if resume:
            manifest["experiment_started_at"] = old_manifest["experiment_started_at"]
            manifest["resume_count"] = int(old_manifest.get("resume_count", 0)) + 1
        base_runner.write_manifest(run_dir, manifest)
        write_run_readme(run_dir, run_id, manifest["status"], selected_candidate_id)
        recorder.event("model_loaded", runtime=manifest["runtime"], lens=manifest["lens"])

        finished = base_runner.completed_items(run_dir / "events.jsonl")
        for item in selected:
            if item["item_id"] in finished:
                recorder.logger.info("skip completed item=%s", item["item_id"])
                continue
            try:
                execute_item(
                    recorder, lens_model, lens, tokenizer, item, candidate,
                    word_ids, config,
                )
            except Exception as exc:
                recorder.error(item["item_id"], "paired_trial", exc)

        phase_gate = None
        if args.phase == "pilot":
            phase_gate = pilot_summary(run_dir, len(selected))
        from .analysis import analyze

        plot_status = analyze(run_dir)
        new_errors = base_runner.record_count(run_dir / "errors.jsonl") - initial_errors
        if new_errors:
            status = "completed_with_errors"
        elif args.phase == "pilot" and phase_gate and not phase_gate["passed"]:
            status = "pilot_gate_failed"
        else:
            status = "completed"
        manifest["status"] = status
        manifest["logged_error_count"] = base_runner.record_count(run_dir / "errors.jsonl")
        manifest["experiment_ended_at"] = base_runner.utc_now()
        manifest["plot_status"] = plot_status
        base_runner.write_manifest(run_dir, manifest)
        write_run_readme(run_dir, run_id, status, selected_candidate_id)
        recorder.event("run_finished", status=status, plot_status=plot_status)
        recorder.logger.info("run=%s status=%s path=%s", run_id, status, run_dir)
        return 0 if status == "completed" else 2
    except Exception as exc:
        recorder.error(None, "run", exc)
        manifest["status"] = "failed"
        manifest["logged_error_count"] = base_runner.record_count(run_dir / "errors.jsonl")
        manifest["experiment_ended_at"] = base_runner.utc_now()
        base_runner.write_manifest(run_dir, manifest)
        write_run_readme(run_dir, run_id, "failed", selected_candidate_id)
        return 1
    finally:
        recorder.close()


if __name__ == "__main__":
    raise SystemExit(main())
