"""Run the frozen Neuronpedia-global J-Lens metacognition experiment.

The primary estimand is one predeclared global candidate/layer/polarity family
per sample. Localized interventions and adaptive candidate/layer searches are
recorded as separate control and rescue families and never enter primary CIs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import platform
import random
import re
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from . import protocol
from . import steering


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = SCRIPT_DIR / "experiment_config.json"

FORCED_OUTPUTS = (
    ("forced_pass", "For this formatting test, what must you reply? Reply exactly PASS.", ("PASS", "FAIL"), "PASS"),
    ("forced_fail", "For this formatting test, what must you reply? Reply exactly FAIL.", ("PASS", "FAIL"), "FAIL"),
    ("forced_correct", "For this formatting test, what must you reply? Reply exactly CORRECT.", ("CORRECT", "INCORRECT"), "CORRECT"),
    ("forced_incorrect", "For this formatting test, what must you reply? Reply exactly INCORRECT.", ("CORRECT", "INCORRECT"), "INCORRECT"),
)

SUMMARY_FIELDS = (
    "item_id", "item_type", "condition", "domain", "difficulty",
    "pre_meta_output_raw", "pre_meta_output_normalized", "pre_meta_valid",
    "factual_answer", "answer_key", "factual_correct", "factual_response_invalid",
    "scoring_method", "judgment_prompt", "judgment_labels", "expected_judgment",
    "baseline_output_raw", "baseline_output_normalized", "baseline_valid",
    "baseline_judgment_correct", "baseline_margin", "baseline_oriented_margin",
    "direction_source_selector", "direction_source_position", "direction_source_token",
    "primary_candidate_token_id", "primary_layer", "primary_raw_rank",
    "primary_word_filtered_rank", "primary_visible", "primary_status",
    "primary_negative_flipped", "primary_positive_flipped", "adaptive_status",
)

INTERVENTION_FIELDS = (
    "item_id", "item_type", "condition", "domain", "difficulty", "analysis_family",
    "is_primary_estimand", "attempt_order", "intervention_mode", "intervention_scope",
    "direction_source_selector", "direction_source_position", "localized_target_selector",
    "localized_target_position", "applied_prompt_position_count", "steer_generated",
    "max_injection_fraction", "requested_strength", "effective_strength_after_cap",
    "feature_label", "feature_id", "token_id", "direction_sha256", "layer",
    "layer_selection_reason", "raw_rank", "word_filtered_rank", "rank_policy",
    "baseline_output_raw", "baseline_output_normalized", "intervened_output_raw",
    "intervened_output_normalized", "baseline_valid", "intervened_valid",
    "labels", "expected_judgment", "factual_correct", "baseline_judgment_correct",
    "intervened_judgment_correct", "baseline_margin", "intervened_margin", "delta_margin",
    "baseline_oriented_margin", "intervened_oriented_margin", "delta_oriented_margin",
    "candidate_score_before", "candidate_score_after", "flipped", "flip_effect",
    "generated_positions_steered", "duration_seconds",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("model", "lens", "dataset", "candidates", "interventions", "generation"):
        if key not in config:
            raise ValueError(f"config missing {key!r}")
    return config


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def validate_config(config: dict[str, Any], rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    ids = []
    for row in rows:
        ids.append(row["item_id"])
        counts[row["item_type"]] = counts.get(row["item_type"], 0) + 1
        protocol.build_trial_protocol(row)
        try:
            re.compile(row["answer_key"], re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"invalid answer regex for item {row['item_id']}") from exc
    if len(rows) != 90 or counts != {
        "calibration": 66,
        "prospective": 8,
        "error_detection": 8,
        "knowledge_boundary": 8,
    }:
        raise ValueError(f"unexpected dataset composition: {counts}")
    candidates = [item for item in config["candidates"] if item.get("enabled", True)]
    token_ids = [int(item["token_id"]) for item in candidates]
    if len(candidates) != 2 or len(set(token_ids)) != 2:
        raise ValueError("exactly two distinct enabled candidates are required")
    primary = config["interventions"]["primary"]
    if primary["mode"] != "neuronpedia_global":
        raise ValueError("primary mode must be neuronpedia_global")
    if int(primary["candidate_token_id"]) != 97817:
        raise ValueError("the frozen Neuronpedia primary candidate must be token 97817")
    if int(primary["candidate_token_id"]) not in token_ids:
        raise ValueError("primary candidate is absent from candidate registry")
    if int(primary["layer"]) != 40:
        raise ValueError("the frozen primary layer must be 40")
    if [float(value) for value in primary["strengths"]] != [0.0, -1.7, 1.8]:
        raise ValueError("primary strengths must be [0, -1.7, 1.8]")
    if float(primary["max_injection_fraction"]) != 1.0:
        raise ValueError("Neuronpedia parity requires a 1.0 cap")
    if not primary.get("steer_generated"):
        raise ValueError("Neuronpedia parity requires generated-token steering")
    localized = config["interventions"]["localized_control"]
    if localized["mode"] != "single_position" or localized["positions"] != [
        "question_mark", "meaningful_token_before_question_mark"
    ]:
        raise ValueError("localized control positions changed")
    if 0.0 not in [float(value) for value in localized["strengths"]]:
        raise ValueError("localized strengths must declare the shared zero baseline")
    if config["rank_policy"]["appearance_rank"] != "word_filtered_rank":
        raise ValueError("UI-parity fallback must use word_filtered_rank")
    if config["rank_policy"].get("word_filter") != "jlens_meaningful_token_mask":
        raise ValueError("word-filtered rank must use the public jlens meaningful-token rule")
    for gate in ("parity", "pilot"):
        missing = set(str(value) for value in config[gate]["item_ids"]) - set(ids)
        if missing:
            raise ValueError(f"{gate} IDs absent from dataset: {sorted(missing)}")
    return {
        "dataset_counts": counts,
        "all_sample_count": len(rows),
        "primary": primary,
        "candidate_token_ids": token_ids,
        "analysis_families": ["frozen_primary", "localized_control", "adaptive_rescue"],
        "baseline_policy": "generated and scored once per sample; zero is not duplicated",
    }


def require_runtime() -> tuple[Any, Any]:
    try:
        import transformers
        import jlens
    except ImportError as exc:
        raise SystemExit("Run `uv sync` from the project root before executing model phases") from exc
    return transformers, jlens


def torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_model(args: argparse.Namespace, transformers: Any) -> tuple[Any, Any]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, cache_dir=args.hf_cache_dir
    )
    auto_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if auto_cls is None:
        auto_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
    if auto_cls is None:
        raise RuntimeError("Transformers lacks a compatible multimodal auto class")
    model = auto_cls.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.hf_cache_dir,
        dtype=torch_dtype(args.dtype),
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def load_lens(jlens: Any, config: dict[str, Any], cache_dir: Path | None) -> tuple[Any, Path, str | None]:
    from huggingface_hub import snapshot_download

    spec = config["lens"]
    snapshot = Path(snapshot_download(
        spec["repo"], allow_patterns=[spec["file"]], revision=spec.get("revision", "main"),
        cache_dir=None if cache_dir is None else str(cache_dir),
    ))
    path = snapshot / spec["file"]
    lens = jlens.JacobianLens.load(str(path))
    revision = snapshot.name if snapshot.parent.name == "snapshots" else None
    return lens, path, revision


def place_lens(lens: Any, lens_model: Any) -> dict[str, list[int]]:
    placement: dict[str, list[int]] = {}
    for layer in lens.source_layers:
        try:
            device = next(lens_model.layers[layer].parameters()).device
        except StopIteration:
            device = lens_model.input_device
        lens.jacobians[layer] = lens.jacobians[layer].to(device)
        placement.setdefault(str(device), []).append(int(layer))
    return placement


class Recorder:
    def __init__(self, run_dir: Path, run_id: str, resume: bool = False) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.logger = logging.getLogger(f"global_jlens.{run_id}")
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
        }
        if not resume:
            self._header(self.csv_paths["trial_summary"], SUMMARY_FIELDS)
            self._header(self.csv_paths["intervention_results"], INTERVENTION_FIELDS)

    @staticmethod
    def _header(path: Path, fields: Sequence[str]) -> None:
        with path.open("x", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()

    def jsonl(self, name: str, value: dict[str, Any]) -> None:
        with self.jsonl_paths[name].open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, default=json_default) + "\n")
            handle.flush()

    def event(self, event_type: str, **values: Any) -> None:
        self.jsonl("events.jsonl", {
            "timestamp": utc_now(), "run_id": self.run_id,
            "event_type": event_type, **values,
        })

    def csv(self, name: str, fields: Sequence[str], value: dict[str, Any]) -> None:
        with self.csv_paths[name].open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore").writerow(value)
            handle.flush()

    def error(self, item_id: str | None, stage: str, exc: BaseException) -> None:
        value = {
            "timestamp": utc_now(), "run_id": self.run_id, "item_id": item_id,
            "stage": stage, "exception_type": type(exc).__name__,
            "exception": str(exc), "traceback": traceback.format_exc(),
        }
        self.jsonl("errors.jsonl", value)
        self.event("error", item_id=item_id, stage=stage)
        self.logger.exception("item=%s stage=%s", item_id, stage)

    def close(self) -> None:
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)


def make_run_id(model_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model_id).strip("-").lower()
    return f"{stamp}_{slug}_global"


def prepare_run_dir(root: Path, run_id: str) -> Path:
    path = root / run_id
    path.mkdir(parents=True, exist_ok=False)
    (path / "plots").mkdir()
    (path / "checkpoints").mkdir()
    return path


def completed_items(events_path: Path) -> set[str]:
    completed = set()
    if not events_path.exists():
        return completed
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if value.get("event_type") == "trial_finished":
                completed.add(str(value["item_id"]))
    return completed


@dataclass
class PreparedTrial:
    item: dict[str, str]
    condition: str
    messages: list[dict[str, str]]
    input_ids: torch.Tensor
    positions: dict[str, protocol.SelectedPosition]
    readouts: dict[str, dict[str, Any]]
    labels: tuple[str, str]
    expected_judgment: str
    factual_correct: bool
    factual_answer: str | None
    factual_invalid: bool
    factual_scoring: dict[str, Any]
    pre_generation: dict[str, Any] | None
    pre_normalized: str | None
    pre_valid: bool | None
    baseline_generation: dict[str, Any]
    baseline_normalized: str | None
    baseline_valid: bool
    baseline_scores: dict[str, Any]


def run_plain_turn(
    lens_model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    prompt_text: str,
    max_tokens: int,
) -> dict[str, Any]:
    messages.append({"role": "user", "content": prompt_text})
    input_ids = protocol.tokenize_for_generation(tokenizer, messages)
    generated = steering.generate_greedy(
        lens_model, input_ids, tokenizer, max_new_tokens=max_tokens
    )
    messages.append({"role": "assistant", "content": generated["raw"]})
    return generated


def expected_judgment(labels: Sequence[str], factual_correct: bool) -> str:
    return labels[0] if factual_correct else labels[1]


def oriented_margin(margin: float, labels: Sequence[str], expected: str) -> float:
    return float(margin) if expected == labels[0] else -float(margin)


def prepare_trial(
    recorder: Recorder,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    item: dict[str, str],
    base_candidates: Sequence[steering.Candidate],
    word_ids: torch.Tensor,
    config: dict[str, Any],
    *,
    exported: dict[str, Any] | None = None,
) -> PreparedTrial:
    trial_protocol = protocol.build_trial_protocol(item)
    recorder.event(
        "trial_started", item_id=item["item_id"], item_type=item["item_type"],
        condition=trial_protocol.condition, original_prompt=item["prompt"],
        exported_prompt=exported is not None,
    )
    pre_generation = None
    pre_normalized = None
    pre_valid = None
    factual_answer = None
    factual_invalid = False
    if exported is not None:
        messages = [dict(message) for message in exported["messages_before_judgment_output"]]
        input_ids = torch.tensor([exported["input_token_ids"]], dtype=torch.long)
        source = protocol.exported_position(
            tokenizer, exported["input_token_ids"], exported["prompt_tokens"],
            exported["source_position"],
        )
        positions = {"question_mark": source}
        factual_answer = messages[3]["content"]
        factual_scoring = protocol.score_factual_answer(factual_answer, item["answer_key"])
        factual_invalid = protocol.is_invalid_factual_response(item["item_type"], factual_answer)
        recorder.jsonl("tokenizations.jsonl", {
            "timestamp": utc_now(), "run_id": recorder.run_id,
            "item_id": item["item_id"], "prompt_source": "neuronpedia_export_token_ids",
            "token_ids": exported["input_token_ids"], "offsets": None,
            "positions": {"question_mark": asdict(source)},
        })
    else:
        messages: list[dict[str, str]] = []
        if trial_protocol.pre_prompt is not None and trial_protocol.pre_labels is not None:
            pre_generation = run_plain_turn(
                lens_model, tokenizer, messages, trial_protocol.pre_prompt,
                int(config["generation"]["max_choice_tokens"]),
            )
            pre_normalized, pre_valid = protocol.normalize_choice(
                pre_generation["raw"], trial_protocol.pre_labels
            )
        if trial_protocol.factual_prompt is not None:
            factual = run_plain_turn(
                lens_model, tokenizer, messages, trial_protocol.factual_prompt,
                int(config["generation"]["max_answer_tokens"]),
            )
            factual_answer = factual["raw"]
            factual_invalid = protocol.is_invalid_factual_response(
                item["item_type"], factual_answer
            )
            factual_scoring = protocol.score_factual_answer(
                factual_answer, item["answer_key"]
            )
        else:
            source_correct = item["answer_key"].strip().upper() == "CORRECT"
            factual_scoring = {
                "normalized_answer": None,
                "answer_key": item["answer_key"],
                "scoring_method": "dataset_error_detection_answer_key",
                "factual_correct": source_correct,
                "score_debug": None,
            }
        messages.append({"role": "user", "content": trial_protocol.judgment_prompt})
        input_ids, rendered, offsets, positions = protocol.locate_judgment_positions(
            tokenizer, messages
        )
        recorder.jsonl("tokenizations.jsonl", {
            "timestamp": utc_now(), "run_id": recorder.run_id,
            "item_id": item["item_id"], "rendered_chat": rendered,
            "token_ids": input_ids[0].tolist(), "offsets": offsets,
            "positions": {name: asdict(value) for name, value in positions.items()},
        })
    source = positions["question_mark"]
    readouts: dict[str, dict[str, Any]] = {}
    for name, position in positions.items():
        readout = steering.readout_across_layers(
            lens_model, lens, input_ids, position=position.index,
            candidates=base_candidates, top_k=int(config["readout_top_k"]),
            word_ids=word_ids,
            residual_path=(recorder.run_dir / "checkpoints" / "residuals" /
                           f"{item['item_id']}_{name}.pt"),
        )
        readouts[name] = readout
        recorder.jsonl("readouts.jsonl", {
            "timestamp": utc_now(), "run_id": recorder.run_id,
            "item_id": item["item_id"], "position": asdict(position),
            "readout": readout,
        })
    baseline_generation = steering.generate_greedy(
        lens_model, input_ids, tokenizer,
        max_new_tokens=int(config["generation"]["max_choice_tokens"]),
    )
    baseline_normalized, baseline_valid = protocol.normalize_choice(
        baseline_generation["raw"], trial_protocol.judgment_labels
    )
    baseline_scores = steering.score_label_pair(
        lens_model, input_ids, tokenizer, trial_protocol.judgment_labels
    )
    expected = expected_judgment(
        trial_protocol.judgment_labels, bool(factual_scoring["factual_correct"])
    )
    return PreparedTrial(
        item=item, condition=trial_protocol.condition, messages=messages,
        input_ids=input_ids, positions=positions, readouts=readouts,
        labels=trial_protocol.judgment_labels, expected_judgment=expected,
        factual_correct=bool(factual_scoring["factual_correct"]),
        factual_answer=factual_answer, factual_invalid=factual_invalid,
        factual_scoring=factual_scoring, pre_generation=pre_generation,
        pre_normalized=pre_normalized, pre_valid=pre_valid,
        baseline_generation=baseline_generation,
        baseline_normalized=baseline_normalized,
        baseline_valid=baseline_valid, baseline_scores=baseline_scores,
    )


def run_intervention(
    recorder: Recorder,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    prepared: PreparedTrial,
    candidate: steering.Candidate,
    *,
    analysis_family: str,
    is_primary_estimand: bool,
    attempt_order: int,
    mode: str,
    requested_strength: float,
    source_selector: str,
    target_selector: str | None,
    layer_selection_reason: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    if requested_strength == 0:
        raise ValueError("zero is the shared baseline and must not be an intervention row")
    started = time.perf_counter()
    source = prepared.positions[source_selector]
    target = prepared.positions[target_selector] if target_selector else None
    metric_position = target_selector or source_selector
    readout = prepared.readouts[metric_position]
    metric = readout["candidate_metrics"][candidate.feature_id][str(candidate.layer)]
    localized_scale = (
        float(readout["mean_residual_scales"][str(candidate.layer)])
        if mode == "single_position" else None
    )
    primary = config["interventions"]["primary"]
    spec = steering.build_steering_spec(
        mode=mode, candidate=candidate, requested_strength=requested_strength,
        direction_source_position=source.index, input_ids=prepared.input_ids,
        tokenizer=tokenizer,
        selected_position=None if target is None else target.index,
        localized_residual_scale=localized_scale,
        max_injection_fraction=(float(primary["max_injection_fraction"])
                                if mode == "neuronpedia_global" else None),
        steer_generated=(bool(primary["steer_generated"])
                         if mode == "neuronpedia_global" else False),
    )
    recorder.event(
        "intervention_applied", item_id=prepared.item["item_id"],
        analysis_family=analysis_family, intervention_mode=mode,
        intervention_scope=spec.intervention_scope,
        direction_source_selector=source_selector,
        direction_source_position=source.index,
        localized_target_selector=target_selector,
        localized_target_position=None if target is None else target.index,
        applied_prompt_position_count=spec.applied_prompt_position_count,
        steer_generated=spec.steer_generated,
        max_injection_fraction=spec.max_injection_fraction,
        requested_strength=spec.requested_strength,
        effective_strength_after_cap=spec.effective_strength_after_cap,
        token_id=candidate.token_id, layer=candidate.layer,
    )
    generation = steering.generate_greedy(
        lens_model, prepared.input_ids, tokenizer,
        max_new_tokens=int(config["generation"]["max_choice_tokens"]), steering=spec,
    )
    scores = steering.score_label_pair(
        lens_model, prepared.input_ids, tokenizer, prepared.labels, spec
    )
    normalized, valid = protocol.normalize_choice(generation["raw"], prepared.labels)
    baseline_correct = bool(
        prepared.baseline_valid
        and prepared.baseline_normalized == prepared.expected_judgment
    )
    intervened_correct = bool(valid and normalized == prepared.expected_judgment)
    flipped = bool(
        prepared.baseline_valid and valid
        and prepared.baseline_normalized != normalized
    )
    if flipped and not baseline_correct and intervened_correct:
        flip_effect = "improved"
    elif flipped and baseline_correct and not intervened_correct:
        flip_effect = "worsened"
    elif flipped:
        flip_effect = "changed"
    else:
        flip_effect = "no_flip"
    baseline_margin = float(prepared.baseline_scores["margin"])
    intervention_margin = float(scores["margin"])
    baseline_oriented = oriented_margin(
        baseline_margin, prepared.labels, prepared.expected_judgment
    )
    intervened_oriented = oriented_margin(
        intervention_margin, prepared.labels, prepared.expected_judgment
    )
    candidate_score_position = source.index if target is None else target.index
    after_score = steering.candidate_score_after_steering(
        lens_model, lens, prepared.input_ids, candidate, candidate_score_position, spec
    )
    row = {
        "item_id": prepared.item["item_id"], "item_type": prepared.item["item_type"],
        "condition": prepared.condition, "domain": prepared.item.get("domain"),
        "difficulty": prepared.item.get("difficulty"), "analysis_family": analysis_family,
        "is_primary_estimand": is_primary_estimand, "attempt_order": attempt_order,
        "intervention_mode": mode, "intervention_scope": spec.intervention_scope,
        "direction_source_selector": source_selector,
        "direction_source_position": source.index,
        "localized_target_selector": target_selector,
        "localized_target_position": None if target is None else target.index,
        "applied_prompt_position_count": spec.applied_prompt_position_count,
        "steer_generated": spec.steer_generated,
        "max_injection_fraction": spec.max_injection_fraction,
        "requested_strength": spec.requested_strength,
        "effective_strength_after_cap": spec.effective_strength_after_cap,
        "feature_label": candidate.label, "feature_id": candidate.feature_id,
        "token_id": candidate.token_id, "direction_sha256": candidate.direction_sha256,
        "layer": candidate.layer, "layer_selection_reason": layer_selection_reason,
        "raw_rank": metric["raw_rank"], "word_filtered_rank": metric["word_filtered_rank"],
        "rank_policy": config["rank_policy"]["appearance_rank"],
        "baseline_output_raw": prepared.baseline_generation["raw"],
        "baseline_output_normalized": prepared.baseline_normalized,
        "intervened_output_raw": generation["raw"],
        "intervened_output_normalized": normalized,
        "baseline_valid": prepared.baseline_valid, "intervened_valid": valid,
        "labels": "/".join(prepared.labels),
        "expected_judgment": prepared.expected_judgment,
        "factual_correct": prepared.factual_correct,
        "baseline_judgment_correct": baseline_correct,
        "intervened_judgment_correct": intervened_correct,
        "baseline_margin": baseline_margin, "intervened_margin": intervention_margin,
        "delta_margin": intervention_margin - baseline_margin,
        "baseline_oriented_margin": baseline_oriented,
        "intervened_oriented_margin": intervened_oriented,
        "delta_oriented_margin": intervened_oriented - baseline_oriented,
        "candidate_score_before": metric["score"], "candidate_score_after": after_score,
        "flipped": flipped, "flip_effect": flip_effect,
        "generated_positions_steered": generation["hook_audit"]["generated_positions_steered"],
        "duration_seconds": time.perf_counter() - started,
    }
    recorder.csv("intervention_results", INTERVENTION_FIELDS, row)
    recorder.jsonl("raw_runs.jsonl", {
        "timestamp": utc_now(), "run_id": recorder.run_id,
        "item": prepared.item, "messages_before_output": prepared.messages,
        "candidate": candidate.metadata(), "steering": asdict(spec),
        "baseline_generation": prepared.baseline_generation,
        "baseline_scores": prepared.baseline_scores,
        "intervened_generation": generation, "intervened_scores": scores,
        "result": row,
    })
    recorder.event(
        "intervention_result", item_id=prepared.item["item_id"],
        analysis_family=analysis_family, attempt_order=attempt_order,
        requested_strength=requested_strength, flipped=flipped,
        flip_effect=flip_effect, delta_oriented_margin=row["delta_oriented_margin"],
    )
    return row


def _candidate_spec(config: dict[str, Any], token_id: int) -> dict[str, Any]:
    matches = [
        item for item in config["candidates"]
        if item.get("enabled", True) and int(item["token_id"]) == int(token_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one candidate specification for token {token_id}")
    return matches[0]


def adaptive_layer_plan(
    readout: dict[str, Any],
    candidate_specs: Sequence[dict[str, Any]],
    *,
    primary_token_id: int,
    primary_layer: int,
    threshold: int,
    maximum: int,
    eligible_layers: set[int] | None = None,
) -> list[tuple[dict[str, Any], int, str]]:
    """Pre-outcome fallback plan using only frozen filtered-rank visibility."""
    result = []
    for spec in candidate_specs:
        token_id = int(spec["token_id"])
        metrics = readout["candidate_metrics"][f"vocab_token:{token_id}"]
        visible = sorted(
            int(layer) for layer, metric in metrics.items()
            if int(metric["word_filtered_rank"]) <= threshold
            and (eligible_layers is None or int(layer) in eligible_layers)
            and not (token_id == primary_token_id and int(layer) == primary_layer)
        )
        if token_id != primary_token_id and primary_layer in visible:
            visible.remove(primary_layer)
            visible.insert(0, primary_layer)
        for layer in visible[:maximum]:
            reason = (
                "alternate_candidate_preferred_layer"
                if token_id != primary_token_id and layer == primary_layer
                else "word_filtered_appearance_fallback"
            )
            result.append((spec, layer, reason))
    return result


def primary_condition_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one global condition per nonzero polarity, never per local position."""
    primary = config["interventions"]["primary"]
    return [
        {
            "mode": "neuronpedia_global",
            "source_selector": primary["direction_source_selector"],
            "target_selector": None,
            "requested_strength": float(strength),
        }
        for strength in primary["strengths"]
        if float(strength) != 0.0
    ]


def execute_prepared_trial(
    recorder: Recorder,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    prepared: PreparedTrial,
    base_candidates: Sequence[steering.Candidate],
    config: dict[str, Any],
    *,
    run_localized: bool = True,
    run_adaptive: bool = True,
) -> dict[str, Any]:
    primary_config = config["interventions"]["primary"]
    primary_token = int(primary_config["candidate_token_id"])
    primary_layer = int(primary_config["layer"])
    primary_candidate = next(
        candidate for candidate in base_candidates
        if candidate.token_id == primary_token and candidate.layer == primary_layer
    )
    source_selector = str(primary_config["direction_source_selector"])
    source_readout = prepared.readouts[source_selector]
    primary_metric = source_readout["candidate_metrics"][primary_candidate.feature_id][str(primary_layer)]
    threshold = int(config["rank_policy"]["appearance_rank_threshold"])
    visible = int(primary_metric[config["rank_policy"]["appearance_rank"]]) <= threshold
    attempts: list[dict[str, Any]] = []
    primary_rows = []
    attempt_order = 1
    if visible:
        for condition in primary_condition_plan(config):
            row = run_intervention(
                recorder, lens_model, lens, tokenizer, prepared, primary_candidate,
                analysis_family="frozen_primary", is_primary_estimand=True,
                attempt_order=attempt_order, mode=condition["mode"],
                requested_strength=condition["requested_strength"],
                source_selector=condition["source_selector"],
                target_selector=condition["target_selector"],
                layer_selection_reason="frozen_primary_layer",
                config=config,
            )
            attempts.append(row)
            primary_rows.append(row)
            attempt_order += 1
        primary_status = (
            "completed_with_flip" if any(row["flipped"] for row in primary_rows)
            else "completed_no_flip"
        )
    else:
        primary_status = "skipped_frozen_visibility_rule"
        recorder.event(
            "primary_skipped", item_id=prepared.item["item_id"],
            token_id=primary_token, layer=primary_layer,
            raw_rank=primary_metric["raw_rank"],
            word_filtered_rank=primary_metric["word_filtered_rank"],
            threshold=threshold,
        )

    localized_config = config["interventions"]["localized_control"]
    if run_localized and localized_config.get("enabled", True):
        for target_selector in localized_config["positions"]:
            if target_selector not in prepared.positions:
                continue
            for strength in localized_config["strengths"]:
                if float(strength) == 0:
                    continue
                attempts.append(run_intervention(
                    recorder, lens_model, lens, tokenizer, prepared, primary_candidate,
                    analysis_family="localized_control", is_primary_estimand=False,
                    attempt_order=attempt_order, mode="single_position",
                    requested_strength=float(strength), source_selector=source_selector,
                    target_selector=target_selector,
                    layer_selection_reason="localized_primary_layer_control",
                    config=config,
                ))
                attempt_order += 1

    rescue_config = config["interventions"]["adaptive_rescue"]
    rescue_path = []
    rescue_success = None
    should_rescue = (
        run_adaptive and rescue_config.get("enabled", True)
        and not any(row["flipped"] for row in primary_rows)
    )
    if should_rescue:
        plan = adaptive_layer_plan(
            source_readout,
            [item for item in config["candidates"] if item.get("enabled", True)],
            primary_token_id=primary_token, primary_layer=primary_layer,
            threshold=threshold,
            maximum=int(rescue_config["max_appearance_layers_per_candidate"]),
            eligible_layers={int(layer) for layer in lens.jacobians},
        )
        for candidate_spec, layer, reason in plan:
            candidate = steering.resolve_candidate(
                candidate_spec, layer, tokenizer, lens_model, lens,
                recorder.run_dir / "checkpoints",
            )
            layer_rows = []
            for strength in rescue_config["strengths"]:
                row = run_intervention(
                    recorder, lens_model, lens, tokenizer, prepared, candidate,
                    analysis_family="adaptive_rescue", is_primary_estimand=False,
                    attempt_order=attempt_order, mode="neuronpedia_global",
                    requested_strength=float(strength), source_selector=source_selector,
                    target_selector=None, layer_selection_reason=reason, config=config,
                )
                attempts.append(row)
                layer_rows.append(row)
                attempt_order += 1
            rescue_path.append({
                "token_id": candidate.token_id, "layer": layer,
                "selection_reason": reason,
                "raw_rank": layer_rows[0]["raw_rank"],
                "word_filtered_rank": layer_rows[0]["word_filtered_rank"],
                "flipped": any(row["flipped"] for row in layer_rows),
            })
            flipped = next((row for row in layer_rows if row["flipped"]), None)
            if flipped is not None:
                rescue_success = flipped
                if rescue_config.get("stop_after_first_flip", True):
                    break
        adaptive_status = (
            "flip_found" if rescue_success is not None
            else ("no_visible_fallback" if not plan else "no_flip")
        )
    else:
        adaptive_status = "not_run_primary_flipped" if run_adaptive else "disabled_for_phase"
    recorder.jsonl("adaptive_paths.jsonl", {
        "timestamp": utc_now(), "run_id": recorder.run_id,
        "item_id": prepared.item["item_id"], "primary_status": primary_status,
        "primary_visible": visible, "primary_rows": len(primary_rows),
        "adaptive_status": adaptive_status, "path": rescue_path,
        "adaptive_attempts": sum(row["analysis_family"] == "adaptive_rescue" for row in attempts),
    })
    baseline_correct = bool(
        prepared.baseline_valid
        and prepared.baseline_normalized == prepared.expected_judgment
    )
    baseline_margin = float(prepared.baseline_scores["margin"])
    negative = next((row for row in primary_rows if row["requested_strength"] < 0), None)
    positive = next((row for row in primary_rows if row["requested_strength"] > 0), None)
    summary = {
        "item_id": prepared.item["item_id"], "item_type": prepared.item["item_type"],
        "condition": prepared.condition, "domain": prepared.item.get("domain"),
        "difficulty": prepared.item.get("difficulty"),
        "pre_meta_output_raw": None if prepared.pre_generation is None else prepared.pre_generation["raw"],
        "pre_meta_output_normalized": prepared.pre_normalized,
        "pre_meta_valid": prepared.pre_valid, "factual_answer": prepared.factual_answer,
        "answer_key": prepared.item["answer_key"],
        "factual_correct": prepared.factual_correct,
        "factual_response_invalid": prepared.factual_invalid,
        "scoring_method": prepared.factual_scoring["scoring_method"],
        "judgment_prompt": prepared.messages[-1]["content"],
        "judgment_labels": "/".join(prepared.labels),
        "expected_judgment": prepared.expected_judgment,
        "baseline_output_raw": prepared.baseline_generation["raw"],
        "baseline_output_normalized": prepared.baseline_normalized,
        "baseline_valid": prepared.baseline_valid,
        "baseline_judgment_correct": baseline_correct,
        "baseline_margin": baseline_margin,
        "baseline_oriented_margin": oriented_margin(
            baseline_margin, prepared.labels, prepared.expected_judgment
        ),
        "direction_source_selector": source_selector,
        "direction_source_position": prepared.positions[source_selector].index,
        "direction_source_token": prepared.positions[source_selector].token,
        "primary_candidate_token_id": primary_token, "primary_layer": primary_layer,
        "primary_raw_rank": primary_metric["raw_rank"],
        "primary_word_filtered_rank": primary_metric["word_filtered_rank"],
        "primary_visible": visible, "primary_status": primary_status,
        "primary_negative_flipped": None if negative is None else negative["flipped"],
        "primary_positive_flipped": None if positive is None else positive["flipped"],
        "adaptive_status": adaptive_status,
    }
    recorder.csv("trial_summary", SUMMARY_FIELDS, summary)
    recorder.event(
        "trial_finished", item_id=prepared.item["item_id"],
        item_type=prepared.item["item_type"], condition=prepared.condition,
        primary_status=primary_status, adaptive_status=adaptive_status,
    )
    return {"summary": summary, "attempts": attempts, "primary_rows": primary_rows}


def prepare_forced_trial(
    recorder: Recorder,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    base_candidates: Sequence[steering.Candidate],
    word_ids: torch.Tensor,
    config: dict[str, Any],
    item_id: str,
    prompt_text: str,
    labels: tuple[str, str],
    expected: str,
) -> PreparedTrial:
    item = {
        "item_id": item_id, "item_type": "forced_output", "condition": "forced_output",
        "domain": "formatting", "difficulty": "control", "prompt": prompt_text,
        "detail": "no_evaluation", "answer_key": expected,
    }
    messages = [{"role": "user", "content": prompt_text}]
    input_ids, rendered, offsets, positions = protocol.locate_judgment_positions(tokenizer, messages)
    recorder.jsonl("tokenizations.jsonl", {
        "timestamp": utc_now(), "run_id": recorder.run_id, "item_id": item_id,
        "rendered_chat": rendered, "token_ids": input_ids[0].tolist(),
        "offsets": offsets, "positions": {name: asdict(value) for name, value in positions.items()},
    })
    readouts = {}
    for name, position in positions.items():
        readout = steering.readout_across_layers(
            lens_model, lens, input_ids, position=position.index,
            candidates=base_candidates, top_k=int(config["readout_top_k"]),
            word_ids=word_ids,
            residual_path=(recorder.run_dir / "checkpoints" / "residuals" /
                           f"{item_id}_{name}.pt"),
        )
        readouts[name] = readout
        recorder.jsonl("readouts.jsonl", {
            "timestamp": utc_now(), "run_id": recorder.run_id,
            "item_id": item_id, "position": asdict(position), "readout": readout,
        })
    baseline_generation = steering.generate_greedy(
        lens_model, input_ids, tokenizer,
        max_new_tokens=int(config["generation"]["max_choice_tokens"]),
    )
    normalized, valid = protocol.normalize_choice(baseline_generation["raw"], labels)
    scores = steering.score_label_pair(lens_model, input_ids, tokenizer, labels)
    return PreparedTrial(
        item=item, condition="forced_output", messages=messages, input_ids=input_ids,
        positions=positions, readouts=readouts, labels=labels,
        expected_judgment=expected, factual_correct=expected == labels[0],
        factual_answer=None, factual_invalid=False,
        factual_scoring={"scoring_method": "forced_output_expected_label"},
        pre_generation=None, pre_normalized=None, pre_valid=None,
        baseline_generation=baseline_generation, baseline_normalized=normalized,
        baseline_valid=valid, baseline_scores=scores,
    )


def package_versions(names: Sequence[str]) -> dict[str, str | None]:
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def git_value(arguments: Sequence[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={REPO_ROOT}", *arguments],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(
    *,
    run_id: str,
    phase: str,
    config: dict[str, Any],
    dataset_path: Path,
    selected_count: int,
    args: argparse.Namespace,
    hf_model: Any | None = None,
    tokenizer: Any | None = None,
    lens_path: Path | None = None,
    lens_revision: str | None = None,
    lens_model: Any | None = None,
    candidates: Sequence[steering.Candidate] = (),
) -> dict[str, Any]:
    primary = config["interventions"]["primary"]
    runtime = None
    if hf_model is not None and lens_model is not None:
        runtime = {
            "architecture": type(hf_model).__name__,
            "model_type": getattr(hf_model.config.get_text_config(), "model_type", None),
            "n_layers": lens_model.n_layers, "d_model": lens_model.d_model,
            "parameters": sum(parameter.numel() for parameter in hf_model.parameters()),
            "input_device": str(lens_model.input_device),
            "cuda": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            runtime["gpu"] = torch.cuda.get_device_name(0)
    return {
        "run_id": run_id, "experiment_started_at": utc_now(),
        "experiment_ended_at": None,
        "status": "running" if hf_model is not None else "initializing",
        "phase": phase, "sample_count": selected_count,
        "model_checkpoint": args.model, "model_requested_revision": args.revision,
        "model_resolved_revision": getattr(getattr(hf_model, "config", None), "_commit_hash", None),
        "tokenizer_resolved_revision": (
            getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
            if tokenizer is not None else None
        ),
        "lens": {
            "repo": config["lens"]["repo"], "file": config["lens"]["file"],
            "requested_revision": config["lens"].get("revision", "main"),
            "resolved_revision": lens_revision,
            "local_path": None if lens_path is None else str(lens_path.resolve()),
        },
        "runtime": runtime, "python": sys.version, "platform": platform.platform(),
        "hostname": socket.gethostname(), "cuda_runtime": torch.version.cuda,
        "packages": package_versions((
            "torch", "transformers", "jlens", "huggingface-hub", "pandas", "matplotlib"
        )),
        "repository": {
            "commit": git_value(("rev-parse", "HEAD")),
            "status": git_value(("status", "--short")),
        },
        "dataset": {"path": str(dataset_path), "sha256": sha256_file(dataset_path)},
        "seed": args.seed, "dtype": args.dtype, "device_map": args.device_map,
        "decoding": {
            "method": "manual greedy full-prefix argmax", "use_cache": False,
            "thinking": False, **config["generation"],
        },
        "intervention_api": {
            "provider": "centralized local forward hook",
            "direction": "normalize(J_l.T @ lm_head.weight[token_id])",
            "primary_mode": "neuronpedia_global",
            "primary_scope": "all prompt positions and generated-token recomputations",
            "scale": "per-position residual L2 norm",
            "max_injection_fraction": primary["max_injection_fraction"],
            "steer_generated": primary["steer_generated"],
            "cache_policy": "disabled; complete prefix recomputed for every token",
            "direction_source_semantics": "readout selection only; not intervention target",
        },
        "analysis_contract": {
            "primary_family": "frozen_primary",
            "primary_unit": "sample",
            "excluded_from_primary_ci": ["localized_control", "adaptive_rescue"],
            "baseline_policy": "one shared generation and label score per sample",
        },
        "feature_metadata": [candidate.metadata() for candidate in candidates],
        "config": config,
    }


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def record_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def relocatable_gate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove location-only fields while preserving experimental settings."""
    normalized = json.loads(json.dumps(config, default=json_default))
    dataset = normalized.get("dataset")
    if isinstance(dataset, dict):
        dataset.pop("path", None)
    return normalized


def validate_gate_run(path: Path, expected_phase: str, config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = path.resolve() / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{expected_phase} gate has no manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_config = manifest.get("config")
    config_matches = (
        isinstance(stored_config, dict)
        and relocatable_gate_config(stored_config) == relocatable_gate_config(config)
    )
    if manifest.get("phase") != expected_phase or not config_matches:
        raise ValueError(f"{expected_phase} gate phase or configuration mismatch")
    current_dataset = resolve_repo_path(config["dataset"]["path"])
    stored_dataset_sha = manifest.get("dataset", {}).get("sha256")
    if not current_dataset.is_file() or stored_dataset_sha != sha256_file(current_dataset):
        raise ValueError(f"{expected_phase} gate dataset content mismatch")
    summary_name = "parity_summary.json" if expected_phase == "parity" else "pilot_summary.json"
    summary_path = path.resolve() / summary_name
    if not summary_path.is_file():
        raise ValueError(f"{expected_phase} gate has no {summary_name}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gate_key = "research_gate_passed" if expected_phase == "parity" else "passed"
    if not summary.get(gate_key):
        raise ValueError(f"{expected_phase} gate did not pass")
    return {
        "run_id": manifest["run_id"], "path": str(path.resolve()),
        "manifest_sha256": sha256_file(manifest_path), "summary": summary,
    }


def parity_summary(
    run_dir: Path,
    results: dict[str, dict[str, Any]],
    exported: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    parity = config["parity"]
    item73 = results["73"]
    item72 = results["72"]
    negative73 = next(row for row in item73["primary_rows"] if row["requested_strength"] < 0)
    positive73 = next(row for row in item73["primary_rows"] if row["requested_strength"] > 0)
    positive72 = next(row for row in item72["primary_rows"] if row["requested_strength"] > 0)
    exported_baseline, _ = protocol.normalize_choice(
        exported["exported_baseline"], ("PASS", "FAIL")
    )
    exported_intervention, _ = protocol.normalize_choice(
        exported["exported_intervention"], ("PASS", "FAIL")
    )
    checks = {
        "exact_prompt_token_count": exported["prompt_len"] == int(parity["expected_prompt_tokens"]),
        "source_question_mark_position": exported["source_position"] == int(parity["expected_source_position"]),
        "baseline_matches_export": item73["summary"]["baseline_output_normalized"] == exported_baseline,
        "primary_candidate_97817": all(row["token_id"] == 97817 for row in item73["primary_rows"]),
        "primary_layer_40": all(row["layer"] == 40 for row in item73["primary_rows"]),
        "global_scope_logged": all(row["intervention_mode"] == "neuronpedia_global" for row in item73["primary_rows"]),
        "generated_steering_enabled": all(bool(row["steer_generated"]) for row in item73["primary_rows"]),
        "all_eligible_prompt_positions_targeted": negative73["applied_prompt_position_count"] > 200,
        "candidate_moves_under_exported_polarity": negative73["candidate_score_after"] < negative73["candidate_score_before"],
        "exported_polarity_increases_pass_margin": negative73["delta_margin"] >= float(parity["minimum_pass_margin_increase"]),
        "local_positive_polarity_flips_item73": bool(positive73["flipped"]),
        "item72_stays_pass_under_positive": (
            item72["summary"]["baseline_output_normalized"] == "PASS"
            and positive72["intervened_output_normalized"] == "PASS"
        ),
    }
    strict = (
        negative73["intervened_output_normalized"] == exported_intervention
        and exported_intervention == str(parity["expected_exported_intervention"])
    )
    research_keys = [key for key in checks if key != "local_positive_polarity_flips_item73"]
    result = {
        "checks": checks,
        "strict_backend_parity": strict,
        "strict_backend_parity_status": "passed" if strict else "incomplete_boundary_mismatch",
        "research_gate_passed": all(checks[key] for key in research_keys)
        and checks["local_positive_polarity_flips_item73"],
        "item73_exported_polarity": negative73,
        "item73_local_effective_polarity": positive73,
        "item72_positive_control": positive72,
        "exported_baseline": exported_baseline,
        "exported_intervention": exported_intervention,
    }
    (run_dir / "parity_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    return result


def pilot_summary(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    with (run_dir / "trial_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        summaries = list(csv.DictReader(handle))
    with (run_dir / "intervention_results.csv").open("r", encoding="utf-8", newline="") as handle:
        attempts = list(csv.DictReader(handle))
    normalized = [row["baseline_output_normalized"] for row in summaries]
    difficulties = {row["difficulty"] for row in summaries}
    correctness = {row["factual_correct"].lower() == "true" for row in summaries}
    families = {row["analysis_family"] for row in attempts}
    zero_rows = [row for row in attempts if float(row["requested_strength"]) == 0.0]
    primary_keys = [
        (row["item_id"], row["requested_strength"]) for row in attempts
        if row["analysis_family"] == "frozen_primary"
    ]
    checks = {
        "configured_sample_count": len(summaries) == len(config["pilot"]["item_ids"]),
        "all_baselines_valid": all(row["baseline_valid"].lower() == "true" for row in summaries),
        "no_missing_correctness": all(row["factual_correct"] != "" for row in summaries),
        "baseline_pass_and_fail": {"PASS", "FAIL"}.issubset(set(normalized)),
        "easy_and_hard": {"easy", "hard"}.issubset(difficulties),
        "factually_correct_and_incorrect": correctness == {True, False},
        "global_and_localized_distinct": {"frozen_primary", "localized_control"}.issubset(families),
        "one_baseline_per_sample": len({row["item_id"] for row in summaries}) == len(summaries),
        "no_zero_intervention_rows": not zero_rows,
        "primary_rows_unique": len(primary_keys) == len(set(primary_keys)),
        "errors_empty": record_count(run_dir / "errors.jsonl") == 0,
    }
    result = {"checks": checks, "passed": all(checks.values())}
    (run_dir / "pilot_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("validate", "parity", "pilot", "full"), default="validate")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device-map", choices=("cuda", "auto", "cpu"))
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--hf-cache-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--parity-run", type=Path)
    parser.add_argument("--pilot-run", type=Path)
    parser.add_argument("--skip-gate-requirement", action="store_true")
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--seed", type=int)
    return parser


def _configure_args(args: argparse.Namespace, config: dict[str, Any]) -> None:
    model = config["model"]
    args.model = args.model or model["id"]
    args.revision = args.revision or model.get("revision", "main")
    args.dtype = args.dtype or model.get("dtype", "bfloat16")
    args.device_map = args.device_map or model.get("device_map", "cuda")
    args.seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    config["seed"] = args.seed


def _write_run_readme(run_dir: Path, manifest: dict[str, Any]) -> None:
    (run_dir / "README_run.md").write_text(
        f"""# Global J-Lens run `{manifest['run_id']}`

Status: {manifest['status']}

The baseline is computed once per sample. `frozen_primary` is the only family
used for primary confidence intervals. `localized_control` and
`adaptive_rescue` are separately labelled descriptive families.

Global effects establish a distributed causal intervention effect, not a
higher-order representation M(P).
""",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    _configure_args(args, config)
    dataset_path = resolve_repo_path(config["dataset"]["path"])
    rows = protocol.load_dataset_rows(dataset_path)
    static = validate_config(config, rows)
    print(json.dumps(static, indent=2, ensure_ascii=False))
    if args.phase == "validate":
        return 0
    if args.device_map in {"cuda", "auto"} and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Qwen3.6-27B; use CPU only with a small test model")
    if args.device_map == "cpu" and not args.allow_cpu:
        raise SystemExit("pass --allow-cpu explicitly for a small debug model")

    gate_evidence = None
    if args.phase == "pilot" and not args.skip_gate_requirement:
        if args.parity_run is None:
            raise SystemExit("--phase pilot requires --parity-run PATH")
        gate_evidence = validate_gate_run(args.parity_run, "parity", config)
    if args.phase == "full" and not args.skip_gate_requirement:
        if args.pilot_run is None:
            raise SystemExit("--phase full requires --pilot-run PATH")
        gate_evidence = validate_gate_run(args.pilot_run, "pilot", config)

    by_id = {row["item_id"]: row for row in rows}
    if args.phase == "parity":
        selected = [by_id[str(item)] for item in config["parity"]["item_ids"]]
    elif args.phase == "pilot":
        selected = [by_id[str(item)] for item in config["pilot"]["item_ids"]]
    else:
        selected = list(rows)

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
        run_id = make_run_id(args.model)
        run_dir = prepare_run_dir(args.output_root.resolve(), run_id)
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    recorder = Recorder(run_dir, run_id, resume=resume)
    initial_errors = record_count(run_dir / "errors.jsonl")
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
    write_manifest(run_dir, manifest)
    _write_run_readme(run_dir, manifest)

    try:
        transformers, jlens = require_runtime()
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        hf_model, tokenizer = load_model(args, transformers)
        lens_model = jlens.from_hf(hf_model, tokenizer, compile=False)
        lens, lens_path, lens_revision = load_lens(jlens, config, args.hf_cache_dir)
        if lens.d_model != lens_model.d_model:
            raise RuntimeError("lens/model width mismatch")
        placement = place_lens(lens, lens_model) if config["lens"].get("resident_jacobians", True) else {}
        primary_layer = int(config["interventions"]["primary"]["layer"])
        base_candidates = [
            steering.resolve_candidate(
                spec, primary_layer, tokenizer, lens_model, lens, run_dir / "checkpoints"
            )
            for spec in config["candidates"] if spec.get("enabled", True)
        ]
        filtered_ids = steering.word_token_ids(tokenizer)
        manifest = build_manifest(
            run_id=run_id, phase=args.phase, config=config, dataset_path=dataset_path,
            selected_count=len(selected), args=args, hf_model=hf_model,
            tokenizer=tokenizer, lens_path=lens_path, lens_revision=lens_revision,
            lens_model=lens_model, candidates=base_candidates,
        )
        manifest["lens"]["matrix_placement"] = placement
        manifest["rank_policy"] = {
            **config["rank_policy"], "word_filtered_vocabulary_size": len(filtered_ids)
        }
        manifest["gate_evidence"] = gate_evidence
        manifest["gate_requirement_bypassed"] = bool(args.skip_gate_requirement)
        manifest["resumed"] = resume
        if resume:
            manifest["experiment_started_at"] = old_manifest["experiment_started_at"]
            manifest["resume_count"] = int(old_manifest.get("resume_count", 0)) + 1
        write_manifest(run_dir, manifest)
        _write_run_readme(run_dir, manifest)
        recorder.event("model_loaded", runtime=manifest["runtime"], lens=manifest["lens"])

        finished = completed_items(run_dir / "events.jsonl")
        results: dict[str, dict[str, Any]] = {}
        exported = None
        if args.phase == "parity":
            exported = protocol.load_neuronpedia_export(
                resolve_repo_path(config["parity"]["export_path"]), config["parity"]
            )
        for item in selected:
            if item["item_id"] in finished:
                recorder.logger.info("skip completed item=%s", item["item_id"])
                continue
            try:
                prepared = prepare_trial(
                    recorder, lens_model, lens, tokenizer, item, base_candidates,
                    filtered_ids, config,
                    exported=exported if args.phase == "parity" and item["item_id"] == "73" else None,
                )
                results[item["item_id"]] = execute_prepared_trial(
                    recorder, lens_model, lens, tokenizer, prepared, base_candidates, config,
                    run_localized=args.phase != "parity" and not args.no_controls,
                    run_adaptive=args.phase != "parity",
                )
            except Exception as exc:
                recorder.error(item["item_id"], "dataset_trial", exc)

        phase_gate = None
        if args.phase == "parity":
            if set(results) != {"73", "72"} or exported is None:
                raise RuntimeError("parity phase did not complete both required items")
            phase_gate = parity_summary(run_dir, results, exported, config)
        elif args.phase == "pilot":
            phase_gate = pilot_summary(run_dir, config)
        elif not args.no_controls:
            for item_id, prompt_text, labels, expected in FORCED_OUTPUTS[
                : int(config["controls"]["forced_output_items"])
            ]:
                if item_id in completed_items(run_dir / "events.jsonl"):
                    continue
                try:
                    recorder.event("trial_started", item_id=item_id, item_type="forced_output", condition="forced_output")
                    prepared = prepare_forced_trial(
                        recorder, lens_model, lens, tokenizer, base_candidates,
                        filtered_ids, config, item_id, prompt_text, labels, expected,
                    )
                    execute_prepared_trial(
                        recorder, lens_model, lens, tokenizer, prepared, base_candidates,
                        config, run_localized=True, run_adaptive=False,
                    )
                except Exception as exc:
                    recorder.error(item_id, "forced_output", exc)

        from .analysis import analyze

        plot_status = analyze(run_dir)
        new_errors = record_count(run_dir / "errors.jsonl") - initial_errors
        if new_errors:
            status = "completed_with_errors"
        elif args.phase == "parity" and phase_gate and not phase_gate["research_gate_passed"]:
            status = "parity_gate_failed"
        elif args.phase == "pilot" and phase_gate and not phase_gate["passed"]:
            status = "pilot_gate_failed"
        else:
            status = "completed"
        manifest["status"] = status
        manifest["logged_error_count"] = record_count(run_dir / "errors.jsonl")
        manifest["experiment_ended_at"] = utc_now()
        manifest["plot_status"] = plot_status
        if args.phase == "parity" and phase_gate:
            manifest["strict_backend_parity"] = phase_gate["strict_backend_parity"]
        write_manifest(run_dir, manifest)
        _write_run_readme(run_dir, manifest)
        print(f"Run directory: {run_dir}")
        return 0 if status == "completed" else 2
    except Exception as exc:
        recorder.error(None, "experiment", exc)
        manifest["status"] = "failed"
        manifest["experiment_ended_at"] = utc_now()
        manifest["fatal_error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_manifest(run_dir, manifest)
        _write_run_readme(run_dir, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
