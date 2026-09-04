"""Fail-closed runner for the complete process-sensitive replay protocol."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import logging
import math
import os
import platform
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from experiments.higher_v_readout_global.runner import load_lens, load_model
from experiments.higher_v_readout_global.steering import (
    candidate_direction,
    word_token_ids,
)

from .answer_bank import discover_answer
from .analysis import (
    analyze_candidate_effects,
    generate_required_plots,
    generate_support_match_plot,
    json_safe,
)
from .discovery import (
    DISCOVERY_CANDIDATE_CONDITIONS,
    alpha_grid_diagnostics,
    beta_grid_diagnostics,
    candidate_ranking_row,
    measure_strength_grid_item,
    rank_candidate_grid,
    select_discovery_alpha,
    select_discovery_strengths,
)
from .protocol import (
    GateStatus,
    allocate_discovery_split,
    assert_phase_prerequisites,
    canonical_json,
    direct_factual_question,
    load_config,
    load_dataset,
    gate_path,
    item_support_matched,
    read_gate,
    sha256_file,
    sha256_json,
    support_match_summary,
    validate_frozen_protocol,
    validate_config,
    write_gate,
)
from .replay import QwenReplayAdapter, verify_thinking_disabled
from .smoke import run_smoke_item, summarize_smoke


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_CONFIG = PACKAGE_DIR / "experiment_config.json"
SUPPORTED_PHASES = (
    "validate",
    "answer_bank",
    "pre_discovery_smoke",
    "discovery",
    "freeze",
    "post_freeze_smoke",
    "heldout",
    "analyze",
)
LOG_FILES = (
    "events.jsonl", "raw_runs.jsonl", "process_interventions.jsonl",
    "tokenizations.jsonl", "jlens_readouts.jsonl", "state_audits.jsonl",
    "errors.jsonl",
)
RUNTIME_PACKAGE_NAMES = ("torch", "transformers", "jlens", "huggingface-hub")
BASE_IDENTITY_HASH_KEYS = (
    "config",
    "dataset",
    "scientific_protocol",
    "code",
    "model_spec",
    "lens_spec",
    "model_revision",
    "tokenizer_revision",
    "lens_revision",
    "lens_sha256",
    "runtime_packages",
)


class PhaseGateFailure(RuntimeError):
    """A fail-closed phase result whose diagnostics must survive runner exit."""

    def __init__(
        self,
        status: str,
        reason: str,
        measurements: Mapping[str, Any],
    ) -> None:
        super().__init__(reason)
        self.status = status
        self.measurements = dict(measurements)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, default=str) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def initialize_run_dir(run_dir: Path, config_path: Path, config: Mapping[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in LOG_FILES:
        (run_dir / name).touch(exist_ok=True)
    resolved = run_dir / "config.json"
    payload = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    if resolved.exists() and resolved.read_text(encoding="utf-8") != payload:
        raise RuntimeError("run directory contains a different resolved config")
    resolved.write_text(payload, encoding="utf-8")
    protocol_source = REPO_ROOT / str(config["scientific_protocol"])
    frozen_copy = run_dir / "scientific_protocol.md"
    if frozen_copy.exists() and sha256_file(frozen_copy) != sha256_file(protocol_source):
        raise RuntimeError("run directory contains a different scientific protocol")
    if not frozen_copy.exists():
        shutil.copyfile(protocol_source, frozen_copy)


def begin_phase_once(run_dir: Path, phase: str) -> Path:
    directory = run_dir / phase
    started = directory / "phase_started.json"
    if started.exists() or gate_path(run_dir, phase).exists():
        raise RuntimeError(
            f"phase {phase} already started in this campaign; partial or completed phase outputs cannot be reused"
        )
    directory.mkdir(parents=True, exist_ok=True)
    started.write_text(
        json.dumps({"phase": phase, "started_at": utc_now()}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return started


def code_hash() -> str:
    files = sorted(path for path in PACKAGE_DIR.glob("*.py"))
    return sha256_json({str(path.name): sha256_file(path) for path in files})


def campaign_hashes(config_path: Path, config: Mapping[str, Any]) -> dict[str, str]:
    dataset_path = REPO_ROOT / str(config["dataset"]["path"])
    protocol_path = REPO_ROOT / str(config["scientific_protocol"])
    packages = package_versions(RUNTIME_PACKAGE_NAMES)
    return {
        "config": sha256_file(config_path),
        "dataset": sha256_file(dataset_path),
        "scientific_protocol": sha256_file(protocol_path),
        "code": code_hash(),
        "model_spec": sha256_json(config["model"]),
        "lens_spec": sha256_json(config["lens"]),
        "model_revision": str(config["model"]["revision"]),
        "tokenizer_revision": str(config["model"]["tokenizer_revision"]),
        "lens_revision": str(config["lens"]["revision"]),
        "lens_sha256": str(config["lens"]["sha256"]),
        "runtime_packages": sha256_json(packages),
    }


def combined_protocol_hash(hashes: Mapping[str, str]) -> str:
    return sha256_json({key: hashes[key] for key in BASE_IDENTITY_HASH_KEYS})


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def write_manifest(
    run_dir: Path,
    *,
    phase: str,
    status: str,
    config: Mapping[str, Any],
    hashes: Mapping[str, str],
    runtime: Mapping[str, Any] | None = None,
) -> None:
    path = run_dir / "run_manifest.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    manifest = {
        **previous,
        "experiment_name": "process_sensitive_replay",
        "campaign_created_at": previous.get("campaign_created_at", utc_now()),
        "updated_at": utc_now(),
        "active_phase": phase,
        "status": status,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": package_versions(RUNTIME_PACKAGE_NAMES),
        "hashes": dict(hashes),
        "runtime": previous.get("runtime") if runtime is None else dict(runtime),
        "config": dict(config),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def require_cuda_host(config: Mapping[str, Any]) -> None:
    if config["smoke"].get("require_cuda_for_real_model", True) and not torch.cuda.is_available():
        raise RuntimeError(
            "real Qwen3.6-27B phases are disabled: this environment has CPU-only PyTorch; "
            "use the CUDA host from the prior experiments"
        )


def load_runtime(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    *,
    require_lens_checkpoint: bool = True,
) -> tuple[Any, Any, Any, Any | None, QwenReplayAdapter, dict[str, Any]]:
    require_cuda_host(config)
    import jlens
    import transformers

    model_args = argparse.Namespace(
        model=config["model"]["id"],
        revision=config["model"]["revision"],
        hf_cache_dir=args.hf_cache_dir,
        dtype=config["model"]["dtype"],
        device_map=config["model"]["device_map"],
    )
    hf_model, tokenizer = load_model(model_args, transformers)
    lens_model = jlens.from_hf(hf_model, tokenizer, compile=False, force_bos=False)
    lens = None
    lens_path = None
    lens_revision = None
    if require_lens_checkpoint:
        lens, lens_path, lens_revision = load_lens(jlens, dict(config), args.hf_cache_dir)
    adapter = QwenReplayAdapter(hf_model, lens_model)
    text_config = hf_model.config.get_text_config()
    expected = config["layers"]
    if type(hf_model).__name__ != "Qwen3_5ForConditionalGeneration":
        raise RuntimeError(f"unexpected model architecture {type(hf_model).__name__}")
    if int(text_config.num_hidden_layers) != int(expected["expected_model_layers"]):
        raise RuntimeError("Qwen layer count changed")
    if int(text_config.hidden_size) != int(expected["expected_hidden_size"]):
        raise RuntimeError("Qwen hidden size changed")
    process_layer = int(expected["process"])
    if text_config.layer_types[process_layer] != expected["expected_process_layer_type"]:
        raise RuntimeError("configured process layer is not the expected full-attention layer")
    for alternative_layer in expected["alternative_candidates"]:
        if (
            text_config.layer_types[int(alternative_layer)]
            != expected["expected_alternative_layer_type"]
        ):
            raise RuntimeError(
                f"configured alternative layer {alternative_layer} is not the expected "
                f"{expected['expected_alternative_layer_type']} block"
            )
    if lens is not None:
        missing = sorted(set(int(value) for value in expected["readout"]) - set(lens.source_layers))
        if missing:
            raise RuntimeError(f"J-Lens checkpoint lacks required layers {missing}")
        if lens.d_model != lens_model.d_model:
            raise RuntimeError("J-Lens/model residual widths differ")
    if lens is not None and config["lens"].get("resident_jacobians", True):
        for layer in lens.source_layers:
            device = next(lens_model.layers[layer].parameters()).device
            lens.jacobians[layer] = lens.jacobians[layer].to(device)
    expected_model_revision = str(config["model"]["revision"])
    expected_tokenizer_revision = str(config["model"]["tokenizer_revision"])
    model_revision = getattr(hf_model.config, "_commit_hash", None)
    if model_revision != expected_model_revision:
        raise RuntimeError(
            "resolved model revision differs from the frozen commit: "
            f"expected={expected_model_revision} observed={model_revision}"
        )
    detected_tokenizer_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if (
        detected_tokenizer_revision is not None
        and detected_tokenizer_revision != expected_tokenizer_revision
    ):
        raise RuntimeError(
            "resolved tokenizer revision differs from the frozen commit: "
            f"expected={expected_tokenizer_revision} observed={detected_tokenizer_revision}"
        )
    lens_sha256 = None if lens_path is None else sha256_file(Path(lens_path))
    if lens is not None:
        if lens_revision != str(config["lens"]["revision"]):
            raise RuntimeError("resolved J-Lens revision differs from the frozen commit")
        if lens_sha256 != str(config["lens"]["sha256"]):
            raise RuntimeError(
                "resolved J-Lens file SHA-256 differs from the frozen checksum"
            )
    runtime = {
        "architecture": type(hf_model).__name__,
        "model_type": text_config.model_type,
        "num_hidden_layers": text_config.num_hidden_layers,
        "hidden_size": text_config.hidden_size,
        "process_layer_type": text_config.layer_types[process_layer],
        "alternative_layer_types": {
            str(int(layer)): text_config.layer_types[int(layer)]
            for layer in expected["alternative_candidates"]
        },
        "input_device": str(lens_model.input_device),
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
        "lens_path": None if lens_path is None else str(lens_path),
        "lens_revision": lens_revision,
        "lens_source_layers": None if lens is None else list(lens.source_layers),
        "jlens_force_bos": False,
        "model_resolved_revision": model_revision,
        # AutoTokenizer does not consistently expose _commit_hash. The exact
        # immutable revision passed to from_pretrained remains authoritative;
        # a conflicting exposed revision is rejected above.
        "tokenizer_resolved_revision": (
            detected_tokenizer_revision or expected_tokenizer_revision
        ),
        "tokenizer_revision_source": (
            "tokenizer_metadata"
            if detected_tokenizer_revision is not None
            else "frozen_from_pretrained_argument"
        ),
        "lens_sha256": lens_sha256,
        "runtime_packages": package_versions(RUNTIME_PACKAGE_NAMES),
    }
    return hf_model, tokenizer, lens_model, lens, adapter, runtime


def _write_trial_summary(run_dir: Path, records: list[Mapping[str, Any]]) -> None:
    path = run_dir / "trial_summary.csv"
    fields = [
        "item_id", "phase", "clean_support", "targeted_drop",
        "alternative_drop", "random_drop", "alternative_random_drop",
        "reset_parity", "cache_integrity",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "item_id": record["item_id"],
                "phase": record["phase"],
                "clean_support": record["support"]["clean"],
                "targeted_drop": record["support"]["targeted_drop"],
                "alternative_drop": record["support"]["alternative_drop"],
                "random_drop": record["support"]["random_drop"],
                "alternative_random_drop": record["support"][
                    "alternative_random_drop"
                ],
                "reset_parity": record["checks"]["reset_parity"],
                "cache_integrity": record["checks"]["hybrid_cache_integrity"],
            })


def _write_candidate_scores(run_dir: Path, records: list[Mapping[str, Any]]) -> None:
    path = run_dir / "candidate_scores.csv"
    fields = ["item_id", "phase", "condition", "branch", "layer", "token_id", "score", "raw_rank"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for condition, branches in record["meta"].items():
                for branch, branch_data in branches.items():
                    for layer, layer_data in branch_data["jlens"].items():
                        for token_id, token_data in layer_data["explicit"].items():
                            writer.writerow({
                                "item_id": record["item_id"],
                                "phase": record["phase"],
                                "condition": condition,
                                "branch": branch,
                                "layer": layer,
                                "token_id": token_id,
                                "score": token_data["score"],
                                "raw_rank": token_data["raw_rank"],
                            })


def _append_frozen_discovery_candidate_scores(
    path: Path,
    vocab_scores: torch.Tensor,
    *,
    item_ids: list[str],
    condition_names: tuple[str, ...],
    branch_names: tuple[str, ...],
    layers: list[int],
    candidates: list[Mapping[str, Any]],
) -> None:
    fields = ["item_id", "phase", "condition", "branch", "layer", "token_id", "score", "raw_rank"]
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        for candidate in candidates:
            layer = int(candidate["layer"])
            layer_offset = layers.index(layer)
            token_id = int(candidate["token_id"])
            for item_offset, item_id in enumerate(item_ids):
                for condition_offset, condition in enumerate(condition_names):
                    for branch_offset, branch in enumerate(branch_names):
                        vector = vocab_scores[
                            item_offset, condition_offset, branch_offset, layer_offset
                        ].float()
                        score = vector[token_id]
                        writer.writerow({
                            "item_id": item_id,
                            "phase": "discovery",
                            "condition": condition,
                            "branch": branch,
                            "layer": layer,
                            "token_id": token_id,
                            "score": float(score.item()),
                            "raw_rank": int((vector > score).sum().item()) + 1,
                        })


def _log_smoke_record(run_dir: Path, record: Mapping[str, Any]) -> None:
    stamp = utc_now()
    append_jsonl(run_dir / "raw_runs.jsonl", {"timestamp": stamp, **record})
    append_jsonl(run_dir / "tokenizations.jsonl", {
        "timestamp": stamp,
        "item_id": record["item_id"],
        "hashes": record["hashes"],
        "gradient_alignment": record["gradient"],
    })
    for row in record["interventions"]:
        append_jsonl(run_dir / "process_interventions.jsonl", {
            "timestamp": stamp, "item_id": record["item_id"], **row,
        })
    append_jsonl(run_dir / "state_audits.jsonl", {
        "timestamp": stamp, "item_id": record["item_id"],
        "conditions": record["state_audits"],
        "checks": record["checks"],
    })
    for condition, branches in record["meta"].items():
        for branch, value in branches.items():
            append_jsonl(run_dir / "jlens_readouts.jsonl", {
                "timestamp": stamp, "item_id": record["item_id"],
                "condition": condition, "branch": branch,
                "question_position": value["question_position"],
                "question_token_id": value["question_token_id"],
                "readout": value["jlens"],
            })
    append_jsonl(run_dir / "events.jsonl", {
        "timestamp": stamp,
        "event_type": (
            "candidate_discovery_trial_completed"
            if record["phase"] == "discovery"
            else "smoke_trial_completed"
        ),
        "phase": record["phase"], "item_id": record["item_id"],
    })


def run_validate(
    run_dir: Path,
    config_path: Path,
    config: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    rows = load_dataset(REPO_ROOT / config["dataset"]["path"], config["dataset"]["item_types"])
    measurements = validate_config(config, rows)
    gate = GateStatus(
        phase="validate", status="passed",
        protocol_hash=combined_protocol_hash(hashes),
        input_hashes=dict(hashes), measurements=measurements,
    )
    write_gate(run_dir, gate)
    return measurements


def run_answer_bank_phase(
    args: argparse.Namespace,
    run_dir: Path,
    config: Mapping[str, Any],
    hashes: dict[str, str],
) -> dict[str, Any]:
    phase = "answer_bank"
    assert_phase_prerequisites(
        run_dir, phase, protocol_hash=combined_protocol_hash(hashes),
        required_input_hashes=hashes,
    )
    answer_bank_path = run_dir / "answer_bank.jsonl"
    if answer_bank_path.exists() and answer_bank_path.stat().st_size:
        raise RuntimeError("answer_bank.jsonl already contains data; start a new campaign rather than appending")
    _, tokenizer, lens_model, lens, adapter, runtime = load_runtime(
        args, config, require_lens_checkpoint=False
    )
    write_manifest(run_dir, phase=phase, status="running", config=config, hashes=hashes, runtime=runtime)
    rows = load_dataset(REPO_ROOT / config["dataset"]["path"], config["dataset"]["item_types"])
    thinking_examples = []
    for item_type in config["dataset"]["item_types"]:
        row = next(candidate for candidate in rows if candidate["item_type"] == item_type)
        check = verify_thinking_disabled(
            tokenizer,
            [{"role": "user", "content": direct_factual_question(row)}],
        )
        thinking_examples.append({
            "item_id": str(row["item_id"]),
            "item_type": str(row["item_type"]),
            "question_token_hash": sha256_json(
                tokenizer(
                    direct_factual_question(row), add_special_tokens=False
                )["input_ids"]
            ),
            **check,
        })
    thinking_path = run_dir / phase / "thinking_mode_verification.json"
    thinking_path.write_text(
        json.dumps({"passed": True, "examples": thinking_examples}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hashes["thinking_mode_verification"] = sha256_file(thinking_path)
    append_jsonl(run_dir / "events.jsonl", {
        "timestamp": utc_now(), "event_type": "thinking_mode_verified",
        "item_ids": [example["item_id"] for example in thinking_examples],
        "artifact_sha256": hashes["thinking_mode_verification"],
    })
    records = []
    for row in rows:
        record = discover_answer(
            adapter, tokenizer, row,
            max_answer_tokens=int(config["generation"]["max_answer_tokens"]),
        )
        records.append(record)
        append_jsonl(answer_bank_path, record)
        append_jsonl(run_dir / "events.jsonl", {
            "timestamp": utc_now(), "event_type": "answer_discovered",
            "item_id": record["item_id"], "invalid": record["invalid"],
        })
    split = allocate_discovery_split(records, config)
    split_path = run_dir / "split_manifest.json"
    split_payload = {
        **split,
        "seed": config["split"]["seed"],
        "answer_bank_sha256": sha256_file(answer_bank_path),
        "dataset_sha256": hashes["dataset"],
        "thinking_mode_verification_sha256": hashes["thinking_mode_verification"],
    }
    split_path.write_text(json.dumps(split_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes.update({
        "answer_bank": sha256_file(answer_bank_path),
        "split_manifest": sha256_file(split_path),
    })
    measurements = {
        "answers": len(records),
        "invalid_answers": sum(bool(record["invalid"]) for record in records),
        "discovery_items": len(split["discovery_item_ids"]),
        "heldout_items": len(split["heldout_item_ids"]),
        "excluded_invalid_items": len(split["excluded_invalid_item_ids"]),
        "thinking_mode_examples_verified": len(thinking_examples),
    }
    write_gate(run_dir, GateStatus(
        phase=phase, status="passed", protocol_hash=combined_protocol_hash(hashes),
        input_hashes=dict(hashes), measurements=measurements,
    ))
    return measurements


def _load_campaign_inputs(run_dir: Path, hashes: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    answer_path = run_dir / "answer_bank.jsonl"
    split_path = run_dir / "split_manifest.json"
    thinking_path = run_dir / "answer_bank" / "thinking_mode_verification.json"
    hashes["answer_bank"] = sha256_file(answer_path)
    hashes["split_manifest"] = sha256_file(split_path)
    hashes["thinking_mode_verification"] = sha256_file(thinking_path)
    return read_jsonl(answer_path), json.loads(split_path.read_text(encoding="utf-8"))


def _load_pre_discovery_smoke_hash(run_dir: Path, hashes: dict[str, str]) -> None:
    directory = run_dir / "pre_discovery_smoke"
    hashes["pre_discovery_smoke_trials"] = sha256_file(directory / "trials.jsonl")
    hashes["pre_discovery_smoke_report"] = sha256_file(directory / "smoke_report.json")
    hashes["pre_discovery_smoke_trial_summary"] = sha256_file(
        directory / "trial_summary.csv"
    )
    hashes["pre_discovery_smoke_candidate_scores"] = sha256_file(
        directory / "candidate_scores.csv"
    )


def _load_post_freeze_smoke_hash(run_dir: Path, hashes: dict[str, str]) -> None:
    directory = run_dir / "post_freeze_smoke"
    hashes["post_freeze_smoke_trials"] = sha256_file(directory / "trials.jsonl")
    hashes["post_freeze_smoke_report"] = sha256_file(directory / "smoke_report.json")
    hashes["post_freeze_smoke_trial_summary"] = sha256_file(
        directory / "trial_summary.csv"
    )
    hashes["post_freeze_smoke_candidate_scores"] = sha256_file(
        directory / "candidate_scores.csv"
    )


def _load_heldout_hashes(run_dir: Path, hashes: dict[str, str]) -> None:
    for name, path in (
        ("heldout_trials", run_dir / "heldout" / "trials.jsonl"),
        ("heldout_trial_summary", run_dir / "heldout" / "trial_summary.csv"),
        ("heldout_candidate_scores", run_dir / "heldout" / "candidate_scores.csv"),
        ("heldout_support_match", run_dir / "heldout_support_match.csv"),
        (
            "heldout_support_match_summary",
            run_dir / "heldout" / "support_match_summary.json",
        ),
        (
            "heldout_support_match_plot",
            run_dir / "heldout" / "support_match_diagnostic.png",
        ),
        ("heldout_effects", run_dir / "heldout_effects.csv"),
    ):
        hashes[name] = sha256_file(path)


def _load_discovery_hashes(run_dir: Path, hashes: dict[str, str]) -> None:
    discovery_dir = run_dir / "discovery"
    for name, filename in (
        ("discovery_alpha_grid", "alpha_grid.jsonl"),
        ("discovery_alpha_diagnostics", "alpha_grid_diagnostics.json"),
        ("discovery_beta_grid", "beta_grid.jsonl"),
        ("discovery_beta_diagnostics", "beta_grid_diagnostics.json"),
        ("discovery_strength_grid", "strength_grid.jsonl"),
        ("discovery_vocab_scores", "discovery_vocab_scores.pt"),
        ("candidate_metrics", "candidate_metrics.pt"),
        ("discovery_trial_summary", "trial_summary.csv"),
        ("discovery_candidate_scores", "candidate_scores.csv"),
        ("candidate_discovery", "candidate_discovery.json"),
    ):
        hashes[name] = sha256_file(discovery_dir / filename)


def _eligible_candidate_token_ids(tokenizer: Any, config: Mapping[str, Any]) -> list[int]:
    eligible = {int(value) for value in word_token_ids(tokenizer).tolist()}
    excluded = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    excluded.update(int(value) for value in config["readout"]["generic_evaluator_token_ids"])
    for branch in config["meta_branches"].values():
        for text_value in (branch["prompt"], *branch["labels"]):
            token_ids = tokenizer(str(text_value), add_special_tokens=False)["input_ids"]
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            excluded.update(int(value) for value in token_ids)
    result = sorted(eligible - excluded)
    if not result:
        raise RuntimeError("candidate_selection_gate_failed: token eligibility filter is empty")
    return result


def _extract_discovery_vocab_tensor(
    record: dict[str, Any],
    config: Mapping[str, Any],
) -> torch.Tensor:
    full = record.pop("_discovery_vocab_scores")
    branches = tuple(str(value) for value in config["meta_branches"])
    layers = tuple(str(int(value)) for value in config["layers"]["readout"])
    tensor = torch.stack([
        torch.stack([
            torch.stack([
                full[condition][branch][layer].float()
                for layer in layers
            ])
            for branch in branches
        ])
        for condition in DISCOVERY_CANDIDATE_CONDITIONS
    ])
    if tensor.ndim != 4 or not torch.isfinite(tensor).all():
        raise AssertionError("discovery full-vocabulary score tensor is invalid")
    compact = tensor.to(torch.float16)
    if not torch.isfinite(compact).all():
        raise AssertionError("discovery full-vocabulary score compression is non-finite")
    return compact


def _candidate_support_row(
    record: Mapping[str, Any],
    *,
    weak_alpha: float,
    strong_alpha: float,
) -> list[float]:
    clean = float(record["support"]["clean"])
    target_grid = record["support"]["target_grid"]
    return [
        0.0,
        clean - float(target_grid[str(float(weak_alpha))]),
        clean - float(target_grid[str(float(strong_alpha))]),
        float(record["support"]["random_drop"]),
        float(record["support"]["alternative_drop"]),
        float(record["support"]["alternative_random_drop"]),
        clean - float(target_grid[str(float(strong_alpha))]),
    ]


def _assert_candidate_direction_files(
    run_dir: Path,
    candidates: Iterable[Mapping[str, Any]],
) -> None:
    for candidate in candidates:
        direction_path = run_dir / str(candidate["direction_path"])
        if sha256_file(direction_path) != candidate["direction_file_sha256"]:
            raise AssertionError("stale candidate direction file hash")


def _load_and_validate_frozen_protocol(
    run_dir: Path,
    config: Mapping[str, Any],
    split: Mapping[str, Any],
    hashes: dict[str, str],
) -> dict[str, Any]:
    frozen_path = run_dir / "frozen_protocol.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen_sources = {
        str(name): hashes[str(name)]
        for name in frozen.get("source_hashes", {})
    }
    validate_frozen_protocol(frozen, config, split, frozen_sources)
    _assert_candidate_direction_files(run_dir, frozen["candidates"])
    hashes["frozen_protocol"] = sha256_file(frozen_path)
    return frozen


def heldout_support_rows(
    records: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    support = config["support_matching"]
    rows = []
    for record in records:
        targeted = float(record["support"]["targeted_drop"])
        alternative = float(record["support"]["alternative_drop"])
        mismatch = alternative - targeted
        tolerance = max(
            float(support["absolute_tolerance_nat"]),
            float(support["relative_tolerance"]) * abs(targeted),
        )
        rows.append({
            "item_id": str(record["item_id"]),
            "support_drop_targeted": targeted,
            "support_drop_alternative": alternative,
            "signed_mismatch_alternative_minus_targeted": mismatch,
            "absolute_mismatch": abs(mismatch),
            "item_tolerance": tolerance,
            "targeted_drop_positive": targeted > 0,
            "support_matched": item_support_matched(
                targeted,
                alternative,
                absolute_tolerance_nat=float(support["absolute_tolerance_nat"]),
                relative_tolerance=float(support["relative_tolerance"]),
            ),
        })
    return rows


def _write_rows_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"inconsistent columns while writing {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def heldout_effect_rows(
    records: Iterable[Mapping[str, Any]],
    frozen: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    conditions = [str(value) for value in config["conditions"]]
    primary_layer = int(config["layers"]["process"])
    alternative_layer = int(frozen["alternative_layer"])
    intervention_layers: dict[str, int | None] = {
        "clean_preserved": None,
        "targeted_weak_preserved": primary_layer,
        "targeted_strong_preserved": primary_layer,
        "random_strong_preserved": primary_layer,
        "support_matched_alternative_preserved": alternative_layer,
        "alternative_random_preserved": alternative_layer,
        "targeted_strong_reset": primary_layer,
    }
    for record in records:
        clean_support = float(record["support"]["clean"])
        target_grid = record["support"]["target_grid"]
        support_drops = {
            "clean_preserved": 0.0,
            "targeted_weak_preserved": clean_support - float(
                target_grid[str(float(frozen["weak_alpha"]))]
            ),
            "targeted_strong_preserved": float(record["support"]["targeted_drop"]),
            "random_strong_preserved": float(record["support"]["random_drop"]),
            "support_matched_alternative_preserved": float(
                record["support"]["alternative_drop"]
            ),
            "alternative_random_preserved": float(
                record["support"]["alternative_random_drop"]
            ),
            "targeted_strong_reset": float(record["support"]["targeted_drop"]),
        }
        for candidate_rank, candidate in enumerate(frozen["candidates"], start=1):
            layer = str(int(candidate["layer"]))
            token_id = str(int(candidate["token_id"]))
            orientation = int(candidate["orientation"])
            for branch in config["meta_branches"]:
                for condition in conditions:
                    branch_data = record["meta"][condition][branch]
                    raw_score = float(
                        branch_data["jlens"][layer]["explicit"][token_id]["score"]
                    )
                    rows.append({
                        "item_id": str(record["item_id"]),
                        "candidate_rank": candidate_rank,
                        "candidate_label": str(candidate["label"]),
                        "candidate_layer": int(candidate["layer"]),
                        "candidate_token_id": int(candidate["token_id"]),
                        "candidate_orientation": orientation,
                        "branch": str(branch),
                        "condition": condition,
                        "intervention_layer": intervention_layers[condition],
                        "support_drop": support_drops[condition],
                        "candidate_score": raw_score,
                        "oriented_candidate_score": orientation * raw_score,
                        "choice_margin": float(branch_data["scores"]["margin"]),
                    })
    return rows


def run_discovery_phase(
    args: argparse.Namespace,
    run_dir: Path,
    config: Mapping[str, Any],
    hashes: dict[str, str],
) -> dict[str, Any]:
    phase = "discovery"
    answers, split = _load_campaign_inputs(run_dir, hashes)
    _load_pre_discovery_smoke_hash(run_dir, hashes)
    assert_phase_prerequisites(
        run_dir,
        phase,
        protocol_hash=combined_protocol_hash(hashes),
        required_input_hashes=hashes,
    )
    _, tokenizer, lens_model, lens, adapter, runtime = load_runtime(args, config)
    write_manifest(
        run_dir, phase=phase, status="running", config=config, hashes=hashes, runtime=runtime
    )
    discovery_ids = [str(value) for value in split["discovery_item_ids"]]
    heldout_ids = {str(value) for value in split["heldout_item_ids"]}
    if len(discovery_ids) != 16 or set(discovery_ids) & heldout_ids:
        raise AssertionError("candidate discovery split isolation failed")
    by_id = {str(record["item_id"]): record for record in answers}
    if any(item_id not in by_id or bool(by_id[item_id].get("invalid")) for item_id in discovery_ids):
        raise AssertionError("candidate discovery includes a missing or invalid answer")

    discovery_dir = run_dir / phase
    alpha_path = discovery_dir / "alpha_grid.jsonl"
    alpha_diagnostics_path = discovery_dir / "alpha_grid_diagnostics.json"
    beta_path = discovery_dir / "beta_grid.jsonl"
    beta_diagnostics_path = discovery_dir / "beta_grid_diagnostics.json"
    strength_path = discovery_dir / "strength_grid.jsonl"
    for path in (
        alpha_path,
        alpha_diagnostics_path,
        beta_path,
        beta_diagnostics_path,
        strength_path,
    ):
        if path.exists() and path.stat().st_size:
            raise RuntimeError(f"discovery diagnostic artifact already contains data: {path}")
    alpha_records: list[dict[str, Any]] = []
    for item_id in discovery_ids:
        append_jsonl(run_dir / "events.jsonl", {
            "timestamp": utc_now(),
            "event_type": "discovery_alpha_grid_started",
            "item_id": item_id,
        })
        alpha_record = measure_strength_grid_item(
            adapter, by_id[item_id], config, families=("alpha",)
        )
        alpha_records.append(alpha_record)
        append_jsonl(alpha_path, alpha_record)
        logging.getLogger("process_sensitive_replay").info(
            "discovery alpha grid item=%s completed=%d/%d",
            item_id,
            len(alpha_records),
            len(discovery_ids),
        )
        append_jsonl(run_dir / "events.jsonl", {
            "timestamp": utc_now(),
            "event_type": "discovery_alpha_grid_completed",
            "item_id": item_id,
        })
    alpha_diagnostics = alpha_grid_diagnostics(alpha_records, config)
    alpha_diagnostics_path.write_text(
        json.dumps(alpha_diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hashes["discovery_alpha_grid"] = sha256_file(alpha_path)
    hashes["discovery_alpha_diagnostics"] = sha256_file(alpha_diagnostics_path)
    alpha_selection = select_discovery_alpha(alpha_records, config)
    append_jsonl(run_dir / "events.jsonl", {
        "timestamp": utc_now(),
        "event_type": "discovery_alpha_strengths_selected",
        **alpha_selection,
    })

    beta_records: list[dict[str, Any]] = []
    for item_id in discovery_ids:
        append_jsonl(run_dir / "events.jsonl", {
            "timestamp": utc_now(),
            "event_type": "discovery_beta_grid_started",
            "item_id": item_id,
            "frozen_strong_alpha": alpha_selection["strong_alpha"],
        })
        beta_record = measure_strength_grid_item(
            adapter, by_id[item_id], config, families=("beta",)
        )
        beta_records.append(beta_record)
        append_jsonl(beta_path, beta_record)
        logging.getLogger("process_sensitive_replay").info(
            "discovery beta grid item=%s completed=%d/%d",
            item_id,
            len(beta_records),
            len(discovery_ids),
        )
        append_jsonl(run_dir / "events.jsonl", {
            "timestamp": utc_now(),
            "event_type": "discovery_beta_grid_completed",
            "item_id": item_id,
            "frozen_strong_alpha": alpha_selection["strong_alpha"],
        })

    beta_by_id = {str(record["item_id"]): record for record in beta_records}
    strength_records: list[dict[str, Any]] = []
    for alpha_record in alpha_records:
        item_id = str(alpha_record["item_id"])
        beta_record = beta_by_id[item_id]
        parity_fields = (
            "clean_cache_digest",
            "transcript_hash",
            "question_token_hash",
            "answer_token_hash",
        )
        if any(alpha_record[field] != beta_record[field] for field in parity_fields):
            raise AssertionError("alpha/beta discovery passes lost clean replay parity")
        if not math.isclose(
            float(alpha_record["clean_support"]),
            float(beta_record["clean_support"]),
            abs_tol=float(config["reset_parity"]["absolute_tolerance"]),
            rel_tol=float(config["reset_parity"]["relative_tolerance"]),
        ):
            raise AssertionError("alpha/beta discovery passes lost clean support parity")
        record = {
            **alpha_record,
            "beta_grid": beta_record["beta_grid"],
            "alternative_gradient_parity": beta_record[
                "alternative_gradient_parity"
            ],
            "gradient_parity": {
                "alpha_pass": alpha_record["gradient_parity"],
                "beta_pass": beta_record["gradient_parity"],
            },
        }
        strength_records.append(record)
        append_jsonl(strength_path, record)
        append_jsonl(run_dir / "state_audits.jsonl", {
            "timestamp": utc_now(),
            "phase": phase,
            "item_id": item_id,
            "clean_cache_digest": record["clean_cache_digest"],
            "targeted_grid_cache_digests": {
                strength: value["cache_digest"]
                for strength, value in record["alpha_grid"].items()
            },
            "alternative_grid_cache_digests": {
                layer: {
                    strength: value["cache_digest"]
                    for strength, value in layer_grid.items()
                }
                for layer, layer_grid in record["beta_grid"].items()
            },
            "gradient_parity": record["gradient_parity"],
        })
        for strength, measurement in record["alpha_grid"].items():
            for position in measurement["positions"]:
                append_jsonl(run_dir / "process_interventions.jsonl", {
                    "timestamp": utc_now(),
                    "phase": phase,
                    "item_id": item_id,
                    "grid": "alpha_grid",
                    "grid_strength": strength,
                    "support_before": record["clean_support"],
                    "support_after": measurement["support_after"],
                    **position,
                })
        for alternative_layer, layer_grid in record["beta_grid"].items():
            for strength, measurement in layer_grid.items():
                for position in measurement["positions"]:
                    append_jsonl(run_dir / "process_interventions.jsonl", {
                        "timestamp": utc_now(),
                        "phase": phase,
                        "item_id": item_id,
                        "grid": "alternative_layer_beta_grid",
                        "alternative_layer": int(alternative_layer),
                        "grid_strength": strength,
                        "support_before": record["clean_support"],
                        "support_after": measurement["support_after"],
                        **position,
                    })
        append_jsonl(run_dir / "events.jsonl", {
            "timestamp": utc_now(),
            "event_type": "discovery_strength_grid_completed",
            "item_id": item_id,
        })

    beta_diagnostics = beta_grid_diagnostics(
        strength_records,
        config,
        strong_alpha=float(alpha_selection["strong_alpha"]),
    )
    beta_diagnostics_path.write_text(
        json.dumps(beta_diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hashes["discovery_beta_grid"] = sha256_file(beta_path)
    hashes["discovery_beta_diagnostics"] = sha256_file(beta_diagnostics_path)
    hashes["discovery_strength_grid"] = sha256_file(strength_path)
    strength_selection = select_discovery_strengths(strength_records, config)
    if strength_selection["alpha"] != alpha_selection:
        raise AssertionError("beta calibration changed the frozen alpha selection")
    weak_alpha = float(strength_selection["alpha"]["weak_alpha"])
    strong_alpha = float(strength_selection["alpha"]["strong_alpha"])
    beta = float(strength_selection["beta"]["beta"])
    alternative_layer = int(
        strength_selection["beta"]["alternative_layer"]
    )
    selected_strengths = {
        "weak_alpha": weak_alpha,
        "strong_alpha": strong_alpha,
        "beta": beta,
        "alternative_layer": alternative_layer,
    }
    append_jsonl(run_dir / "events.jsonl", {
        "timestamp": utc_now(),
        "event_type": "discovery_strengths_selected",
        **selected_strengths,
    })

    vocab_rows: list[torch.Tensor] = []
    support_rows: list[list[float]] = []
    candidate_records: list[dict[str, Any]] = []
    for item_id in discovery_ids:
        append_jsonl(run_dir / "events.jsonl", {
            "timestamp": utc_now(),
            "event_type": "candidate_discovery_trial_started",
            "item_id": item_id,
        })
        record = run_smoke_item(
            adapter,
            lens_model,
            lens,
            tokenizer,
            by_id[item_id],
            config,
            phase="discovery",
            frozen_protocol=selected_strengths,
            include_full_vocab=True,
        )
        vocab_rows.append(_extract_discovery_vocab_tensor(record, config))
        support_rows.append(_candidate_support_row(
            record, weak_alpha=weak_alpha, strong_alpha=strong_alpha
        ))
        candidate_records.append(record)
        _log_smoke_record(run_dir, record)
        logging.getLogger("process_sensitive_replay").info(
            "discovery candidate replay item=%s completed=%d/%d",
            item_id,
            len(candidate_records),
            len(discovery_ids),
        )

    vocab_scores = torch.stack(vocab_rows)
    support_drops = torch.tensor(support_rows, dtype=torch.float32)
    branch_names = tuple(str(value) for value in config["meta_branches"])
    vocab_payload = {
        "schema_version": 1,
        "item_ids": discovery_ids,
        "condition_names": list(DISCOVERY_CANDIDATE_CONDITIONS),
        "branch_names": list(branch_names),
        "layers": [int(value) for value in config["layers"]["readout"]],
        "score_dtype": "float16",
        "scores": vocab_scores,
        "support_drops": support_drops,
    }
    vocab_path = discovery_dir / "discovery_vocab_scores.pt"
    torch.save(vocab_payload, vocab_path)
    hashes["discovery_vocab_scores"] = sha256_file(vocab_path)

    primary_branch = str(config["candidate_selection"]["primary_branch"])
    primary_branch_index = branch_names.index(primary_branch)
    eligible_token_ids = _eligible_candidate_token_ids(tokenizer, config)
    ranking = rank_candidate_grid(
        vocab_scores[:, :, primary_branch_index, :, :],
        support_drops,
        layers=config["layers"]["readout"],
        condition_names=DISCOVERY_CANDIDATE_CONDITIONS,
        eligible_token_ids=eligible_token_ids,
        config=config,
    )
    metrics_path = discovery_dir / "candidate_metrics.pt"
    torch.save({
        "schema_version": 1,
        "item_ids": discovery_ids,
        "layers": list(ranking["layers"]),
        "vocab_size": ranking["vocab_size"],
        "metric_names": list(ranking["metric_names"]),
        "metrics": ranking["metrics"],
        "eligibility": ranking["eligibility"],
        "orientation": ranking["orientation"],
        "support_adjusted_divergence_ratio": ranking[
            "support_adjusted_divergence_ratio"
        ],
        "structured_sign_consistency": ranking["structured_sign_consistency"],
        "random_sign_consistency": ranking["random_sign_consistency"],
        "ranked_flat_indices": ranking["ranked_flat_indices"],
        "aggregate_rank": ranking["aggregate_rank"],
    }, metrics_path)
    hashes["candidate_metrics"] = sha256_file(metrics_path)

    selected_candidates: list[dict[str, Any]] = []
    selected_directions: list[torch.Tensor] = []
    dedup_rejections: list[dict[str, Any]] = []
    maximum = int(config["candidate_selection"]["max_candidates"])
    cosine_limit = float(
        config["candidate_selection"]["dedup_max_abs_direction_cosine"]
    )
    for rank_offset in range(int(ranking["eligible_count"])):
        row = candidate_ranking_row(ranking, rank_offset)
        token_id = int(row["token_id"])
        layer = int(row["layer"])
        label = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        candidate = candidate_direction(
            label=label,
            token_id=token_id,
            layer=layer,
            lens_model=lens_model,
            lens=lens,
            checkpoint_dir=discovery_dir,
        )
        cosines = [
            abs(float(torch.dot(candidate.direction.float(), previous.float()).item()))
            for previous in selected_directions
        ]
        if cosines and max(cosines) > cosine_limit:
            if len(dedup_rejections) < 100:
                dedup_rejections.append({**row, "label": label, "max_abs_cosine": max(cosines)})
            continue
        direction_path = Path(candidate.direction_path)
        metadata = candidate.metadata()
        metadata["direction_path"] = str(direction_path.relative_to(run_dir))
        selected_candidates.append({
            **row,
            "label": label,
            **metadata,
            "direction_file_sha256": sha256_file(direction_path),
        })
        selected_directions.append(candidate.direction.detach().cpu())
        append_jsonl(run_dir / "events.jsonl", {
            "timestamp": utc_now(),
            "event_type": "candidate_direction_selected",
            "token_id": token_id,
            "layer": layer,
            "orientation": row["orientation"],
            "aggregate_rank": row["aggregate_rank"],
            "direction_sha256": candidate.direction_sha256,
        })
        if len(selected_candidates) == maximum:
            break
    if not selected_candidates:
        raise RuntimeError("candidate_selection_gate_failed: cosine dedup removed every candidate")

    top_ranked = []
    for offset in range(min(100, int(ranking["eligible_count"]))):
        row = candidate_ranking_row(ranking, offset)
        top_ranked.append({
            **row,
            "label": tokenizer.decode(
                [row["token_id"]], clean_up_tokenization_spaces=False
            ),
        })
    _write_trial_summary(discovery_dir, candidate_records)
    _write_candidate_scores(discovery_dir, candidate_records)
    _append_frozen_discovery_candidate_scores(
        discovery_dir / "candidate_scores.csv",
        vocab_scores,
        item_ids=discovery_ids,
        condition_names=DISCOVERY_CANDIDATE_CONDITIONS,
        branch_names=branch_names,
        layers=[int(value) for value in config["layers"]["readout"]],
        candidates=selected_candidates,
    )
    hashes["discovery_trial_summary"] = sha256_file(
        discovery_dir / "trial_summary.csv"
    )
    hashes["discovery_candidate_scores"] = sha256_file(
        discovery_dir / "candidate_scores.csv"
    )
    discovery_payload = {
        "schema_version": 1,
        "experiment_name": "process_sensitive_replay",
        "discovery_item_ids": discovery_ids,
        "heldout_item_ids_accessed": [],
        "source_hashes": dict(hashes),
        "strength_selection": strength_selection,
        **selected_strengths,
        "candidate_selection": dict(config["candidate_selection"]),
        "eligible_word_token_count": len(eligible_token_ids),
        "eligible_word_token_ids_sha256": sha256_json(eligible_token_ids),
        "eligible_direction_count": int(ranking["eligible_count"]),
        "candidates": selected_candidates,
        "candidate_token_ids": sorted({
            int(candidate["token_id"]) for candidate in selected_candidates
        }),
        "top_ranked_candidates": top_ranked,
        "dedup_rejections": dedup_rejections,
    }
    discovery_path = discovery_dir / "candidate_discovery.json"
    discovery_path.write_text(
        json.dumps(discovery_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    hashes["candidate_discovery"] = sha256_file(discovery_path)
    measurements = {
        "discovery_items": len(discovery_ids),
        "heldout_items_accessed": 0,
        **selected_strengths,
        "eligible_word_tokens": len(eligible_token_ids),
        "eligible_directions": int(ranking["eligible_count"]),
        "frozen_candidate_count": len(selected_candidates),
        "frozen_candidates": [
            {
                "token_id": candidate["token_id"],
                "layer": candidate["layer"],
                "orientation": candidate["orientation"],
                "direction_sha256": candidate["direction_sha256"],
            }
            for candidate in selected_candidates
        ],
    }
    write_gate(run_dir, GateStatus(
        phase=phase,
        status="passed",
        protocol_hash=combined_protocol_hash(hashes),
        input_hashes=dict(hashes),
        measurements=measurements,
    ))
    return measurements


def run_freeze_phase(
    run_dir: Path,
    config: Mapping[str, Any],
    hashes: dict[str, str],
) -> dict[str, Any]:
    phase = "freeze"
    # Check the causal prerequisite before trying to hash discovery artifacts.
    # A failed discovery intentionally withholds candidate artifacts, so loading
    # those files first masks the real fail-closed reason with FileNotFoundError.
    discovery_gate = read_gate(run_dir, "discovery")
    if not discovery_gate.passed:
        raise RuntimeError(
            f"prerequisite discovery is {discovery_gate.status}, not passed"
        )
    _answers, split = _load_campaign_inputs(run_dir, hashes)
    _load_pre_discovery_smoke_hash(run_dir, hashes)
    _load_discovery_hashes(run_dir, hashes)
    assert_phase_prerequisites(
        run_dir,
        phase,
        protocol_hash=combined_protocol_hash(hashes),
        required_input_hashes=hashes,
    )
    discovery_path = run_dir / "discovery" / "candidate_discovery.json"
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    if discovery.get("heldout_item_ids_accessed") != []:
        raise AssertionError("freeze rejected discovery with held-out access")
    if [str(value) for value in discovery["discovery_item_ids"]] != [
        str(value) for value in split["discovery_item_ids"]
    ]:
        raise AssertionError("freeze rejected a stale discovery split")
    for name, expected in discovery.get("source_hashes", {}).items():
        if hashes.get(name) != expected:
            raise AssertionError(f"freeze rejected stale discovery source hash {name}")
    _assert_candidate_direction_files(run_dir, discovery["candidates"])

    source_hashes = dict(hashes)
    frozen = {
        "schema_version": 1,
        "experiment_name": "process_sensitive_replay",
        "frozen_at": utc_now(),
        "source_hashes": source_hashes,
        "discovery_item_ids": [str(value) for value in split["discovery_item_ids"]],
        "weak_alpha": float(discovery["weak_alpha"]),
        "strong_alpha": float(discovery["strong_alpha"]),
        "beta": float(discovery["beta"]),
        "alternative_layer": int(discovery["alternative_layer"]),
        "strength_selection": discovery["strength_selection"],
        "candidate_selection": dict(config["candidate_selection"]),
        "support_matching": dict(config["support_matching"]),
        "conditions": list(config["conditions"]),
        "meta_branches": dict(config["meta_branches"]),
        "candidates": discovery["candidates"],
        "candidate_token_ids": discovery["candidate_token_ids"],
        "heldout_access_permitted": False,
    }
    measurements = validate_frozen_protocol(frozen, config, split, source_hashes)
    frozen_path = run_dir / "frozen_protocol.json"
    if frozen_path.exists():
        raise RuntimeError("frozen_protocol.json already exists")
    frozen_temporary = run_dir / "frozen_protocol.json.tmp"
    frozen_temporary.write_text(
        json.dumps(frozen, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(frozen_temporary, frozen_path)
    hashes["frozen_protocol"] = sha256_file(frozen_path)
    measurements["frozen_protocol_sha256"] = hashes["frozen_protocol"]
    write_gate(run_dir, GateStatus(
        phase=phase,
        status="passed",
        protocol_hash=combined_protocol_hash(hashes),
        input_hashes=dict(hashes),
        measurements=measurements,
    ))
    return measurements


def run_smoke_phase(
    args: argparse.Namespace,
    run_dir: Path,
    config: Mapping[str, Any],
    hashes: dict[str, str],
) -> dict[str, Any]:
    phase = args.phase
    answers, split = _load_campaign_inputs(run_dir, hashes)
    frozen = None
    if phase == "post_freeze_smoke":
        _load_pre_discovery_smoke_hash(run_dir, hashes)
        _load_discovery_hashes(run_dir, hashes)
        frozen = _load_and_validate_frozen_protocol(
            run_dir, config, split, hashes
        )
    assert_phase_prerequisites(
        run_dir, phase, protocol_hash=combined_protocol_hash(hashes),
        required_input_hashes=hashes,
    )
    _, tokenizer, lens_model, lens, adapter, runtime = load_runtime(args, config)
    write_manifest(run_dir, phase=phase, status="running", config=config, hashes=hashes, runtime=runtime)
    by_id = {str(record["item_id"]): record for record in answers}
    item_ids = split["discovery_item_ids"][: int(config["smoke"]["item_count"])]
    phase_dir = run_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    trials_path = phase_dir / "trials.jsonl"
    if trials_path.exists() and trials_path.stat().st_size:
        raise RuntimeError(f"{phase} trials already contain data; start a new campaign")
    records = []
    for item_id in item_ids:
        append_jsonl(run_dir / "events.jsonl", {
            "timestamp": utc_now(), "event_type": "smoke_trial_started",
            "phase": phase, "item_id": item_id,
        })
        record = run_smoke_item(
            adapter, lens_model, lens, tokenizer, by_id[str(item_id)], config,
            phase=phase, frozen_protocol=frozen,
        )
        records.append(record)
        append_jsonl(trials_path, record)
        _log_smoke_record(run_dir, record)
    report = summarize_smoke(records, config, phase=phase)
    report_path = run_dir / phase / "smoke_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_trial_summary(phase_dir, records)
    _write_candidate_scores(phase_dir, records)
    report_hash_name = (
        "post_freeze_smoke_report"
        if phase == "post_freeze_smoke"
        else "pre_discovery_smoke_report"
    )
    hashes[report_hash_name] = sha256_file(report_path)
    hashes[f"{phase}_trials"] = sha256_file(trials_path)
    hashes[f"{phase}_trial_summary"] = sha256_file(phase_dir / "trial_summary.csv")
    hashes[f"{phase}_candidate_scores"] = sha256_file(
        phase_dir / "candidate_scores.csv"
    )
    if report.get("passed") is not True:
        raise PhaseGateFailure(
            "invalid_support_match",
            str(report.get("failure_reason", "support_match_gate_failed")),
            report,
        )
    write_gate(run_dir, GateStatus(
        phase=phase, status="passed", protocol_hash=combined_protocol_hash(hashes),
        input_hashes=dict(hashes), measurements=report,
    ))
    return report


def run_heldout_phase(
    args: argparse.Namespace,
    run_dir: Path,
    config: Mapping[str, Any],
    hashes: dict[str, str],
) -> dict[str, Any]:
    phase = "heldout"
    answers, split = _load_campaign_inputs(run_dir, hashes)
    _load_pre_discovery_smoke_hash(run_dir, hashes)
    _load_discovery_hashes(run_dir, hashes)
    frozen = _load_and_validate_frozen_protocol(run_dir, config, split, hashes)
    _load_post_freeze_smoke_hash(run_dir, hashes)
    assert_phase_prerequisites(
        run_dir,
        phase,
        protocol_hash=combined_protocol_hash(hashes),
        required_input_hashes=hashes,
    )
    _, tokenizer, lens_model, lens, adapter, runtime = load_runtime(args, config)
    write_manifest(
        run_dir,
        phase=phase,
        status="running",
        config=config,
        hashes=hashes,
        runtime=runtime,
    )
    heldout_ids = [str(value) for value in split["heldout_item_ids"]]
    discovery_ids = {str(value) for value in split["discovery_item_ids"]}
    if not heldout_ids or discovery_ids.intersection(heldout_ids):
        raise AssertionError("held-out split is empty or overlaps discovery")
    by_id = {str(record["item_id"]): record for record in answers}
    if any(
        item_id not in by_id or bool(by_id[item_id].get("invalid"))
        for item_id in heldout_ids
    ):
        raise AssertionError("held-out split contains a missing or invalid answer")

    heldout_dir = run_dir / phase
    trials_path = heldout_dir / "trials.jsonl"
    if trials_path.exists() and trials_path.stat().st_size:
        raise RuntimeError("held-out trials already contain data; start a new campaign")
    records: list[dict[str, Any]] = []
    for item_id in heldout_ids:
        append_jsonl(run_dir / "events.jsonl", {
            "timestamp": utc_now(),
            "event_type": "heldout_trial_started",
            "item_id": item_id,
        })
        record = run_smoke_item(
            adapter,
            lens_model,
            lens,
            tokenizer,
            by_id[item_id],
            config,
            phase=phase,
            frozen_protocol=frozen,
        )
        records.append(record)
        append_jsonl(trials_path, record)
        _log_smoke_record(run_dir, record)
        logging.getLogger("process_sensitive_replay").info(
            "held-out replay item=%s completed=%d/%d",
            item_id,
            len(records),
            len(heldout_ids),
        )

    summarize_smoke(records, config, phase=phase)
    if [str(record["item_id"]) for record in records] != heldout_ids:
        raise AssertionError("held-out execution order or coverage changed")
    trial_summary_path = heldout_dir / "trial_summary.csv"
    candidate_scores_path = heldout_dir / "candidate_scores.csv"
    _write_trial_summary(heldout_dir, records)
    _write_candidate_scores(heldout_dir, records)
    support_rows = heldout_support_rows(records, config)
    support_path = run_dir / "heldout_support_match.csv"
    _write_rows_csv(support_path, support_rows)
    matching = support_match_summary(
        [
            (
                float(record["support"]["targeted_drop"]),
                float(record["support"]["alternative_drop"]),
            )
            for record in records
        ],
        config,
    )
    support_summary_path = heldout_dir / "support_match_summary.json"
    support_summary_path.write_text(
        json.dumps(matching, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    support_plot_path = generate_support_match_plot(
        support_rows, heldout_dir / "support_match_diagnostic.png"
    )
    hashes.update({
        "heldout_trials": sha256_file(trials_path),
        "heldout_trial_summary": sha256_file(trial_summary_path),
        "heldout_candidate_scores": sha256_file(candidate_scores_path),
        "heldout_support_match": sha256_file(support_path),
        "heldout_support_match_summary": sha256_file(support_summary_path),
        "heldout_support_match_plot": sha256_file(support_plot_path),
    })
    if not matching["passed"]:
        raise RuntimeError("support_match_gate_failed on held-out data")

    effect_rows = heldout_effect_rows(records, frozen, config)
    effects_path = run_dir / "heldout_effects.csv"
    _write_rows_csv(effects_path, effect_rows)
    hashes["heldout_effects"] = sha256_file(effects_path)
    measurements = {
        "heldout_items": len(records),
        "discovery_items_accessed": 0,
        "candidate_count": len(frozen["candidates"]),
        "support_matching": matching,
        "critical_checks_per_item": True,
        "process_hook_disabled_during_turn3": True,
    }
    write_gate(run_dir, GateStatus(
        phase=phase,
        status="passed",
        protocol_hash=combined_protocol_hash(hashes),
        input_hashes=dict(hashes),
        measurements=measurements,
    ))
    return measurements


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _markdown_number(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.6g}"


def _markdown_interval(values: Any) -> str:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return "NA"
    return f"[{_markdown_number(values[0])}, {_markdown_number(values[1])}]"


def render_results_markdown(
    candidate_statistics: Sequence[Mapping[str, Any]],
    support_summary: Mapping[str, Any],
    *,
    heldout_items: int,
    alternative_layer: int | None = None,
) -> str:
    lines = [
        "# Process-sensitive replay held-out results",
        "",
        f"Held-out items: {heldout_items}",
        "Primary intervention layer: 31",
        f"Frozen alternative intervention layer: {alternative_layer}",
        "",
        (
            "Support-match gate: **PASSED** "
            f"({support_summary['matched_items']}/{support_summary['valid_items']}, "
            f"{float(support_summary['item_match_fraction']):.1%})."
        ),
        "",
        "## Interpretation status",
        "",
        (
            "These are descriptive held-out estimates. The frozen protocol does not "
            "contain a numerical held-out convergence decision threshold, so this "
            "report does **not** automatically classify any candidate as "
            "process-sensitive or M(P)-like."
        ),
        "",
        (
            "The experiment does not test candidate-to-judgment/control causal "
            "mediation and cannot prove a higher-order representation."
        ),
        "",
        "## H1–H7 and control contrasts",
        "",
    ]
    for statistic in candidate_statistics:
        label = (
            str(statistic["candidate_label"])
            .replace("|", "\\|")
            .replace("`", "\\`")
        )
        lines.extend([
            (
                f"### Candidate {statistic['candidate_rank']}: `{label}` "
                f"(token {statistic['candidate_token_id']}, "
                f"layer {statistic['candidate_layer']}, branch {statistic['branch']})"
            ),
            "",
            "| Test / contrast | Mean or estimate | Median | Item-bootstrap 95% CI |",
            "|---|---:|---:|---:|",
        ])
        effect_rows = (
            ("H1 targeted − clean", "h1_targeted_minus_clean"),
            ("H2 alternative − clean", "h2_alternative_minus_clean"),
            ("H3 targeted − alternative", "h3_targeted_minus_alternative"),
            (
                "H3 support-normalized targeted − alternative",
                "h3_support_normalized_response_difference",
            ),
            ("H4 targeted − random", "h4_targeted_minus_random"),
            ("H4 alternative − same-layer random", "h4_alternative_minus_random"),
            ("H5 targeted preserved − reset", "h5_targeted_preserved_minus_reset"),
        )
        for row_label, key in effect_rows:
            summary = statistic[key]
            lines.append(
                f"| {row_label} | {_markdown_number(summary['mean'])} | "
                f"{_markdown_number(summary['median'])} | "
                f"{_markdown_interval(summary['item_bootstrap_95_ci'])} |"
            )
        fixed = statistic["h3_item_fixed_effect_model"]
        lines.append(
            "| H3 item-FE mechanism term | "
            f"{_markdown_number(fixed['beta_mechanism'])} | NA | "
            f"{_markdown_interval(fixed['item_bootstrap_95_ci']['beta_mechanism'])} |"
        )
        h6 = statistic["h6_candidate_score_vs_support"]
        lines.append(
            f"| H6 candidate-score/support slope | {_markdown_number(h6['slope'])} "
            f"| NA | {_markdown_interval(h6['item_bootstrap_95_ci']['slope'])} |"
        )
        h7 = statistic.get("h7_confidence_margin_vs_support")
        if h7 is not None:
            lines.append(
                f"| H7 confidence-margin/support slope | {_markdown_number(h7['slope'])} "
                f"| NA | {_markdown_interval(h7['item_bootstrap_95_ci']['slope'])} |"
            )
        else:
            lines.append("| H7 confidence-margin/support slope | NA | NA | NA |")
        lines.extend([
            "",
            (
                "H3 item-FE support coefficient: "
                f"`{_markdown_number(fixed['beta_support'])}`; absolute mechanism/shared-"
                f"effect ratio: `{_markdown_number(fixed['abs_mechanism_to_shared_effect_ratio'])}`."
            ),
            (
                "H6 correlations: Pearson "
                f"`{_markdown_number(h6['pearson'])}`, Spearman "
                f"`{_markdown_number(h6['spearman'])}`."
            ),
            (
                "H7 correlations: NA for this branch."
                if h7 is None
                else (
                    "H7 correlations: Pearson "
                    f"`{_markdown_number(h7['pearson'])}`, Spearman "
                    f"`{_markdown_number(h7['spearman'])}`."
                )
            ),
            "",
        ])
    lines.extend([
        "Exact machine-readable statistics and plot hashes are in "
        "`analysis_report.json` and `plot_manifest.json`.",
        "",
    ])
    return "\n".join(lines)


def run_analyze_phase(
    run_dir: Path,
    config: Mapping[str, Any],
    hashes: dict[str, str],
) -> dict[str, Any]:
    phase = "analyze"
    _answers, split = _load_campaign_inputs(run_dir, hashes)
    _load_pre_discovery_smoke_hash(run_dir, hashes)
    _load_discovery_hashes(run_dir, hashes)
    frozen = _load_and_validate_frozen_protocol(run_dir, config, split, hashes)
    _load_post_freeze_smoke_hash(run_dir, hashes)
    _load_heldout_hashes(run_dir, hashes)
    assert_phase_prerequisites(
        run_dir,
        phase,
        protocol_hash=combined_protocol_hash(hashes),
        required_input_hashes=hashes,
    )
    heldout_ids = {str(value) for value in split["heldout_item_ids"]}
    effect_rows: list[dict[str, Any]] = _read_csv_rows(run_dir / "heldout_effects.csv")
    support_rows: list[dict[str, Any]] = _read_csv_rows(
        run_dir / "heldout_support_match.csv"
    )
    candidate_score_rows: list[dict[str, Any]] = _read_csv_rows(
        run_dir / "heldout" / "candidate_scores.csv"
    )
    if {str(row["item_id"]) for row in effect_rows} != heldout_ids:
        raise AssertionError("analysis effect table does not exactly cover held-out IDs")
    if {str(row["item_id"]) for row in support_rows} != heldout_ids:
        raise AssertionError("analysis support table does not exactly cover held-out IDs")
    for row in support_rows:
        row["support_matched"] = str(row["support_matched"]).lower() == "true"
        row["targeted_drop_positive"] = (
            str(row["targeted_drop_positive"]).lower() == "true"
        )
    support_summary = json.loads(
        (run_dir / "heldout" / "support_match_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if support_summary.get("passed") is not True:
        raise RuntimeError("support_match_gate_failed: analysis refused invalid held-out data")
    candidate_statistics = json_safe(analyze_candidate_effects(effect_rows))
    if not candidate_statistics:
        raise RuntimeError("analysis produced no frozen-candidate statistics")
    analysis_dir = run_dir / phase
    plot_paths = generate_required_plots(
        effect_rows,
        support_rows,
        candidate_score_rows,
        analysis_dir / "plots",
        primary_branch=str(config["candidate_selection"]["primary_branch"]),
        generic_token_ids=config["readout"]["generic_evaluator_token_ids"],
    )
    plot_hashes = {
        str(path.relative_to(run_dir)): sha256_file(path) for path in plot_paths
    }
    plot_manifest_path = analysis_dir / "plot_manifest.json"
    plot_manifest_path.write_text(
        json.dumps(plot_hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "heldout_items": len(heldout_ids),
        "support_matching": support_summary,
        "alternative_layer": int(frozen["alternative_layer"]),
        "candidate_statistics": candidate_statistics,
        "plots": sorted(plot_hashes),
        "interpretation": {
            "maximum_claim": config["interpretation"]["maximum_claim"],
            "prohibited_claim": config["interpretation"]["prohibited_claim"],
            "causal_mediation_tested": False,
            "automatic_candidate_classification": False,
            "classification_reason": (
                "no explicitly frozen held-out convergence decision threshold"
            ),
        },
    }
    report_path = analysis_dir / "analysis_report.json"
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    markdown_path = analysis_dir / "RESULTS.md"
    markdown_path.write_text(
        render_results_markdown(
            candidate_statistics,
            support_summary,
            heldout_items=len(heldout_ids),
            alternative_layer=int(frozen["alternative_layer"]),
        ),
        encoding="utf-8",
    )
    hashes.update({
        "analysis_report": sha256_file(report_path),
        "analysis_plot_manifest": sha256_file(plot_manifest_path),
        "analysis_results_markdown": sha256_file(markdown_path),
    })
    measurements = {
        "heldout_items": len(heldout_ids),
        "candidate_statistics": len(candidate_statistics),
        "plot_count": len(plot_paths),
        "support_matching": support_summary,
        "interpretation_ceiling_enforced": True,
    }
    write_gate(run_dir, GateStatus(
        phase=phase,
        status="passed",
        protocol_hash=combined_protocol_hash(hashes),
        input_hashes=dict(hashes),
        measurements=measurements,
    ))
    return measurements


def invalid_status(exc: BaseException) -> str:
    message = str(exc).lower()
    if "support_match" in message or "support match" in message:
        return "invalid_support_match"
    if "reset" in message:
        return "invalid_reset_parity"
    if any(term in message for term in (
        "cache", "state", "storage", "branch", "hook", "gradient", "logit parity",
        "turn-3", "prefix-suffix", "factual prefix", "transcript token parity",
        "non-finite", "j-lens", "replay produced",
    )):
        return "invalid_cache_state"
    return "failed"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=SUPPORTED_PHASES)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hf-cache-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    config = load_config(config_path)
    initialize_run_dir(run_dir, config_path, config)
    hashes = campaign_hashes(config_path, config)
    protocol_hash = combined_protocol_hash(hashes)
    logger = logging.getLogger("process_sensitive_replay")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        begin_phase_once(run_dir, args.phase)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2
    write_manifest(run_dir, phase=args.phase, status="initializing", config=config, hashes=hashes)
    try:
        if args.phase == "validate":
            result = run_validate(run_dir, config_path, config, hashes)
        elif args.phase == "answer_bank":
            result = run_answer_bank_phase(args, run_dir, config, hashes)
        elif args.phase == "discovery":
            result = run_discovery_phase(args, run_dir, config, hashes)
        elif args.phase == "freeze":
            result = run_freeze_phase(run_dir, config, hashes)
        elif args.phase == "heldout":
            result = run_heldout_phase(args, run_dir, config, hashes)
        elif args.phase == "analyze":
            result = run_analyze_phase(run_dir, config, hashes)
        else:
            result = run_smoke_phase(args, run_dir, config, hashes)
        write_manifest(run_dir, phase=args.phase, status="passed", config=config, hashes=hashes)
        logger.info("phase=%s passed run_dir=%s", args.phase, run_dir)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        status = (
            exc.status if isinstance(exc, PhaseGateFailure) else invalid_status(exc)
        )
        measurements = (
            exc.measurements if isinstance(exc, PhaseGateFailure) else {}
        )
        append_jsonl(run_dir / "errors.jsonl", {
            "timestamp": utc_now(), "phase": args.phase,
            "status": status, "error_type": type(exc).__name__, "message": str(exc),
        })
        write_gate(run_dir, GateStatus(
            phase=args.phase, status=status, protocol_hash=protocol_hash,
            input_hashes=dict(hashes), measurements=measurements, reason=str(exc),
        ))
        write_manifest(run_dir, phase=args.phase, status=status, config=config, hashes=hashes)
        logger.error("phase=%s status=%s: %s", args.phase, status, exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
