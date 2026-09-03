"""Fail-closed runner for process-sensitive replay smoke infrastructure."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import logging
import platform
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from experiments.higher_v_readout_global.runner import load_lens, load_model

from .answer_bank import discover_answer
from .protocol import (
    GateStatus,
    allocate_discovery_split,
    assert_phase_prerequisites,
    canonical_json,
    direct_factual_question,
    load_config,
    load_dataset,
    gate_path,
    sha256_file,
    sha256_json,
    validate_config,
    write_gate,
)
from .replay import QwenReplayAdapter, verify_thinking_disabled
from .smoke import run_smoke_item, summarize_smoke


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_CONFIG = PACKAGE_DIR / "experiment_config.json"
SUPPORTED_PHASES = ("validate", "answer_bank", "pre_discovery_smoke", "post_freeze_smoke")
LOG_FILES = (
    "events.jsonl", "raw_runs.jsonl", "process_interventions.jsonl",
    "tokenizations.jsonl", "jlens_readouts.jsonl", "state_audits.jsonl",
    "errors.jsonl",
)


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
    return {
        "config": sha256_file(config_path),
        "dataset": sha256_file(dataset_path),
        "scientific_protocol": sha256_file(protocol_path),
        "code": code_hash(),
        "model_spec": sha256_json(config["model"]),
        "lens_spec": sha256_json(config["lens"]),
    }


def combined_protocol_hash(hashes: Mapping[str, str]) -> str:
    return sha256_json({key: hashes[key] for key in ("config", "scientific_protocol", "code")})


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
        "packages": package_versions(("torch", "transformers", "jlens", "huggingface-hub")),
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
    runtime = {
        "architecture": type(hf_model).__name__,
        "model_type": text_config.model_type,
        "num_hidden_layers": text_config.num_hidden_layers,
        "hidden_size": text_config.hidden_size,
        "process_layer_type": text_config.layer_types[process_layer],
        "input_device": str(lens_model.input_device),
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
        "lens_path": None if lens_path is None else str(lens_path),
        "lens_revision": lens_revision,
        "lens_source_layers": None if lens is None else list(lens.source_layers),
        "jlens_force_bos": False,
        "model_resolved_revision": getattr(hf_model.config, "_commit_hash", None),
        "tokenizer_resolved_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "lens_sha256": None if lens_path is None else sha256_file(Path(lens_path)),
    }
    return hf_model, tokenizer, lens_model, lens, adapter, runtime


def _write_trial_summary(run_dir: Path, records: list[Mapping[str, Any]]) -> None:
    path = run_dir / "trial_summary.csv"
    fields = [
        "item_id", "phase", "clean_support", "targeted_drop",
        "alternative_drop", "random_drop", "reset_parity", "cache_integrity",
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
        "timestamp": stamp, "event_type": "smoke_trial_completed",
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
        frozen_path = run_dir / "frozen_protocol.json"
        hashes["frozen_protocol"] = sha256_file(frozen_path)
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert_phase_prerequisites(
        run_dir, phase, protocol_hash=combined_protocol_hash(hashes),
        required_input_hashes=hashes,
    )
    _, tokenizer, lens_model, lens, adapter, runtime = load_runtime(args, config)
    write_manifest(run_dir, phase=phase, status="running", config=config, hashes=hashes, runtime=runtime)
    by_id = {str(record["item_id"]): record for record in answers}
    item_ids = split["discovery_item_ids"][: int(config["smoke"]["item_count"])]
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
        _log_smoke_record(run_dir, record)
    report = summarize_smoke(records, config, phase=phase)
    report_path = run_dir / phase / "smoke_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_trial_summary(run_dir, records)
    _write_candidate_scores(run_dir, records)
    hashes["smoke_report"] = sha256_file(report_path)
    write_gate(run_dir, GateStatus(
        phase=phase, status="passed", protocol_hash=combined_protocol_hash(hashes),
        input_hashes=dict(hashes), measurements=report,
    ))
    return report


def invalid_status(exc: BaseException) -> str:
    message = str(exc).lower()
    if "support_match" in message or "support match" in message:
        return "invalid_support_match"
    if "reset" in message:
        return "invalid_reset_parity"
    if any(term in message for term in ("cache", "state", "storage", "branch", "hook")):
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
        else:
            result = run_smoke_phase(args, run_dir, config, hashes)
        write_manifest(run_dir, phase=args.phase, status="passed", config=config, hashes=hashes)
        logger.info("phase=%s passed run_dir=%s", args.phase, run_dir)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        status = invalid_status(exc)
        append_jsonl(run_dir / "errors.jsonl", {
            "timestamp": utc_now(), "phase": args.phase,
            "status": status, "error_type": type(exc).__name__, "message": str(exc),
        })
        write_gate(run_dir, GateStatus(
            phase=args.phase, status=status, protocol_hash=protocol_hash,
            input_hashes=dict(hashes), measurements={}, reason=str(exc),
        ))
        write_manifest(run_dir, phase=args.phase, status=status, config=config, hashes=hashes)
        logger.error("phase=%s status=%s: %s", args.phase, status, exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
