"""Frozen protocol validation, hashing, splitting, and fail-closed phase gates."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


PHASE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "validate": (),
    "answer_bank": ("validate",),
    "pre_discovery_smoke": ("answer_bank",),
    "discovery": ("pre_discovery_smoke",),
    "freeze": ("discovery",),
    "post_freeze_smoke": ("freeze",),
    "heldout": ("post_freeze_smoke",),
    "analyze": ("heldout",),
}

CRITICAL_INVALID_STATUSES = {
    "invalid_support_match",
    "invalid_reset_parity",
    "invalid_cache_state",
}

EXPECTED_CONDITIONS = (
    "clean_preserved",
    "targeted_weak_preserved",
    "targeted_strong_preserved",
    "random_strong_preserved",
    "support_matched_alternative_preserved",
    "alternative_random_preserved",
    "targeted_strong_reset",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_token_ids(token_ids: Sequence[int]) -> str:
    return sha256_json([int(token_id) for token_id in token_ids])


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(path: Path, allowed_types: Iterable[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        "item_id", "prompt", "item_type", "answer_key", "difficulty",
        "domain", "detail",
    }
    if not rows or set(rows[0]) != expected:
        raise ValueError(f"unexpected dataset columns: {sorted(rows[0] if rows else [])}")
    allowed = set(allowed_types)
    return [row for row in rows if row["item_type"] in allowed]


def direct_factual_question(row: Mapping[str, str]) -> str:
    """Reuse the validated extraction while removing old prospective turns."""
    from experiments.higher_v_readout_global.protocol import build_trial_protocol

    if row["item_type"] == "calibration":
        question = build_trial_protocol(dict(row)).factual_prompt
    else:
        question = row["detail"].strip()
    if not question:
        raise ValueError(f"item {row['item_id']} has no factual question")
    return question


def validate_config(config: Mapping[str, Any], dataset_rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    if config.get("experiment_name") != "process_sensitive_replay":
        raise ValueError("experiment name changed")
    if tuple(config.get("conditions", ())) != EXPECTED_CONDITIONS:
        raise ValueError("frozen condition order changed")
    model = config["model"]
    pinned_model_revision = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    if (
        model.get("revision") != pinned_model_revision
        or model.get("tokenizer_revision") != pinned_model_revision
    ):
        raise ValueError("model and tokenizer must use the frozen resolved revision")
    lens = config["lens"]
    if (
        lens.get("revision") != "0731326edff4ae730ffc5356fe1a4728c748b3a6"
        or lens.get("sha256")
        != "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1"
    ):
        raise ValueError("J-Lens revision or file SHA-256 changed")
    layers = config["layers"]
    if int(layers["process"]) != 31 or int(layers["meta_readout"]) != 40:
        raise ValueError("frozen process/meta layer changed")
    alternative_layers = [int(value) for value in layers["alternative_candidates"]]
    if alternative_layers != [15, 19, 23]:
        raise ValueError("declared psr-v8 alternative-layer set changed")
    if int(layers["minimum_alternative_separation"]) != 8:
        raise ValueError("frozen alternative-layer separation changed")
    if any(
        layer >= int(layers["process"])
        or int(layers["process"]) - layer < int(layers["minimum_alternative_separation"])
        or layer >= int(layers["meta_readout"])
        for layer in alternative_layers
    ):
        raise ValueError("alternative layers violate ordering/separation constraints")
    if (
        layers.get("expected_process_layer_type") != "full_attention"
        or layers.get("expected_alternative_layer_type") != "full_attention"
    ):
        raise ValueError("frozen intervention-layer architecture contract changed")
    if list(layers["readout"]) != list(range(36, 45)):
        raise ValueError("frozen J-Lens readout layers changed")
    strengths = config["strengths"]
    if [float(value) for value in strengths["alpha_grid"]] != [
        0.01, 0.02, 0.05, 0.1, 0.11, 0.125, 0.15, 0.2
    ]:
        raise ValueError("declared psr-v8 alpha grid changed")
    if [float(value) for value in strengths["beta_grid"]] != [
        0.05, 0.08, 0.1, 0.11, 0.125, 0.15, 0.2, 0.3, 0.4
    ]:
        raise ValueError("declared psr-v8 beta grid changed")
    alternative = config["alternative"]
    if alternative != {
        "objective": "full_answer_sequence_log_probability",
        "mechanism": "different_layer_same_objective_answer_support_reduction",
        "selection_inputs": [
            "support_match_quality", "finite_state_integrity", "perturbation_size",
        ],
        "prohibited_selection_inputs": [
            "candidate_j_space", "confidence", "correctness", "heldout",
        ],
        "max_median_norm_ratio_to_targeted": 4.0,
        "random_max_abs_cosine_with_answer_gradient": 0.1,
    }:
        raise ValueError("frozen psr-v8 alternative-mechanism contract changed")
    support = config["support_matching"]
    if (
        float(support["absolute_tolerance_nat"]) != 0.5
        or float(support["relative_tolerance"]) != 0.25
        or float(support["discovery_median_drop_relative_tolerance"]) != 0.25
        or float(support["heldout_min_item_match_fraction"]) != 0.65
        or support.get("item_match_requires_positive_targeted_drop") is not True
    ):
        raise ValueError("frozen held-out support-match rule changed")
    generation = config["generation"]
    if (
        int(generation["max_answer_tokens"]) != 256
        or generation.get("do_sample") is not False
        or float(generation.get("temperature")) != 0.0
        or generation.get("enable_thinking") is not False
        or generation.get("canonical_assistant_turn_terminator") != "<|im_end|>"
    ):
        raise ValueError("frozen answer generation or turn-termination contract changed")
    gradient_replay = config["gradient_replay"]
    if gradient_replay != {
        "method": "differentiable_token_by_token_recurrent",
        "reference": "ordinary_cached_token_by_token_recurrent",
        "functionalize_cache_updates_only": True,
        "parity_tolerance_source": "reset_parity",
        "hook_scope": "answer_predictor_positions_only",
    }:
        raise ValueError("frozen recurrent gradient replay contract changed")
    turn3_replay = config["turn3_replay"]
    if turn3_replay != {
        "construction": "suffix_only",
        "frozen_prefix_source": "answer_bank.post_answer_token_ids",
        "rerender_factual_history": False,
        "boundary_token": "<|im_end|>",
        "require_exact_token_hash_parity": True,
    }:
        raise ValueError("frozen suffix-only Turn-3 replay contract changed")
    if config.get("cuda_memory") != {
        "max_post_cleanup_growth_mib": 1024,
        "minimum_trend_step_mib": 128,
        "consecutive_growth_items": 2,
    }:
        raise ValueError("fail-closed CUDA memory-lifetime contract changed")
    candidate_selection = config["candidate_selection"]
    if candidate_selection != {
        "primary_branch": "confidence",
        "max_candidates": 3,
        "nontrivial_variance_epsilon": 1e-8,
        "max_support_adjusted_divergence_ratio": 1.0,
        "dedup_max_abs_direction_cosine": 0.9,
        "rank_metrics": [
            "item_centered_support_slope",
            "targeted_strong_effect",
            "support_matched_alternative_effect",
            "targeted_minus_random",
            "alternative_minus_random",
            "targeted_preserved_minus_reset",
            "support_adjusted_agreement",
            "item_sign_consistency",
        ],
    }:
        raise ValueError("frozen candidate-selection contract changed")
    if len(dataset_rows) != int(config["dataset"]["expected_items"]):
        raise ValueError(f"expected 82 factual items, found {len(dataset_rows)}")
    counts: dict[str, int] = {}
    for row in dataset_rows:
        counts[row["item_type"]] = counts.get(row["item_type"], 0) + 1
        direct_factual_question(row)
    expected_counts = {"calibration": 66, "prospective": 8, "knowledge_boundary": 8}
    if counts != expected_counts:
        raise ValueError(f"dataset type counts changed: {counts}")
    if set(PHASE_DEPENDENCIES) != {
        "validate", "answer_bank", "pre_discovery_smoke", "discovery",
        "freeze", "post_freeze_smoke", "heldout", "analyze",
    }:
        raise AssertionError("phase dependency graph changed")
    return {"item_count": len(dataset_rows), "item_type_counts": counts}


def _balanced_sample(
    rows: Sequence[Mapping[str, Any]], count: int, rng: random.Random
) -> list[Mapping[str, Any]]:
    correct = [row for row in rows if bool(row.get("factual_correct"))]
    incorrect = [row for row in rows if not bool(row.get("factual_correct"))]
    rng.shuffle(correct)
    rng.shuffle(incorrect)
    selected: list[Mapping[str, Any]] = []
    pools = [correct, incorrect]
    turn = 0
    while len(selected) < count and any(pools):
        pool = pools[turn % 2]
        if pool:
            selected.append(pool.pop())
        turn += 1
    if len(selected) != count:
        raise ValueError(f"cannot select {count} balanced rows from {len(rows)}")
    return selected


def allocate_discovery_split(
    answer_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, list[str]]:
    rng = random.Random(int(config["split"]["seed"]))
    chosen: list[Mapping[str, Any]] = []
    for item_type, count in config["split"]["discovery_counts"].items():
        pool = [row for row in answer_rows if row["item_type"] == item_type and not row.get("invalid", False)]
        try:
            chosen.extend(_balanced_sample(pool, int(count), rng))
        except ValueError as exc:
            raise ValueError(
                f"cannot allocate {count} discovery {item_type} items from "
                f"{len(pool)} valid answers"
            ) from exc
    discovery = sorted(str(row["item_id"]) for row in chosen)
    valid_item_ids = sorted(
        str(row["item_id"]) for row in answer_rows if not row.get("invalid", False)
    )
    excluded = sorted(
        str(row["item_id"]) for row in answer_rows if row.get("invalid", False)
    )
    heldout = sorted(set(valid_item_ids) - set(discovery))
    if len(discovery) != 16:
        raise ValueError(f"expected 16 discovery items, found {len(discovery)}")
    maximum_heldout = int(config["split"]["heldout_items"])
    if len(heldout) > maximum_heldout:
        raise ValueError(
            f"held-out assignments exceed frozen maximum {maximum_heldout}: {len(heldout)}"
        )
    return {
        "discovery_item_ids": discovery,
        "heldout_item_ids": heldout,
        "excluded_invalid_item_ids": excluded,
    }


def item_support_matched(
    support_drop_targeted: float,
    support_drop_alternative: float,
    *,
    absolute_tolerance_nat: float = 0.5,
    relative_tolerance: float = 0.25,
) -> bool:
    targeted = float(support_drop_targeted)
    alternative = float(support_drop_alternative)
    if not math.isfinite(targeted) or not math.isfinite(alternative) or targeted <= 0:
        return False
    tolerance = max(float(absolute_tolerance_nat), float(relative_tolerance) * abs(targeted))
    return abs(alternative - targeted) <= tolerance


def support_match_summary(
    pairs: Sequence[tuple[float, float]], config: Mapping[str, Any]
) -> dict[str, Any]:
    finite = [(float(t), float(a)) for t, a in pairs if math.isfinite(t) and math.isfinite(a)]
    if not finite:
        return {"passed": False, "reason": "no_finite_pairs", "valid_items": 0}
    support = config["support_matching"]
    targeted = [pair[0] for pair in finite]
    alternative = [pair[1] for pair in finite]
    signed = [a - t for t, a in finite]
    absolute = [abs(value) for value in signed]
    matched = [
        item_support_matched(
            t, a,
            absolute_tolerance_nat=float(support["absolute_tolerance_nat"]),
            relative_tolerance=float(support["relative_tolerance"]),
        )
        for t, a in finite
    ]
    target_median = median(targeted)
    alternative_median = median(alternative)
    aggregate_tolerance = max(
        float(support["absolute_tolerance_nat"]),
        float(support["relative_tolerance"]) * abs(target_median),
    )
    median_drop_matched = (
        target_median > 0
        and alternative_median > 0
        and abs(alternative_median - target_median)
        <= float(support["discovery_median_drop_relative_tolerance"]) * abs(target_median)
    )
    median_mismatch_matched = median(absolute) <= aggregate_tolerance
    match_fraction = sum(matched) / len(matched)
    fraction_matched = match_fraction >= float(support["heldout_min_item_match_fraction"])
    passed = median_drop_matched and median_mismatch_matched and fraction_matched
    return {
        "passed": passed,
        "valid_items": len(finite),
        "matched_items": sum(matched),
        "unmatched_items": len(matched) - sum(matched),
        "item_match_fraction": match_fraction,
        "required_item_match_fraction": float(support["heldout_min_item_match_fraction"]),
        "targeted_median_drop": target_median,
        "alternative_median_drop": alternative_median,
        "mean_signed_mismatch": sum(signed) / len(signed),
        "median_signed_mismatch": median(signed),
        "mean_absolute_mismatch": sum(absolute) / len(absolute),
        "median_absolute_mismatch": median(absolute),
        "aggregate_tolerance": aggregate_tolerance,
        "median_drop_matched": median_drop_matched,
        "median_mismatch_matched": median_mismatch_matched,
        "item_fraction_matched": fraction_matched,
    }


def validate_frozen_protocol(
    frozen: Mapping[str, Any],
    config: Mapping[str, Any],
    split: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Validate the immutable discovery-to-post-freeze handoff."""
    if int(frozen.get("schema_version", 0)) != 1:
        raise ValueError("unsupported frozen protocol schema")
    if frozen.get("experiment_name") != "process_sensitive_replay":
        raise ValueError("frozen protocol experiment changed")
    discovery_ids = [str(value) for value in frozen.get("discovery_item_ids", ())]
    expected_discovery = [str(value) for value in split["discovery_item_ids"]]
    if discovery_ids != expected_discovery or len(discovery_ids) != 16:
        raise ValueError("frozen protocol discovery split changed")
    heldout = {str(value) for value in split["heldout_item_ids"]}
    if set(discovery_ids) & heldout:
        raise ValueError("frozen protocol leaks held-out IDs into discovery")
    frozen_hashes = frozen.get("source_hashes", {})
    for name, expected in source_hashes.items():
        if frozen_hashes.get(name) != expected:
            raise ValueError(f"frozen protocol has stale or missing {name} hash")

    weak = float(frozen["weak_alpha"])
    strong = float(frozen["strong_alpha"])
    beta = float(frozen["beta"])
    alternative_layer = int(frozen["alternative_layer"])
    alpha_grid = [float(value) for value in config["strengths"]["alpha_grid"]]
    beta_grid = [float(value) for value in config["strengths"]["beta_grid"]]
    if weak not in alpha_grid or strong not in alpha_grid or weak >= strong:
        raise ValueError("frozen alpha strengths violate the discovery grid")
    if beta not in beta_grid:
        raise ValueError("frozen beta violates the discovery grid")
    if alternative_layer not in [
        int(value) for value in config["layers"]["alternative_candidates"]
    ]:
        raise ValueError("frozen alternative layer violates the discovery set")
    if frozen.get("candidate_selection") != config["candidate_selection"]:
        raise ValueError("frozen candidate-selection rule changed")
    if frozen.get("support_matching") != config["support_matching"]:
        raise ValueError("frozen support-matching rule changed")
    if frozen.get("conditions") != config["conditions"]:
        raise ValueError("frozen conditions changed")
    if frozen.get("meta_branches") != config["meta_branches"]:
        raise ValueError("frozen meta prompts changed")

    candidates = list(frozen.get("candidates", ()))
    maximum = int(config["candidate_selection"]["max_candidates"])
    if not 1 <= len(candidates) <= maximum:
        raise ValueError("frozen protocol must contain one to three candidates")
    seen: set[tuple[int, int]] = set()
    for candidate in candidates:
        identity = (int(candidate["token_id"]), int(candidate["layer"]))
        if identity in seen:
            raise ValueError("frozen protocol contains duplicate candidate identities")
        seen.add(identity)
        if identity[1] not in [int(value) for value in config["layers"]["readout"]]:
            raise ValueError("frozen candidate layer is outside the readout layers")
        if int(candidate["orientation"]) not in {-1, 1}:
            raise ValueError("frozen candidate orientation must be -1 or +1")
        if not candidate.get("direction_sha256") or not candidate.get("direction_file_sha256"):
            raise ValueError("frozen candidate direction hashes are missing")
    expected_token_ids = sorted({token_id for token_id, _layer in seen})
    if [int(value) for value in frozen.get("candidate_token_ids", ())] != expected_token_ids:
        raise ValueError("frozen candidate token ID index is inconsistent")
    return {
        "discovery_items": len(discovery_ids),
        "candidate_count": len(candidates),
        "weak_alpha": weak,
        "strong_alpha": strong,
        "beta": beta,
        "alternative_layer": alternative_layer,
    }


@dataclass(frozen=True)
class GateStatus:
    phase: str
    status: str
    protocol_hash: str
    input_hashes: dict[str, str]
    measurements: dict[str, Any]
    reason: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "passed"


def gate_path(run_dir: Path, phase: str) -> Path:
    if phase not in PHASE_DEPENDENCIES:
        raise ValueError(f"unknown phase {phase!r}")
    return run_dir / phase / "gate_status.json"


def phase_success_path(run_dir: Path, phase: str) -> Path:
    return run_dir / phase / "phase_success.json"


def read_gate(run_dir: Path, phase: str) -> GateStatus:
    path = gate_path(run_dir, phase)
    if not path.is_file():
        raise RuntimeError(f"missing prerequisite gate: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return GateStatus(**payload)


def assert_phase_prerequisites(
    run_dir: Path,
    phase: str,
    *,
    protocol_hash: str,
    required_input_hashes: Mapping[str, str],
) -> None:
    for dependency in PHASE_DEPENDENCIES[phase]:
        gate = read_gate(run_dir, dependency)
        if not gate.passed:
            raise RuntimeError(f"prerequisite {dependency} is {gate.status}, not passed")
        if gate.protocol_hash != protocol_hash:
            raise RuntimeError(f"stale protocol hash in {dependency} gate")
        for name, expected in required_input_hashes.items():
            observed = gate.input_hashes.get(name)
            if observed is None:
                raise RuntimeError(f"missing required {name} hash in {dependency} gate")
            if observed != expected:
                raise RuntimeError(f"stale {name} hash in {dependency} gate")
        success_path = phase_success_path(run_dir, dependency)
        if not success_path.is_file():
            raise RuntimeError(f"missing success marker for {dependency}")
        success = json.loads(success_path.read_text(encoding="utf-8"))
        if success.get("phase") != dependency or success.get("gate_sha256") != sha256_file(gate_path(run_dir, dependency)):
            raise RuntimeError(f"stale or invalid success marker for {dependency}")


def write_gate(run_dir: Path, gate: GateStatus) -> Path:
    directory = gate_path(run_dir, gate.phase).parent
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "gate_status.json"
    temporary = directory / "gate_status.json.tmp"
    temporary.write_text(json.dumps(asdict(gate), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    success = phase_success_path(run_dir, gate.phase)
    if gate.passed:
        success.write_text(
            json.dumps({"phase": gate.phase, "gate_sha256": sha256_file(path)}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif success.exists():
        success.unlink()
    return path
