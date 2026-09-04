"""Read-only recovery of the exact frozen upstream identity and answer bank."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch

from .precision import vector_hash

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPSTREAM = ROOT / "assets/psr-quick-v3"
DEFAULT_DIRECTION = ROOT / "assets/psr_quick_big_files/directions/layer42_token75075_8515912d78e8.pt"
FROZEN_SHA = "92c6f7022ec7025313f0da6f68cb0f3b9a27db1dfe35d09b51d871236ed7ee29"
ITEM_IDS = ("0", "2", "3", "4", "57", "67", "68", "82")
SMOKE_IDS = ITEM_IDS[:2]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_sha(value) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def checked_bytes(path: Path, expected: str, *, text: bool = False) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    mode = "byte_exact"
    recovered = raw
    if sha(raw) != expected and text:
        recovered = raw.replace(b"\r\n", b"\n")
        mode = "CRLF_to_LF_transfer_recovery"
    if sha(recovered) != expected:
        raise RuntimeError(f"upstream artifact hash mismatch: {path}")
    return recovered, {"path": str(path), "expected_sha256": expected,
                       "local_sha256": sha(raw), "verification": mode}


def load_upstream(upstream=DEFAULT_UPSTREAM, direction_path=DEFAULT_DIRECTION):
    upstream, direction_path = Path(upstream), Path(direction_path)
    frozen_bytes, frozen_audit = checked_bytes(upstream / "frozen_protocol.json", FROZEN_SHA, text=True)
    frozen = json.loads(frozen_bytes)
    candidate = frozen["candidates"][0]
    if len(frozen["candidates"]) != 1 or (candidate["token_id"], candidate["layer"], candidate["orientation"]) != (75075, 42, -1):
        raise RuntimeError("unexpected frozen candidate")
    _, direction_audit = checked_bytes(direction_path, candidate["direction_file_sha256"])
    saved = torch.load(direction_path, map_location="cpu", weights_only=True)
    vector = saved["direction"]
    if (vector.shape != (5120,) or vector.dtype != torch.float32
            or not bool(torch.isfinite(vector).all())
            or abs(float(vector.double().norm()) - 1) > 1e-5
            or vector_hash(vector) != candidate["direction_sha256"]
            or saved["sha256"] != candidate["direction_sha256"]
            or saved["token_id"] != 75075 or saved["layer"] != 42):
        raise RuntimeError("frozen vector content mismatch")
    # Read only upstream identity, not scientific effect tables or trial outputs.
    manifest = json.loads((upstream / "run_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((upstream / "config.json").read_text(encoding="utf-8"))
    from experiments.process_sensitive_replay.protocol import sha256_json
    if sha256_json(config) != frozen["source_hashes"]["config"]:
        raise RuntimeError("upstream resolved config mismatch")
    packages = manifest["packages"]
    if sha256_json(packages) != frozen["source_hashes"]["runtime_packages"]:
        raise RuntimeError("upstream package identity mismatch")
    for name, actual in (
        ("model_revision", config["model"]["revision"]),
        ("tokenizer_revision", config["model"]["tokenizer_revision"]),
        ("lens_revision", config["lens"]["revision"]),
        ("lens_sha256", config["lens"]["sha256"]),
    ):
        if actual != frozen["source_hashes"][name]:
            raise RuntimeError(f"upstream {name} mismatch")
    if (config["model"]["dtype"] != "bfloat16" or frozen["strong_alpha"] != 0.11
            or frozen["beta"] != 0.20 or frozen["alternative_layer"] != 23
            or config["layers"]["process"] != 31
            or frozen["gradient_answer_token_limit"] != 32
            or config["execution_profile"]["gradient_answer_token_limit"] != 32
            or config["generation"]["enable_thinking"] is not False
            or config["turn3_replay"]["construction"] != "suffix_only"
            or config["reset_parity"]["absolute_tolerance"] != 1e-5
            or config["reset_parity"]["relative_tolerance"] != 1e-5):
        raise RuntimeError("upstream execution contract mismatch")
    answer_bytes, answer_audit = checked_bytes(upstream / "answer_bank.jsonl", frozen["source_hashes"]["answer_bank"], text=True)
    split_bytes, split_audit = checked_bytes(upstream / "split_manifest.json", frozen["source_hashes"]["split_manifest"], text=True)
    if tuple(json.loads(split_bytes)["heldout_item_ids"]) != ITEM_IDS:
        raise RuntimeError("held-out membership/order changed")
    answers = {str(row["item_id"]): row for row in map(json.loads, answer_bytes.decode().splitlines())}
    if any(item not in answers for item in ITEM_IDS):
        raise RuntimeError("missing predeclared answer")
    return vector, config, answers, {
        "frozen_candidate": {k: candidate[k] for k in ("token_id", "layer", "orientation", "direction_sha256", "direction_file_sha256")},
        "frozen_protocol": frozen_audit, "direction": direction_audit,
        "answer_bank": answer_audit, "split": split_audit,
        "runtime_packages": packages, "smoke_item_ids": list(SMOKE_IDS),
        "heldout_item_ids": list(ITEM_IDS), "upstream_hashes": frozen["source_hashes"],
        "upstream_manifest_hashes": manifest["hashes"],
    }


def require_runtime_packages(expected):
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"exact CUDA runtime packages required; expected={expected}, actual={actual}")
    if not torch.cuda.is_available():
        raise RuntimeError("precision smoke requires the original CUDA runtime; CPU diagnostics cannot qualify it")
    return actual
