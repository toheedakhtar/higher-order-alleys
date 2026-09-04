from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.process_sensitive_replay.protocol import GateStatus, load_config, write_gate
from experiments.process_sensitive_replay.runner import (
    _load_campaign_inputs,
    _load_pre_discovery_smoke_hash,
    campaign_hashes,
    combined_protocol_hash,
    initialize_run_dir,
    run_freeze_phase,
)


CONFIG_PATH = Path(__file__).with_name("experiment_config.json")


class FreezePhaseTests(unittest.TestCase):
    def test_freeze_reports_failed_discovery_before_missing_artifacts(self) -> None:
        config = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            initialize_run_dir(run_dir, CONFIG_PATH, config)
            hashes = campaign_hashes(CONFIG_PATH, config)
            write_gate(run_dir, GateStatus(
                phase="discovery",
                status="invalid_support_match",
                protocol_hash=combined_protocol_hash(hashes),
                input_hashes=dict(hashes),
                measurements={},
                reason="support_match_gate_failed",
            ))

            with self.assertRaisesRegex(
                RuntimeError,
                "prerequisite discovery is invalid_support_match, not passed",
            ):
                run_freeze_phase(run_dir, config, hashes)

    def test_freeze_writes_hash_bound_protocol_without_loading_model(self) -> None:
        config = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            initialize_run_dir(run_dir, CONFIG_PATH, config)
            (run_dir / "answer_bank.jsonl").write_text("{}\n", encoding="utf-8")
            split = {
                "discovery_item_ids": [str(value) for value in range(16)],
                "heldout_item_ids": [str(value) for value in range(16, 73)],
                "excluded_invalid_item_ids": [],
            }
            (run_dir / "split_manifest.json").write_text(
                json.dumps(split), encoding="utf-8"
            )
            answer_dir = run_dir / "answer_bank"
            answer_dir.mkdir()
            (answer_dir / "thinking_mode_verification.json").write_text(
                "{}\n", encoding="utf-8"
            )
            smoke_dir = run_dir / "pre_discovery_smoke"
            smoke_dir.mkdir()
            (smoke_dir / "trials.jsonl").write_text("{}\n", encoding="utf-8")
            (smoke_dir / "smoke_report.json").write_text("{}\n", encoding="utf-8")
            (smoke_dir / "trial_summary.csv").write_text("item_id\n", encoding="utf-8")
            (smoke_dir / "candidate_scores.csv").write_text("item_id\n", encoding="utf-8")
            discovery_dir = run_dir / "discovery"
            discovery_dir.mkdir()
            for filename in (
                "alpha_grid.jsonl",
                "alpha_grid_diagnostics.json",
                "beta_grid.jsonl",
                "beta_grid_diagnostics.json",
                "strength_grid.jsonl",
                "discovery_vocab_scores.pt",
                "candidate_metrics.pt",
                "trial_summary.csv",
                "candidate_scores.csv",
            ):
                (discovery_dir / filename).write_bytes(filename.encode("utf-8"))
            directions = discovery_dir / "directions"
            directions.mkdir()
            direction = directions / "candidate.pt"
            direction.write_bytes(b"direction")

            hashes = campaign_hashes(CONFIG_PATH, config)
            _load_campaign_inputs(run_dir, hashes)
            _load_pre_discovery_smoke_hash(run_dir, hashes)
            from experiments.process_sensitive_replay.protocol import sha256_file

            hashes["discovery_strength_grid"] = sha256_file(
                discovery_dir / "strength_grid.jsonl"
            )
            hashes["discovery_alpha_grid"] = sha256_file(
                discovery_dir / "alpha_grid.jsonl"
            )
            hashes["discovery_alpha_diagnostics"] = sha256_file(
                discovery_dir / "alpha_grid_diagnostics.json"
            )
            hashes["discovery_beta_grid"] = sha256_file(
                discovery_dir / "beta_grid.jsonl"
            )
            hashes["discovery_beta_diagnostics"] = sha256_file(
                discovery_dir / "beta_grid_diagnostics.json"
            )
            hashes["discovery_vocab_scores"] = sha256_file(
                discovery_dir / "discovery_vocab_scores.pt"
            )
            hashes["candidate_metrics"] = sha256_file(
                discovery_dir / "candidate_metrics.pt"
            )
            hashes["discovery_trial_summary"] = sha256_file(
                discovery_dir / "trial_summary.csv"
            )
            hashes["discovery_candidate_scores"] = sha256_file(
                discovery_dir / "candidate_scores.csv"
            )
            candidate = {
                "token_id": 123,
                "layer": 40,
                "orientation": 1,
                "direction_sha256": "tensor-digest",
                "direction_path": "discovery/directions/candidate.pt",
                "direction_file_sha256": sha256_file(direction),
            }
            discovery = {
                "discovery_item_ids": split["discovery_item_ids"],
                "heldout_item_ids_accessed": [],
                "source_hashes": dict(hashes),
                "weak_alpha": 0.02,
                "strong_alpha": 0.1,
                "beta": 0.2,
                "alternative_layer": 19,
                "strength_selection": {},
                "candidates": [candidate],
                "candidate_token_ids": [123],
            }
            discovery_path = discovery_dir / "candidate_discovery.json"
            discovery_path.write_text(json.dumps(discovery), encoding="utf-8")
            hashes["candidate_discovery"] = sha256_file(discovery_path)
            write_gate(run_dir, GateStatus(
                phase="discovery",
                status="passed",
                protocol_hash=combined_protocol_hash(hashes),
                input_hashes=dict(hashes),
                measurements={},
            ))

            result = run_freeze_phase(
                run_dir, config, campaign_hashes(CONFIG_PATH, config)
            )

            self.assertEqual(result["candidate_count"], 1)
            frozen = json.loads(
                (run_dir / "frozen_protocol.json").read_text(encoding="utf-8")
            )
            self.assertEqual(frozen["candidate_token_ids"], [123])
            self.assertFalse(frozen["heldout_access_permitted"])


if __name__ == "__main__":
    unittest.main()
