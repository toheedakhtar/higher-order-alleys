from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.process_sensitive_replay.discovery import (
    AlphaTrial, BetaTrial, select_alpha_strengths, select_beta_strength,
)
from experiments.process_sensitive_replay.protocol import (
    GateStatus,
    allocate_discovery_split,
    assert_phase_prerequisites,
    item_support_matched,
    load_config,
    load_dataset,
    phase_success_path,
    sha256_file,
    support_match_summary,
    validate_frozen_protocol,
    validate_config,
    write_gate,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("experiment_config.json")


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)

    def test_repository_config_and_dataset_are_frozen(self) -> None:
        rows = load_dataset(
            ROOT / self.config["dataset"]["path"],
            self.config["dataset"]["item_types"],
        )
        result = validate_config(self.config, rows)
        self.assertEqual(result["item_count"], 82)
        self.assertEqual(result["item_type_counts"]["calibration"], 66)
        self.assertEqual(self.config["generation"]["max_answer_tokens"], 256)
        self.assertIs(self.config["generation"]["enable_thinking"], False)
        self.assertEqual(
            self.config["generation"]["canonical_assistant_turn_terminator"],
            "<|im_end|>",
        )
        self.assertEqual(
            self.config["gradient_replay"]["method"],
            "differentiable_token_by_token_recurrent",
        )
        self.assertEqual(
            self.config["gradient_replay"]["hook_scope"],
            "answer_predictor_positions_only",
        )
        self.assertEqual(self.config["turn3_replay"]["construction"], "suffix_only")
        self.assertIs(self.config["turn3_replay"]["rerender_factual_history"], False)
        self.assertIs(
            self.config["turn3_replay"]["require_exact_token_hash_parity"], True
        )

    def test_item_support_match_uses_absolute_target_and_requires_positive_drop(self) -> None:
        self.assertTrue(item_support_matched(1.0, 1.5))
        self.assertFalse(item_support_matched(1.0, 1.500001))
        self.assertTrue(item_support_matched(4.0, 5.0))
        self.assertFalse(item_support_matched(-4.0, -3.0))
        self.assertFalse(item_support_matched(0.0, 0.0))

    def test_heldout_fraction_is_fail_closed_at_sixty_five_percent(self) -> None:
        passing = [(2.0, 2.1)] * 13
        failing = [(-1.0, -1.0)] * 7
        result = support_match_summary([*passing, *failing], self.config)
        self.assertTrue(result["passed"])
        self.assertEqual(result["matched_items"], 13)
        self.assertEqual(result["item_match_fraction"], 0.65)
        result = support_match_summary([*passing[:12], *failing, (2.0, 3.0)], self.config)
        self.assertFalse(result["passed"])
        self.assertLess(result["item_match_fraction"], 0.65)

    def test_frozen_protocol_binds_split_hashes_strengths_and_candidates(self) -> None:
        split = {
            "discovery_item_ids": [str(value) for value in range(16)],
            "heldout_item_ids": [str(value) for value in range(16, 73)],
        }
        hashes = {"code": "abc", "candidate_discovery": "def"}
        frozen = {
            "schema_version": 1,
            "experiment_name": "process_sensitive_replay",
            "source_hashes": dict(hashes),
            "discovery_item_ids": list(split["discovery_item_ids"]),
            "weak_alpha": 0.02,
            "strong_alpha": 0.1,
            "beta": 0.2,
            "candidate_selection": self.config["candidate_selection"],
            "support_matching": self.config["support_matching"],
            "conditions": self.config["conditions"],
            "meta_branches": self.config["meta_branches"],
            "candidates": [{
                "token_id": 123,
                "layer": 40,
                "orientation": 1,
                "direction_sha256": "tensor",
                "direction_file_sha256": "file",
            }],
            "candidate_token_ids": [123],
        }
        result = validate_frozen_protocol(frozen, self.config, split, hashes)
        self.assertEqual(result["candidate_count"], 1)
        stale = json.loads(json.dumps(frozen))
        stale["source_hashes"]["code"] = "changed"
        with self.assertRaisesRegex(ValueError, "stale or missing code hash"):
            validate_frozen_protocol(stale, self.config, split, hashes)

    def test_split_is_deterministic_and_stratified(self) -> None:
        rows = []
        item_id = 0
        for item_type, count in (("calibration", 66), ("prospective", 8), ("knowledge_boundary", 8)):
            for index in range(count):
                rows.append({
                    "item_id": str(item_id),
                    "item_type": item_type,
                    "factual_correct": index % 2 == 0,
                    "invalid": False,
                })
                item_id += 1
        first = allocate_discovery_split(rows, self.config)
        second = allocate_discovery_split(rows, self.config)
        self.assertEqual(first, second)
        self.assertEqual(len(first["discovery_item_ids"]), 16)
        self.assertEqual(len(first["heldout_item_ids"]), 66)
        by_id = {row["item_id"]: row for row in rows}
        counts = {}
        correctness = {}
        for selected in first["discovery_item_ids"]:
            row = by_id[selected]
            counts[row["item_type"]] = counts.get(row["item_type"], 0) + 1
            correctness.setdefault(row["item_type"], set()).add(row["factual_correct"])
        self.assertEqual(counts, self.config["split"]["discovery_counts"])
        self.assertTrue(all(values == {False, True} for values in correctness.values()))
        rows[0]["invalid"] = True
        split_with_invalid = allocate_discovery_split(rows, self.config)
        self.assertEqual(len(split_with_invalid["heldout_item_ids"]), 65)
        self.assertNotIn("0", split_with_invalid["discovery_item_ids"])
        self.assertNotIn("0", split_with_invalid["heldout_item_ids"])
        self.assertEqual(split_with_invalid["excluded_invalid_item_ids"], ["0"])


class DiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)

    def test_alpha_selection_uses_complete_declared_grid(self) -> None:
        trials = [
            AlphaTrial(alpha, tuple([drop] * 16))
            for alpha, drop in zip(
                self.config["strengths"]["alpha_grid"],
                (0.1, 0.3, 0.7, 1.0, 2.2, 3.0, 4.0, 5.0),
                strict=True,
            )
        ]
        selected = select_alpha_strengths(trials, self.config)
        self.assertEqual(selected["weak_alpha"], 0.1)
        self.assertEqual(selected["strong_alpha"], 0.11)

    def test_beta_selection_minimizes_mismatch_subject_to_all_gates(self) -> None:
        trials = []
        for beta in self.config["strengths"]["beta_grid"]:
            alternative = 2.0 + abs(float(beta) - 0.2)
            trials.append(BetaTrial(
                float(beta), tuple([2.0] * 16), tuple([alternative] * 16),
                median_norm_ratio=2.0, max_abs_cosine=0.05,
            ))
        selected = select_beta_strength(trials, self.config)
        self.assertEqual(selected["beta"], 0.2)

    def test_beta_selection_rejects_norm_or_cosine_failures(self) -> None:
        trials = [
            BetaTrial(
                float(beta), tuple([2.0] * 16), tuple([2.0] * 16),
                median_norm_ratio=5.0, max_abs_cosine=0.05,
            )
            for beta in self.config["strengths"]["beta_grid"]
        ]
        with self.assertRaisesRegex(RuntimeError, "support_match_gate_failed"):
            select_beta_strength(trials, self.config)


class GateTests(unittest.TestCase):
    def test_missing_failed_and_stale_gates_block_next_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "missing prerequisite"):
                assert_phase_prerequisites(
                    run_dir, "answer_bank", protocol_hash="p", required_input_hashes={"dataset": "d"}
                )
            write_gate(run_dir, GateStatus(
                phase="validate", status="failed", protocol_hash="p",
                input_hashes={"dataset": "d"}, measurements={}, reason="test",
            ))
            self.assertFalse(phase_success_path(run_dir, "validate").exists())
            with self.assertRaisesRegex(RuntimeError, "not passed"):
                assert_phase_prerequisites(
                    run_dir, "answer_bank", protocol_hash="p", required_input_hashes={"dataset": "d"}
                )
            write_gate(run_dir, GateStatus(
                phase="validate", status="passed", protocol_hash="p",
                input_hashes={"dataset": "d"}, measurements={},
            ))
            assert_phase_prerequisites(
                run_dir, "answer_bank", protocol_hash="p", required_input_hashes={"dataset": "d"}
            )
            marker = phase_success_path(run_dir, "validate")
            payload = json.loads(marker.read_text(encoding="utf-8"))
            payload["gate_sha256"] = "stale"
            marker.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "stale or invalid success marker"):
                assert_phase_prerequisites(
                    run_dir, "answer_bank", protocol_hash="p", required_input_hashes={"dataset": "d"}
                )


if __name__ == "__main__":
    unittest.main()
