from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.process_sensitive_replay.analysis import (
    CONDITION_LABELS,
    analyze_candidate_effects,
    generate_required_plots,
)
from experiments.process_sensitive_replay.protocol import GateStatus, load_config, write_gate
from experiments.process_sensitive_replay.runner import (
    SUPPORTED_PHASES,
    _candidate_support_row,
    _load_campaign_inputs,
    _load_discovery_hashes,
    _load_pre_discovery_smoke_hash,
    _load_post_freeze_smoke_hash,
    campaign_hashes,
    combined_protocol_hash,
    heldout_effect_rows,
    heldout_support_rows,
    initialize_run_dir,
    run_analyze_phase,
    run_heldout_phase,
)


CONFIG_PATH = Path(__file__).with_name("experiment_config.json")


class PostFreezePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)

    def test_all_frozen_post_freeze_phases_are_callable(self) -> None:
        self.assertIn("post_freeze_smoke", SUPPORTED_PHASES)
        self.assertIn("heldout", SUPPORTED_PHASES)
        self.assertIn("analyze", SUPPORTED_PHASES)

    def test_campaign_hashes_bind_runtime_and_checkpoint_identities(self) -> None:
        hashes = campaign_hashes(self.config)
        self.assertEqual(
            hashes["model_revision"], self.config["model"]["revision"]
        )
        self.assertEqual(
            hashes["tokenizer_revision"],
            self.config["model"]["tokenizer_revision"],
        )
        self.assertEqual(hashes["lens_sha256"], self.config["lens"]["sha256"])
        self.assertIn("runtime_packages", hashes)
        baseline = combined_protocol_hash(hashes)
        changed = {**hashes, "runtime_packages": "different"}
        self.assertNotEqual(baseline, combined_protocol_hash(changed))

    def test_heldout_support_rows_enforce_positive_target_and_item_tolerance(self) -> None:
        records = [
            {"item_id": "a", "support": {"targeted_drop": 2.0, "alternative_drop": 2.5}},
            {"item_id": "b", "support": {"targeted_drop": -2.0, "alternative_drop": -2.0}},
            {"item_id": "c", "support": {"targeted_drop": 4.0, "alternative_drop": 5.01}},
        ]
        rows = heldout_support_rows(records, self.config)
        self.assertTrue(rows[0]["support_matched"])
        self.assertFalse(rows[1]["support_matched"])
        self.assertFalse(rows[2]["support_matched"])
        self.assertEqual(rows[2]["item_tolerance"], 1.0)

    def test_reset_retains_targeted_process_support_while_state_is_reset(self) -> None:
        row = _candidate_support_row(
            {
                "support": {
                    "clean": -1.0,
                    "target_grid": {"0.1": -1.5, "0.11": -3.0},
                    "random_drop": 0.2,
                    "alternative_drop": 2.1,
                    "alternative_random_drop": 0.3,
                }
            },
            weak_alpha=0.1,
            strong_alpha=0.11,
        )
        self.assertEqual(row[0], 0.0)
        self.assertEqual(row[2], 2.0)
        self.assertEqual(row[6], 2.0)

    def test_effect_extraction_and_model_free_analysis_cover_h1_to_h7_and_plots(self) -> None:
        frozen = {
            "weak_alpha": 0.1,
            "alternative_layer": 19,
            "candidates": [{
                "label": " reliability",
                "layer": 40,
                "token_id": 123,
                "orientation": 1,
            }],
        }
        records = []
        for item in range(4):
            meta = {}
            condition_scores = {
                "clean_preserved": 0.0,
                "targeted_weak_preserved": 0.5,
                "targeted_strong_preserved": 2.0,
                "random_strong_preserved": 0.25,
                "support_matched_alternative_preserved": 1.9,
                "alternative_random_preserved": 0.3,
                "targeted_strong_reset": 0.0,
            }
            for condition, candidate_score in condition_scores.items():
                meta[condition] = {}
                for branch in self.config["meta_branches"]:
                    meta[condition][branch] = {
                        "scores": {"margin": 1.0 - candidate_score * 0.1},
                        "jlens": {
                            str(layer): {
                                "explicit": {
                                    "123": {"score": candidate_score + item * 0.01}
                                }
                            }
                            for layer in self.config["layers"]["readout"]
                        },
                    }
            records.append({
                "item_id": str(item),
                "support": {
                    "clean": -1.0,
                    "target_grid": {"0.1": -1.5},
                    "targeted_drop": 2.0,
                    "alternative_drop": 2.1,
                    "random_drop": 0.25,
                    "alternative_random_drop": 0.3,
                },
                "meta": meta,
            })
        effect_rows = heldout_effect_rows(records, frozen, self.config)
        reset_rows = [
            row for row in effect_rows
            if row["condition"] == "targeted_strong_reset"
        ]
        self.assertTrue(reset_rows)
        self.assertTrue(all(row["support_drop"] == 2.0 for row in reset_rows))
        statistics = analyze_candidate_effects(effect_rows)
        self.assertEqual(len(statistics), 2)
        confidence = next(row for row in statistics if row["branch"] == "confidence")
        for hypothesis in ("h1_", "h2_", "h3_", "h4_", "h5_", "h6_", "h7_"):
            self.assertTrue(any(key.startswith(hypothesis) for key in confidence))
        self.assertIn("process-property", confidence["interpretation_ceiling"])
        self.assertIn("h4_alternative_minus_random", confidence)
        self.assertGreater(
            confidence["h4_alternative_minus_random"]["mean"], 0
        )

        support_rows = heldout_support_rows(records, self.config)
        candidate_score_rows = []
        for record in records:
            for condition in CONDITION_LABELS:
                for layer in self.config["layers"]["readout"]:
                    for token_id in (123, *self.config["readout"]["generic_evaluator_token_ids"]):
                        candidate_score_rows.append({
                            "item_id": record["item_id"],
                            "condition": condition,
                            "branch": "confidence",
                            "layer": layer,
                            "token_id": token_id,
                            "score": float(token_id % 10) + layer * 0.01,
                        })
        with tempfile.TemporaryDirectory() as temporary:
            paths = generate_required_plots(
                effect_rows,
                support_rows,
                candidate_score_rows,
                Path(temporary),
                primary_branch="confidence",
                generic_token_ids=self.config["readout"]["generic_evaluator_token_ids"],
            )
            self.assertEqual(len(paths), 15)
            self.assertTrue(all(path.is_file() and path.stat().st_size for path in paths))

    def _fake_record(self, item_id: str) -> dict:
        meta = {}
        scores = {
            "clean_preserved": 0.0,
            "targeted_weak_preserved": 0.5,
            "targeted_strong_preserved": 2.0,
            "random_strong_preserved": 0.2,
            "support_matched_alternative_preserved": 1.9,
            "alternative_random_preserved": 0.3,
            "targeted_strong_reset": 0.0,
            "clean_reset": 0.0,
        }
        explicit_ids = [123, *self.config["readout"]["generic_evaluator_token_ids"]]
        for condition, value in scores.items():
            meta[condition] = {}
            for branch in self.config["meta_branches"]:
                meta[condition][branch] = {
                    "scores": {"margin": 1.0 - value * 0.1},
                    "jlens": {
                        str(layer): {
                            "explicit": {
                                str(token_id): {
                                    "score": value + layer * 0.001,
                                    "raw_rank": 10,
                                }
                                for token_id in explicit_ids
                            }
                        }
                        for layer in self.config["layers"]["readout"]
                    },
                }
        return {
            "item_id": item_id,
            "phase": "heldout",
            "support": {
                "clean": -1.0,
                "target_grid": {"0.1": -1.5, "0.11": -3.0},
                "targeted_drop": 2.0,
                "alternative_drop": 2.1,
                "random_drop": 0.2,
                "alternative_random_drop": 0.3,
            },
            "checks": {"reset_parity": True, "hybrid_cache_integrity": True},
            "meta": meta,
        }

    def test_heldout_and_analysis_execute_after_hash_gated_post_freeze_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            initialize_run_dir(run_dir, self.config)
            answers = [
                {"item_id": str(item), "invalid": False} for item in range(18)
            ]
            (run_dir / "answer_bank.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in answers), encoding="utf-8"
            )
            split = {
                "discovery_item_ids": [str(item) for item in range(16)],
                "heldout_item_ids": ["16", "17"],
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
            pre = run_dir / "pre_discovery_smoke"
            pre.mkdir()
            for filename in (
                "trials.jsonl", "smoke_report.json", "trial_summary.csv",
                "candidate_scores.csv", "cuda_memory.jsonl",
            ):
                (pre / filename).write_text("{}\n", encoding="utf-8")
            discovery = run_dir / "discovery"
            discovery.mkdir()
            for filename in (
                "alpha_grid.jsonl", "alpha_grid_diagnostics.json", "beta_grid.jsonl",
                "beta_grid_diagnostics.json", "strength_grid.jsonl",
                "discovery_vocab_scores.pt", "candidate_metrics.pt", "trial_summary.csv",
                "candidate_scores.csv", "candidate_discovery.json",
                "cuda_memory.jsonl",
            ):
                (discovery / filename).write_text(filename + "\n", encoding="utf-8")
            directions = discovery / "directions"
            directions.mkdir()
            direction = directions / "candidate.pt"
            direction.write_bytes(b"direction")

            hashes = campaign_hashes(self.config)
            _load_campaign_inputs(run_dir, hashes)
            _load_pre_discovery_smoke_hash(run_dir, hashes)
            _load_discovery_hashes(run_dir, hashes)
            from experiments.process_sensitive_replay.protocol import sha256_file

            candidate = {
                "token_id": 123,
                "layer": 40,
                "label": " reliability",
                "orientation": 1,
                "direction_sha256": "tensor-digest",
                "direction_path": "discovery/directions/candidate.pt",
                "direction_file_sha256": sha256_file(direction),
            }
            frozen = {
                "schema_version": 1,
                "experiment_name": "process_sensitive_replay",
                "answer_support_objective": self.config["alternative"]["objective"],
                "gradient_answer_token_limit": None,
                "source_hashes": dict(hashes),
                "discovery_item_ids": split["discovery_item_ids"],
                "weak_alpha": 0.1,
                "strong_alpha": 0.11,
                "beta": 0.2,
                "alternative_layer": 19,
                "candidate_selection": self.config["candidate_selection"],
                "support_matching": self.config["support_matching"],
                "conditions": self.config["conditions"],
                "meta_branches": self.config["meta_branches"],
                "candidates": [candidate],
                "candidate_token_ids": [123],
                "heldout_access_permitted": False,
            }
            (run_dir / "frozen_protocol.json").write_text(
                json.dumps(frozen), encoding="utf-8"
            )
            hashes["frozen_protocol"] = sha256_file(run_dir / "frozen_protocol.json")
            post = run_dir / "post_freeze_smoke"
            post.mkdir()
            for filename in (
                "trials.jsonl", "smoke_report.json", "trial_summary.csv",
                "candidate_scores.csv", "cuda_memory.jsonl",
            ):
                (post / filename).write_text("{}\n", encoding="utf-8")
            _load_post_freeze_smoke_hash(run_dir, hashes)
            write_gate(run_dir, GateStatus(
                phase="post_freeze_smoke",
                status="passed",
                protocol_hash=combined_protocol_hash(hashes),
                input_hashes=dict(hashes),
                measurements={},
            ))
            (run_dir / "heldout").mkdir()
            args = argparse.Namespace(phase="heldout", hf_cache_dir=None)
            fake_records = [self._fake_record("16"), self._fake_record("17")]

            def run_cuda_stub(_guard, memory_path, _hashes, **kwargs):
                with memory_path.open("a", encoding="utf-8") as handle:
                    handle.write("{}\n")
                return kwargs["operation"]()

            with (
                patch(
                    "experiments.process_sensitive_replay.runner.load_runtime",
                    return_value=(None, None, None, None, None, {}),
                ),
                patch(
                    "experiments.process_sensitive_replay.runner.run_smoke_item",
                    side_effect=fake_records,
                ),
                patch(
                    "experiments.process_sensitive_replay.runner.run_cuda_item",
                    side_effect=run_cuda_stub,
                ),
                patch("experiments.process_sensitive_replay.runner._log_smoke_record"),
                patch(
                    "experiments.process_sensitive_replay.runner.summarize_smoke",
                    return_value={"passed": True},
                ),
            ):
                heldout = run_heldout_phase(
                    args, run_dir, self.config, campaign_hashes(self.config)
                )
            self.assertEqual(heldout["heldout_items"], 2)
            self.assertTrue((run_dir / "heldout_effects.csv").is_file())
            self.assertTrue((run_dir / "heldout" / "support_match_diagnostic.png").is_file())

            (run_dir / "analyze").mkdir()
            analyzed = run_analyze_phase(
                run_dir, self.config, campaign_hashes(self.config)
            )
            self.assertEqual(analyzed["plot_count"], 15)
            self.assertTrue((run_dir / "analyze" / "RESULTS.md").is_file())
            results = (run_dir / "analyze" / "RESULTS.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("H1 targeted", results)
            self.assertIn("H4 alternative", results)
            self.assertIn("H7 confidence", results)
            self.assertIn("does **not** automatically classify", results)


if __name__ == "__main__":
    unittest.main()
