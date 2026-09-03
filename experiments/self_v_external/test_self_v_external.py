"""CPU-only tests for the matched SELF-versus-OTHER experiment."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.higher_v_readout_global import protocol as base_protocol
from experiments.higher_v_readout_global import runner as base_runner
from experiments.higher_v_readout_global.test_global_experiment import (
    TinyLens,
    TinyModel,
    TinyTokenizer,
    fake_candidate,
)

from . import analysis, protocol, runner


class SelfVersusExternalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = runner.load_config(runner.DEFAULT_CONFIG)
        self.rows = base_protocol.load_dataset_rows(
            runner.REPO_ROOT / "dataset/metacognition.csv"
        )

    def test_static_contract_selects_82_factual_items(self):
        report = runner.validate_config(self.config, self.rows)
        self.assertEqual(report["paired_item_count"], 82)
        self.assertEqual(report["condition_count"], 164)
        self.assertEqual(report["excluded_item_types"], ["error_detection"])
        self.assertFalse(report["candidate_visibility_gate"])

    def test_pair_differs_only_in_your_their(self):
        row = next(item for item in self.rows if item["item_type"] == "calibration")
        paired = protocol.build_paired_protocol(row)
        self_messages = protocol.evaluation_messages(
            paired.factual_prompt, "answer X exactly", paired.self_evaluation_prompt
        )
        other_messages = protocol.evaluation_messages(
            paired.factual_prompt, "answer X exactly", paired.other_evaluation_prompt
        )
        protocol.assert_matched_pair(self_messages, other_messages)
        self.assertEqual(self_messages[:2], other_messages[:2])
        self.assertEqual(
            self_messages[-1]["content"].replace(" your ", " their ", 1),
            other_messages[-1]["content"],
        )

    def test_label_family_follows_original_prompt(self):
        calibration = next(item for item in self.rows if item["item_type"] == "calibration")
        prospective = next(item for item in self.rows if item["item_type"] == "prospective")
        self.assertEqual(
            protocol.build_paired_protocol(calibration).labels,
            ("CORRECT", "INCORRECT"),
        )
        self.assertEqual(
            protocol.build_paired_protocol(prospective).labels,
            ("PASS", "FAIL"),
        )

    def test_no_prospective_or_knowledge_pre_turns(self):
        for row in runner.selected_rows(self.rows, self.config):
            paired = protocol.build_paired_protocol(row)
            messages = protocol.evaluation_messages(
                paired.factual_prompt, "X", paired.self_evaluation_prompt
            )
            self.assertEqual([message["role"] for message in messages], ["user", "assistant", "user"])
            self.assertNotIn("WILL_PASS", paired.factual_prompt)
            self.assertNotIn("I_KNOW", paired.factual_prompt)

    def test_primary_outputs_are_in_paired_schema(self):
        required = {
            "candidate_score_self", "candidate_rank_self", "candidate_score_other",
            "candidate_rank_other", "self_minus_other_candidate_score",
            "steering_delta_self", "steering_delta_other",
            "self_minus_other_steering_effect",
        }
        self.assertTrue(required.issubset(set(runner.PAIRED_FIELDS)))

    def test_execute_item_generates_once_and_writes_matched_pair(self):
        model, lens, tokenizer = TinyModel(), TinyLens(), TinyTokenizer()
        candidate = fake_candidate(token_id=7, layer=0)
        config = {
            "readout_top_k": 1,
            "rank_policy": {"appearance_rank": "raw_rank", "visibility_gate": False},
            "generation": {"max_choice_tokens": 1, "max_answer_tokens": 1},
            "interventions": {"primary": {
                "candidate_token_id": 7, "layer": 0,
                "strengths": [0.0, -1.7, -1.8],
                "max_injection_fraction": 1.0, "steer_generated": True,
            }},
        }
        item = {
            "item_id": "paired-x", "item_type": "calibration",
            "condition": "self", "domain": "test", "difficulty": "1",
            "detail": "What is one plus one?", "answer_key": ".*",
            "prompt": (
                "What is one plus one?\n"
                "Is your assesement correct or incorrect? "
                "REPLY with only CORRECT or INCORRECT."
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "checkpoints").mkdir()
            recorder = runner.Recorder(run_dir, "paired-test")
            runner.execute_item(
                recorder, model, lens, tokenizer, item, candidate,
                __import__("torch").arange(32), config,
            )
            recorder.close()
            trials = pd.read_csv(run_dir / "trial_summary.csv")
            pairs = pd.read_csv(run_dir / "paired_results.csv")
            interventions = pd.read_csv(run_dir / "intervention_results.csv")
        self.assertEqual(len(trials), 2)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len(interventions), 4)
        self.assertEqual(trials["factual_answer_sha256"].nunique(), 1)
        self.assertEqual(set(trials["condition"]), {"self", "other"})
        self.assertTrue(pairs["same_question_and_answer"].all())
        self.assertTrue((interventions["analysis_family"] == "paired_primary").all())
        self.assertEqual(set(interventions["requested_strength"]), {-1.7, -1.8})
        self.assertTrue(trials["candidate_score"].notna().all())
        self.assertTrue(trials["candidate_rank"].notna().all())

    def test_paired_analysis_creates_all_requested_plots(self):
        rows = []
        for item_id in range(4):
            for strength in (-1.7, -1.8):
                rows.append({
                    "item_id": str(item_id), "item_type": "calibration",
                    "domain": "test", "difficulty": 1,
                    "same_question_and_answer": True,
                    "requested_strength": strength,
                    "candidate_score_self": 10.0 + item_id,
                    "candidate_rank_self": 4 + item_id,
                    "candidate_score_other": 6.0 + item_id,
                    "candidate_rank_other": 20 + item_id,
                    "self_minus_other_candidate_score": 4.0,
                    "steering_delta_self": strength * (item_id + 1),
                    "steering_delta_other": strength * 0.5,
                    "self_minus_other_steering_effect": strength * (item_id + 0.5),
                    "flipped_self": item_id == 0,
                    "flipped_other": False,
                    "flip_effect_self": "worsened" if item_id == 0 else "no_flip",
                    "flip_effect_other": "no_flip",
                })
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "plots").mkdir()
            pd.DataFrame(rows).to_csv(run_dir / "paired_results.csv", index=False)
            (run_dir / "config.json").write_text(
                json.dumps({"analysis": {"bootstrap_samples": 100}}),
                encoding="utf-8",
            )
            status = analysis.analyze(run_dir)
            self.assertTrue(all(status.values()))
            self.assertTrue((run_dir / "plots" / "01_paired_candidate_score.png").is_file())
            self.assertTrue((run_dir / "plots" / "02_paired_candidate_rank.png").is_file())
            self.assertTrue((run_dir / "plots" / "03_paired_steering_effect.png").is_file())
            self.assertTrue((run_dir / "plots" / "04_candidate_vs_steering_difference.png").is_file())
            self.assertTrue((run_dir / "paired_summary.csv").is_file())
            self.assertTrue((run_dir / "results.md").is_file())

    def test_recorder_initializes_reproducibility_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            recorder = runner.Recorder(run_dir, "test-run")
            recorder.close()
            expected = {
                "experiment.log", "events.jsonl", "raw_runs.jsonl",
                "readouts.jsonl", "tokenizations.jsonl", "adaptive_paths.jsonl",
                "errors.jsonl", "trial_summary.csv", "intervention_results.csv",
                "paired_results.csv",
            }
            self.assertTrue(all((run_dir / name).is_file() for name in expected))


if __name__ == "__main__":
    unittest.main()
