from __future__ import annotations

import argparse
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from experiments.process_sensitive_replay.protocol import load_config
from experiments.process_sensitive_replay.runner import (
    PhaseGateFailure,
    campaign_hashes,
    main,
    require_cuda_host,
    run_smoke_phase,
)
from experiments.process_sensitive_replay.smoke import summarize_smoke


CONFIG_PATH = Path(__file__).with_name("experiment_config.json")


def smoke_record(item_id: str, targeted: float, alternative: float) -> dict:
    return {
        "item_id": item_id,
        "support": {"targeted_drop": targeted, "alternative_drop": alternative},
        "checks": {
            "visible_hash_parity": True,
            "teacher_forcing_all_conditions": True,
            "clean_gradient_cache_support_parity": True,
            "gradient_token_logit_parity": True,
            "gradient_total_support_parity": True,
            "gradient_residual_parity": True,
            "gradient_finite_nonzero": True,
            "gradient_hook_scope": True,
            "gradient_sign_finite_difference": True,
            "alternative_gradient_parity": True,
            "intervention_hook_scope": True,
            "hybrid_cache_integrity": True,
            "downstream_state_changed": True,
            "reset_parity": True,
            "branch_isolation": True,
            "turn3_suffix_integrity": True,
            "turn3_process_hook_calls": 0,
            "random_norm_match": True,
            "alternative_random_norm_match": True,
            "alternative_norm_ceiling": True,
        },
    }


class SmokeGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)

    def test_pre_discovery_smoke_does_not_require_frozen_beta_match(self) -> None:
        result = summarize_smoke(
            [smoke_record("1", 2.0, -10.0)], self.config,
            phase="pre_discovery_smoke",
        )
        self.assertTrue(result["passed"])
        self.assertFalse(result["support_matching"]["required"])

    def test_post_freeze_smoke_enforces_support_matching(self) -> None:
        records = [smoke_record(str(index), 2.0, 2.1) for index in range(4)]
        self.assertTrue(summarize_smoke(records, self.config, phase="post_freeze_smoke")["passed"])
        failed = summarize_smoke(
            [smoke_record(str(index), 2.0, 5.0) for index in range(4)],
            self.config,
            phase="post_freeze_smoke",
        )
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["failure_reason"], "support_match_gate_failed")

    def test_failed_post_freeze_smoke_persists_phase_local_report(self) -> None:
        records = [smoke_record(str(index), 2.0, 5.0) for index in range(4)]
        for record in records:
            record["phase"] = "post_freeze_smoke"
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            args = argparse.Namespace(phase="post_freeze_smoke", hf_cache_dir=None)
            split = {"discovery_item_ids": [str(index) for index in range(16)]}
            answers = [{"item_id": str(index)} for index in range(16)]

            def write_trial_stub(directory, _records):
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "trial_summary.csv").write_text("item_id\n", encoding="utf-8")

            def write_candidate_stub(directory, _records):
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "candidate_scores.csv").write_text("item_id\n", encoding="utf-8")

            def run_cuda_stub(_guard, memory_path, _hashes, **kwargs):
                memory_path.parent.mkdir(parents=True, exist_ok=True)
                with memory_path.open("a", encoding="utf-8") as handle:
                    handle.write("{}\n")
                return kwargs["operation"]()

            with (
                patch(
                    "experiments.process_sensitive_replay.runner._load_campaign_inputs",
                    return_value=(answers, split),
                ),
                patch("experiments.process_sensitive_replay.runner._load_pre_discovery_smoke_hash"),
                patch("experiments.process_sensitive_replay.runner._load_discovery_hashes"),
                patch(
                    "experiments.process_sensitive_replay.runner._load_and_validate_frozen_protocol",
                    return_value={
                        "weak_alpha": 0.1,
                        "strong_alpha": 0.11,
                        "beta": 0.2,
                        "alternative_layer": 19,
                    },
                ),
                patch("experiments.process_sensitive_replay.runner.assert_phase_prerequisites"),
                patch(
                    "experiments.process_sensitive_replay.runner.load_runtime",
                    return_value=(None, None, None, None, None, {}),
                ),
                patch(
                    "experiments.process_sensitive_replay.runner.run_smoke_item",
                    side_effect=records,
                ),
                patch(
                    "experiments.process_sensitive_replay.runner.run_cuda_item",
                    side_effect=run_cuda_stub,
                ),
                patch("experiments.process_sensitive_replay.runner._log_smoke_record"),
                patch(
                    "experiments.process_sensitive_replay.runner._write_trial_summary",
                    side_effect=write_trial_stub,
                ),
                patch(
                    "experiments.process_sensitive_replay.runner._write_candidate_scores",
                    side_effect=write_candidate_stub,
                ),
            ):
                with self.assertRaises(PhaseGateFailure) as raised:
                    run_smoke_phase(
                        args,
                        run_dir,
                        self.config,
                        campaign_hashes(CONFIG_PATH, self.config),
                    )
            report_path = run_dir / "post_freeze_smoke" / "smoke_report.json"
            self.assertTrue(report_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertEqual(len(report["item_support_matching"]), 4)
            self.assertTrue(
                (run_dir / "post_freeze_smoke" / "trials.jsonl").is_file()
            )
            self.assertEqual(raised.exception.measurements, report)

    def test_any_critical_assertion_blocks_smoke(self) -> None:
        record = smoke_record("1", 2.0, 2.0)
        record["checks"]["reset_parity"] = False
        with self.assertRaisesRegex(AssertionError, "critical smoke assertion"):
            summarize_smoke([record], self.config, phase="pre_discovery_smoke")

    def test_missing_turn3_suffix_integrity_blocks_smoke(self) -> None:
        record = smoke_record("1", 2.0, 2.0)
        del record["checks"]["turn3_suffix_integrity"]
        with self.assertRaisesRegex(AssertionError, "critical smoke assertion"):
            summarize_smoke([record], self.config, phase="pre_discovery_smoke")

    def test_cpu_host_refuses_real_model_phase(self) -> None:
        # The checked-in environment is CPU-only; if this test is later run on
        # the CUDA host, the guard correctly becomes a no-op.
        import torch

        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CPU-only PyTorch"):
                require_cuda_host(self.config)

    def test_cpu_runner_records_failure_without_success_marker(self) -> None:
        import torch

        if torch.cuda.is_available():
            self.skipTest("CPU fail-closed behavior is specific to this development host")
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            self.assertEqual(main(["--phase", "validate", "--run-dir", str(run_dir)]), 0)
            self.assertEqual(main(["--phase", "validate", "--run-dir", str(run_dir)]), 2)
            validate_gate = json.loads(
                (run_dir / "validate" / "gate_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(validate_gate["status"], "passed")
            self.assertEqual(main(["--phase", "answer_bank", "--run-dir", str(run_dir)]), 2)
            gate = json.loads(
                (run_dir / "answer_bank" / "gate_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(gate["status"], "failed")
            self.assertFalse((run_dir / "answer_bank" / "phase_success.json").exists())


if __name__ == "__main__":
    unittest.main()
