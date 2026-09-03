from __future__ import annotations

import unittest
import tempfile
import json
from pathlib import Path

from experiments.process_sensitive_replay.protocol import load_config
from experiments.process_sensitive_replay.runner import main, require_cuda_host
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
            "hybrid_cache_integrity": True,
            "downstream_state_changed": True,
            "reset_parity": True,
            "branch_isolation": True,
            "turn3_process_hook_calls": 0,
            "random_norm_match": True,
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
        with self.assertRaisesRegex(RuntimeError, "support_match_gate_failed"):
            summarize_smoke(
                [smoke_record(str(index), 2.0, 5.0) for index in range(4)],
                self.config,
                phase="post_freeze_smoke",
            )

    def test_any_critical_assertion_blocks_smoke(self) -> None:
        record = smoke_record("1", 2.0, 2.0)
        record["checks"]["reset_parity"] = False
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
