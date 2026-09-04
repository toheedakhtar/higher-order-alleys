from __future__ import annotations

import argparse
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from experiments.process_sensitive_replay.run_all_phases import (
    phase_command,
    run_campaign,
)
from experiments.process_sensitive_replay.runner import SUPPORTED_PHASES


class RunAllPhasesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = argparse.Namespace(
            run_dir=Path("assets/psr-v9"),
            config=Path("experiments/process_sensitive_replay/experiment_config.json"),
            hf_cache_dir=Path("assets/hf-cache"),
        )

    def test_command_forwards_paths_without_a_shell(self) -> None:
        command = phase_command(self.args, "discovery")
        self.assertIn("experiments.process_sensitive_replay.runner", command)
        self.assertEqual(command[command.index("--phase") + 1], "discovery")
        self.assertEqual(
            Path(command[command.index("--run-dir") + 1]), Path("assets/psr-v9")
        )
        self.assertEqual(
            Path(command[command.index("--hf-cache-dir") + 1]),
            Path("assets/hf-cache"),
        )

    def test_complete_campaign_uses_exact_frozen_phase_order(self) -> None:
        with mock.patch(
            "experiments.process_sensitive_replay.run_all_phases.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            self.assertEqual(run_campaign(self.args), 0)
        observed = [
            call.args[0][call.args[0].index("--phase") + 1]
            for call in run.call_args_list
        ]
        self.assertEqual(observed, list(SUPPORTED_PHASES))
        self.assertTrue(all(call.kwargs == {"check": False} for call in run.call_args_list))

    def test_failure_stops_before_later_phases(self) -> None:
        results = [
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 2),
        ]
        with mock.patch(
            "experiments.process_sensitive_replay.run_all_phases.subprocess.run",
            side_effect=results,
        ) as run:
            self.assertEqual(run_campaign(self.args), 2)
        self.assertEqual(run.call_count, 3)
        last = run.call_args_list[-1].args[0]
        self.assertEqual(
            last[last.index("--phase") + 1],
            SUPPORTED_PHASES[2],
        )


if __name__ == "__main__":
    unittest.main()
