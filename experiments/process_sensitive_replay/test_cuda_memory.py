from __future__ import annotations

import unittest
from unittest import mock

from experiments.process_sensitive_replay.cuda_memory import CudaMemoryTrendGuard


class CudaMemoryTrendGuardTests(unittest.TestCase):
    def test_sustained_post_cleanup_growth_fails_closed(self) -> None:
        guard = CudaMemoryTrendGuard(
            max_growth_bytes=100,
            minimum_step_bytes=50,
            required_consecutive_steps=2,
        )
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.synchronize"),
            mock.patch("torch.cuda.empty_cache"),
            mock.patch("torch.cuda.reset_peak_memory_stats"),
            mock.patch("torch.cuda.memory_allocated", side_effect=[1000, 1060, 1120]),
            mock.patch("torch.cuda.memory_reserved", return_value=1200),
            mock.patch("torch.cuda.max_memory_allocated", return_value=1400),
            mock.patch("torch.cuda.max_memory_reserved", return_value=1500),
        ):
            measurements = []
            for item in range(3):
                guard.begin_item()
                measurements.append(guard.finish_item(
                    phase="discovery", stage="beta", item_id=str(item)
                ))
        self.assertTrue(measurements[0]["memory_trend_gate_passed"])
        self.assertTrue(measurements[1]["memory_trend_gate_passed"])
        self.assertFalse(measurements[2]["memory_trend_gate_passed"])
        with self.assertRaisesRegex(RuntimeError, "cuda_memory_trend_gate_failed"):
            guard.assert_passed(measurements[2])

    def test_stable_or_falling_allocations_reset_growth_streak(self) -> None:
        guard = CudaMemoryTrendGuard(
            max_growth_bytes=100,
            minimum_step_bytes=50,
            required_consecutive_steps=2,
        )
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.synchronize"),
            mock.patch("torch.cuda.empty_cache"),
            mock.patch("torch.cuda.reset_peak_memory_stats"),
            mock.patch("torch.cuda.memory_allocated", side_effect=[1000, 1060, 1020]),
            mock.patch("torch.cuda.memory_reserved", return_value=1200),
            mock.patch("torch.cuda.max_memory_allocated", return_value=1400),
            mock.patch("torch.cuda.max_memory_reserved", return_value=1500),
        ):
            final = None
            for item in range(3):
                guard.begin_item()
                final = guard.finish_item(
                    phase="heldout", stage="replay", item_id=str(item)
                )
        self.assertIsNotNone(final)
        self.assertEqual(final["consecutive_growth_steps"], 0)
        self.assertTrue(final["memory_trend_gate_passed"])


if __name__ == "__main__":
    unittest.main()
