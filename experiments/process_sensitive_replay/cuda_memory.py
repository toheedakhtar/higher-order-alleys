"""Deterministic CUDA cleanup, measurement, and leak-trend gating."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Mapping

import torch


MIB = 1024 * 1024


def reclaim_cuda_memory() -> None:
    """Collect dead Python objects and return unused CUDA blocks to the driver."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@dataclass
class CudaMemoryTrendGuard:
    """Reject sustained post-cleanup allocation growth between item trials."""

    max_growth_bytes: int
    minimum_step_bytes: int
    required_consecutive_steps: int
    baseline_allocated_bytes: int | None = None
    previous_allocated_bytes: int | None = None
    consecutive_growth_steps: int = 0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CudaMemoryTrendGuard":
        memory = config["cuda_memory"]
        return cls(
            max_growth_bytes=int(memory["max_post_cleanup_growth_mib"]) * MIB,
            minimum_step_bytes=int(memory["minimum_trend_step_mib"]) * MIB,
            required_consecutive_steps=int(memory["consecutive_growth_items"]),
        )

    def begin_item(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA memory guard requires an available CUDA device")
        if self.previous_allocated_bytes is None:
            reclaim_cuda_memory()
        torch.cuda.reset_peak_memory_stats()

    def finish_item(
        self,
        *,
        phase: str,
        stage: str,
        item_id: str,
        operation_error: BaseException | None = None,
    ) -> dict[str, Any]:
        reclaim_cuda_memory()
        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())

        if self.baseline_allocated_bytes is None:
            self.baseline_allocated_bytes = allocated
        else:
            self.baseline_allocated_bytes = min(
                self.baseline_allocated_bytes, allocated
            )
        step_growth = (
            0
            if self.previous_allocated_bytes is None
            else allocated - self.previous_allocated_bytes
        )
        if (
            self.previous_allocated_bytes is not None
            and step_growth >= self.minimum_step_bytes
        ):
            self.consecutive_growth_steps += 1
        else:
            self.consecutive_growth_steps = 0
        total_growth = allocated - self.baseline_allocated_bytes
        trend_failed = (
            total_growth > self.max_growth_bytes
            and self.consecutive_growth_steps >= self.required_consecutive_steps
        )
        self.previous_allocated_bytes = allocated

        return {
            "phase": str(phase),
            "stage": str(stage),
            "item_id": str(item_id),
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "baseline_allocated_bytes": self.baseline_allocated_bytes,
            "step_growth_bytes": step_growth,
            "total_growth_bytes": total_growth,
            "consecutive_growth_steps": self.consecutive_growth_steps,
            "thresholds": {
                "max_post_cleanup_growth_bytes": self.max_growth_bytes,
                "minimum_trend_step_bytes": self.minimum_step_bytes,
                "consecutive_growth_items": self.required_consecutive_steps,
            },
            "operation_error_type": (
                None if operation_error is None else type(operation_error).__name__
            ),
            "operation_error": (
                None if operation_error is None else str(operation_error)
            ),
            "memory_trend_gate_passed": not trend_failed,
        }

    @staticmethod
    def assert_passed(measurement: Mapping[str, Any]) -> None:
        if measurement.get("memory_trend_gate_passed") is not True:
            raise RuntimeError(
                "cuda_memory_trend_gate_failed: post-cleanup allocation grew by "
                f"{int(measurement['total_growth_bytes']) / MIB:.1f} MiB with "
                f"{measurement['consecutive_growth_steps']} consecutive rising items"
            )
