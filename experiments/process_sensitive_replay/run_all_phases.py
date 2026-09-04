"""Run a fresh process-sensitive-replay campaign in fail-closed phase order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .profiles import PROFILE_NAMES
from .runner import DEFAULT_CONFIG, SUPPORTED_PHASES


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Fresh campaign directory; failed or partial campaigns are not resumed.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hf-cache-dir", type=Path)
    parser.add_argument(
        "--profile",
        choices=PROFILE_NAMES,
        default="full",
        help="full confirmatory protocol or reduced exploratory quick profile",
    )
    return parser.parse_args(argv)


def phase_command(args: argparse.Namespace, phase: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.process_sensitive_replay.runner",
        "--phase",
        str(phase),
        "--run-dir",
        str(args.run_dir),
        "--config",
        str(args.config),
        "--profile",
        str(args.profile),
    ]
    if args.hf_cache_dir is not None:
        command.extend(("--hf-cache-dir", str(args.hf_cache_dir)))
    return command


def run_campaign(args: argparse.Namespace) -> int:
    for index, phase in enumerate(SUPPORTED_PHASES, start=1):
        print(
            f"\n[{index}/{len(SUPPORTED_PHASES)}] starting phase={phase}",
            flush=True,
        )
        completed = subprocess.run(phase_command(args, phase), check=False)
        if completed.returncode != 0:
            print(
                f"campaign halted fail-closed at phase={phase} "
                f"exit_code={completed.returncode}; no later phase was started",
                file=sys.stderr,
                flush=True,
            )
            return int(completed.returncode)
        print(f"[{index}/{len(SUPPORTED_PHASES)}] passed phase={phase}", flush=True)
    print(f"\ncampaign completed: {args.run_dir}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_campaign(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
