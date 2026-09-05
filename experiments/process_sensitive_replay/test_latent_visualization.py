"""Synthetic I/O test for the standalone latent-geometry plotter."""

from pathlib import Path
import tempfile
import unittest

import torch

from .plot_candidate_anchored_latents import main


class CandidateAnchoredPlotTests(unittest.TestCase):
    def test_native_smoke_layout_generates_auditable_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            direction = torch.zeros(5120, dtype=torch.float32)
            direction[0] = 1
            direction_path = root / "direction.pt"
            torch.save({"direction": direction, "token_id": 75075, "layer": 42,
                        "sha256": "synthetic"}, direction_path)
            for item in ("0", "2"):
                for branch_index, branch in enumerate(("confidence", "correctness")):
                    clean = torch.zeros(5120, dtype=torch.bfloat16)
                    primary = clean.clone()
                    alternative = clean.clone()
                    primary[0] = 0.02 + 0.01 * int(item)
                    primary[1 + branch_index] = 1 + int(item)
                    alternative[0] = 0.03 + 0.01 * int(item)
                    alternative[3 + branch_index] = 1.5 + int(item)
                    torch.save({"item_id": item, "branch": branch,
                                "residuals": {"clean": clean, "primary": primary,
                                              "alternative": alternative}},
                               run / f"residuals_{item}_{branch}.pt")
            output = root / "latent.png"
            result = main(["--run-dir", str(run), "--direction", str(direction_path),
                           "--output", str(output), "--no-pdf", "--dpi", "80"])
            self.assertEqual(result, 0)
            for suffix in (".png", ".csv", ".json"):
                self.assertTrue(output.with_suffix(suffix).is_file())
            self.assertGreater(output.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
