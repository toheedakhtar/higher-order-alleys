from __future__ import annotations

import unittest

import torch

from experiments.process_sensitive_replay.discovery import (
    DISCOVERY_CANDIDATE_CONDITIONS,
    candidate_ranking_row,
    rank_candidate_grid,
    select_discovery_strengths,
)
from experiments.process_sensitive_replay.protocol import load_config


class DiscoveryExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from pathlib import Path

        cls.config = load_config(Path(__file__).with_name("experiment_config.json"))

    def test_complete_discovery_grid_selects_global_strengths(self) -> None:
        alpha_drops = dict(zip(
            [float(value) for value in self.config["strengths"]["alpha_grid"]],
            (0.1, 0.8, 1.5, 2.5, 4.5),
            strict=True,
        ))
        beta_drops = dict(zip(
            [float(value) for value in self.config["strengths"]["beta_grid"]],
            (0.2, 1.0, 2.6, 4.5, 8.0),
            strict=True,
        ))
        records = []
        for item_index in range(16):
            alpha_grid = {}
            beta_grid = {}
            for alpha, drop in alpha_drops.items():
                alpha_grid[str(alpha)] = {
                    "support_drop": drop,
                    "positions": [{"position": 4, "perturbation_norm": alpha}],
                }
            for beta, drop in beta_drops.items():
                beta_grid[str(beta)] = {
                    "support_drop": drop,
                    "positions": [{
                        "position": 4,
                        "perturbation_norm": beta,
                        "direction_cosine": 0.01,
                    }],
                }
            records.append({
                "item_id": str(item_index),
                "alpha_grid": alpha_grid,
                "beta_grid": beta_grid,
            })

        selected = select_discovery_strengths(records, self.config)

        self.assertEqual(selected["alpha"]["weak_alpha"], 0.02)
        self.assertEqual(selected["alpha"]["strong_alpha"], 0.1)
        self.assertEqual(selected["beta"]["beta"], 0.2)
        self.assertTrue(selected["beta"]["diagnostics"]["passed"])

    def test_candidate_ranking_uses_oriented_structured_effects(self) -> None:
        item_count, condition_count, layer_count, vocab_size = 4, 6, 2, 8
        support = torch.tensor([
            [0.0, 0.5, 2.0, 0.2, 2.1, 0.0]
            for _ in range(item_count)
        ])
        scores = torch.zeros(item_count, condition_count, layer_count, vocab_size)
        for item in range(item_count):
            baseline = float(item) * 0.1
            scores[item, :, :, :] = baseline
            scores[item, :, 0, 2] = baseline + 2.0 * support[item]
            scores[item, :, 1, 3] = baseline - 1.5 * support[item]

        ranking = rank_candidate_grid(
            scores,
            support,
            layers=[36, 37],
            condition_names=DISCOVERY_CANDIDATE_CONDITIONS,
            eligible_token_ids=[2, 3, 4],
            config=self.config,
        )

        self.assertEqual(ranking["eligible_count"], 2)
        rows = [candidate_ranking_row(ranking, index) for index in range(2)]
        identities = {(row["layer"], row["token_id"]) for row in rows}
        self.assertEqual(identities, {(36, 2), (37, 3)})
        orientations = {(row["layer"], row["token_id"]): row["orientation"] for row in rows}
        self.assertEqual(orientations[(36, 2)], 1)
        self.assertEqual(orientations[(37, 3)], -1)

    def test_candidate_ranking_fails_closed_without_eligible_direction(self) -> None:
        scores = torch.zeros(4, 6, 1, 8)
        support = torch.tensor([[0.0, 0.5, 2.0, 0.2, 2.0, 0.0]] * 4)
        with self.assertRaisesRegex(RuntimeError, "candidate_selection_gate_failed"):
            rank_candidate_grid(
                scores,
                support,
                layers=[36],
                condition_names=DISCOVERY_CANDIDATE_CONDITIONS,
                eligible_token_ids=[2, 3],
                config=self.config,
            )


if __name__ == "__main__":
    unittest.main()
