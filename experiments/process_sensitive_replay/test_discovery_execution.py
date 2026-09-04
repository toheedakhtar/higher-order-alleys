from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from experiments.process_sensitive_replay.discovery import (
    DISCOVERY_CANDIDATE_CONDITIONS,
    alpha_grid_diagnostics,
    candidate_ranking_row,
    measure_strength_grid_item,
    rank_candidate_grid,
    select_discovery_strengths,
    select_discovery_alpha,
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
            (0.1, 0.3, 0.7, 1.0, 2.2, 3.0, 4.0, 5.0),
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
            for layer in self.config["layers"]["alternative_candidates"]:
                beta_grid[str(layer)] = {}
                for beta in self.config["strengths"]["beta_grid"]:
                    beta = float(beta)
                    drop = 2.2 + abs(beta - 0.11) * 5 + abs(layer - 19) * 0.02
                    beta_grid[str(layer)][str(beta)] = {
                        "support_drop": drop,
                        "positions": [{
                            "position": 4,
                            "perturbation_norm": beta,
                            "direction_cosine": -1.0,
                        }],
                    }
            records.append({
                "item_id": str(item_index),
                "alpha_grid": alpha_grid,
                "beta_grid": beta_grid,
            })

        selected = select_discovery_strengths(records, self.config)

        self.assertEqual(selected["alpha"]["weak_alpha"], 0.1)
        self.assertEqual(selected["alpha"]["strong_alpha"], 0.11)
        self.assertEqual(selected["beta"]["beta"], 0.11)
        self.assertEqual(selected["beta"]["alternative_layer"], 19)
        self.assertTrue(selected["beta"]["diagnostics"]["passed"])

    def test_candidate_ranking_uses_oriented_structured_effects(self) -> None:
        item_count, condition_count, layer_count, vocab_size = 4, 7, 2, 8
        support = torch.tensor([
            [0.0, 0.5, 2.0, 0.2, 2.1, 0.25, 2.0]
            for _ in range(item_count)
        ])
        scores = torch.zeros(item_count, condition_count, layer_count, vocab_size)
        for item in range(item_count):
            baseline = float(item) * 0.1
            scores[item, :, :, :] = baseline
            scores[item, :, 0, 2] = baseline + 2.0 * support[item]
            scores[item, :, 1, 3] = baseline - 1.5 * support[item]
            reset_index = DISCOVERY_CANDIDATE_CONDITIONS.index(
                "targeted_strong_reset"
            )
            scores[item, reset_index, 0, 2] = baseline
            scores[item, reset_index, 1, 3] = baseline

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

    def test_alpha_collision_is_reportable_before_gate_failure(self) -> None:
        drops = (0.1, 0.2, 0.3, 2.5, 5.0, 5.5, 6.0, 7.0)
        records = []
        for item_index in range(16):
            records.append({
                "item_id": str(item_index),
                "alpha_grid": {
                    str(float(alpha)): {
                        "support_drop": drop,
                        "positions": [],
                    }
                    for alpha, drop in zip(
                        self.config["strengths"]["alpha_grid"], drops, strict=True
                    )
                },
            })
        diagnostics = alpha_grid_diagnostics(records, self.config)
        collision = diagnostics["grid"][3]
        self.assertTrue(collision["weak_eligible"])
        self.assertTrue(collision["strong_in_target_range"])
        self.assertEqual(collision["alpha"], 0.1)
        with self.assertRaisesRegex(
            RuntimeError, "selected weak=0.1 strong=0.1"
        ):
            select_discovery_alpha(records, self.config)

    def test_candidate_ranking_fails_closed_without_eligible_direction(self) -> None:
        scores = torch.zeros(4, 7, 1, 8)
        support = torch.tensor([[0.0, 0.5, 2.0, 0.2, 2.0, 0.25, 2.0]] * 4)
        with self.assertRaisesRegex(RuntimeError, "candidate_selection_gate_failed"):
            rank_candidate_grid(
                scores,
                support,
                layers=[36],
                condition_names=DISCOVERY_CANDIDATE_CONDITIONS,
                eligible_token_ids=[2, 3],
                config=self.config,
            )

    def test_beta_only_grid_does_not_recompute_primary_gradient(self) -> None:
        called_layers = []

        def gradient_bundle(*_args, process_layer, **_kwargs):
            called_layers.append(process_layer)
            return SimpleNamespace(
                process_layer=process_layer,
                answer_sequence_logp=-2.0,
                token_logprobs=(-2.0,),
                parity={"layer": process_layer},
            )

        class Outcome(SimpleNamespace):
            def release_cache(self):
                self.cache = None

        def replay(*_args, **_kwargs):
            return Outcome(
                cache=object(),
                cache_audit=SimpleNamespace(
                    digest="digest", layer_digests=("a",) * 64
                ),
                answer_sequence_logp=-2.0,
                token_logprobs=(-2.0,),
                process_hook_positions=(),
                transcript_hash="transcript",
                question_token_hash="question",
                answer_token_hash="answer",
            )

        adapter = SimpleNamespace(
            text_config=SimpleNamespace(layer_types=["full_attention"] * 64)
        )
        answer = {
            "item_id": "x",
            "invalid": False,
            "post_answer_token_ids": [1, 2, 3],
            "question_prefix_token_ids": [1, 2],
            "answer_token_ids": [3],
        }
        with (
            mock.patch(
                "experiments.process_sensitive_replay.discovery.compute_clean_gradients",
                side_effect=gradient_bundle,
            ),
            mock.patch(
                "experiments.process_sensitive_replay.discovery._replay",
                side_effect=replay,
            ),
            mock.patch(
                "experiments.process_sensitive_replay.discovery._schedule",
                return_value=SimpleNamespace(process_layer=0, positions={}),
            ),
            mock.patch(
                "experiments.process_sensitive_replay.discovery.assert_hybrid_cache_integrity"
            ),
            mock.patch(
                "experiments.process_sensitive_replay.discovery.assert_process_propagated"
            ),
        ):
            record = measure_strength_grid_item(
                adapter, answer, self.config, families=("beta",)
            )

        self.assertEqual(
            called_layers,
            [int(value) for value in self.config["layers"]["alternative_candidates"]],
        )
        self.assertIsNone(record["gradient_parity"])
        self.assertEqual(
            len(record["beta_grid"]),
            len(self.config["layers"]["alternative_candidates"]),
        )


if __name__ == "__main__":
    unittest.main()
