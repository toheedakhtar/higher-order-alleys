from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from experiments.process_sensitive_replay.gradient_intervention import (
    compute_clean_gradients,
)
from experiments.process_sensitive_replay.profiles import (
    QUICK_DISCOVERY_ITEM_IDS,
    QUICK_HELDOUT_ITEM_IDS,
    resolve_execution_profile,
)
from experiments.process_sensitive_replay.protocol import (
    allocate_discovery_split,
    load_config,
    load_dataset,
    validate_config,
)
from experiments.process_sensitive_replay.runner import campaign_hashes


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("experiment_config.json")


class ExecutionProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.full = load_config(CONFIG_PATH)
        self.quick = resolve_execution_profile(self.full, "quick")

    def test_quick_profile_is_predeclared_valid_and_hash_distinct(self) -> None:
        rows = load_dataset(
            ROOT / self.quick["dataset"]["path"],
            self.quick["dataset"]["item_types"],
        )
        result = validate_config(self.quick, rows)
        self.assertEqual(result["execution_profile"], "quick")
        self.assertTrue(self.quick["execution_profile"]["exploratory"])
        self.assertEqual(self.quick["strengths"]["alpha_grid"], [0.1, 0.11])
        self.assertEqual(self.quick["strengths"]["beta_grid"], [0.1, 0.2, 0.3])
        self.assertEqual(self.quick["layers"]["readout"], [38, 40, 42])
        self.assertNotEqual(campaign_hashes(self.full)["config"], campaign_hashes(self.quick)["config"])

    def test_quick_profile_preserves_the_causal_and_validity_contract(self) -> None:
        unchanged_sections = (
            "model",
            "lens",
            "dataset",
            "conditions",
            "support_matching",
            "reset_parity",
            "gradient_replay",
            "turn3_replay",
            "meta_branches",
            "generation",
            "interpretation",
            "cuda_memory",
        )
        for section in unchanged_sections:
            with self.subTest(section=section):
                self.assertEqual(self.quick[section], self.full[section])

        self.assertEqual(
            self.quick["alternative"]["objective"],
            "first_32_answer_tokens_sequence_log_probability",
        )
        for key in (
            "mechanism",
            "selection_inputs",
            "prohibited_selection_inputs",
            "max_median_norm_ratio_to_targeted",
            "random_max_abs_cosine_with_answer_gradient",
        ):
            with self.subTest(alternative_contract=key):
                self.assertEqual(
                    self.quick["alternative"][key], self.full["alternative"][key]
                )

        for key in (
            "process",
            "alternative_candidates",
            "minimum_alternative_separation",
            "meta_readout",
            "expected_model_layers",
            "expected_hidden_size",
            "expected_process_layer_type",
            "expected_alternative_layer_type",
        ):
            with self.subTest(layer_contract=key):
                self.assertEqual(self.quick["layers"][key], self.full["layers"][key])

        for key in (
            "weak_min_median_drop_nat",
            "strong_median_drop_range_nat",
        ):
            with self.subTest(strength_gate=key):
                self.assertEqual(
                    self.quick["strengths"][key], self.full["strengths"][key]
                )

        for key in (
            "primary_branch",
            "nontrivial_variance_epsilon",
            "max_support_adjusted_divergence_ratio",
            "dedup_max_abs_direction_cosine",
            "rank_metrics",
        ):
            with self.subTest(candidate_contract=key):
                self.assertEqual(
                    self.quick["candidate_selection"][key],
                    self.full["candidate_selection"][key],
                )

    def test_quick_explicit_split_never_touches_unselected_items(self) -> None:
        selected = [*QUICK_DISCOVERY_ITEM_IDS, *QUICK_HELDOUT_ITEM_IDS]
        rows = [
            {"item_id": item_id, "item_type": "calibration", "invalid": False}
            for item_id in selected
        ]
        split = allocate_discovery_split(rows, self.quick)
        self.assertEqual(split["discovery_item_ids"], list(QUICK_DISCOVERY_ITEM_IDS))
        self.assertEqual(split["heldout_item_ids"], list(QUICK_HELDOUT_ITEM_IDS))

    def test_gradient_window_bounds_the_differentiable_objective(self) -> None:
        adapter = SimpleNamespace(
            hf_model=SimpleNamespace(requires_grad_=lambda _value: None)
        )
        observed = {}

        def recurrent(
            _adapter, _ids, positions, answer_ids, process_layer, **_kwargs
        ):
            observed["positions"] = positions
            observed["answer_ids"] = answer_ids
            return (
                torch.ones(len(answer_ids), 5),
                torch.ones(len(answer_ids), 5),
                -1.0,
                tuple([-1.0 / len(answer_ids)] * len(answer_ids)),
                {"process_layer": process_layer},
            )

        with mock.patch(
            "experiments.process_sensitive_replay.gradient_intervention._recurrent_gradient_pass",
            side_effect=recurrent,
        ):
            bundle = compute_clean_gradients(
                adapter,
                [10, 11, 12, 13, 14, 15, 16],
                prefix_length=2,
                answer_token_ids=[12, 13, 14, 15, 16],
                process_layer=31,
                gradient_answer_token_limit=3,
            )
        self.assertEqual(tuple(observed["answer_ids"]), (12, 13, 14))
        self.assertEqual(bundle.predictor_positions, (1, 2, 3))
        self.assertEqual(bundle.answer_token_ids, (12, 13, 14))


if __name__ == "__main__":
    unittest.main()
