from __future__ import annotations

import unittest
import types

import torch

from experiments.process_sensitive_replay.gradient_intervention import (
    GradientBundle,
    InterventionSchedule,
    answer_predictor_positions,
    build_interventions,
    cosine,
    deterministic_seed,
    _make_differentiable_recurrent_cache,
)


class GradientInterventionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = GradientBundle(
            predictor_positions=(4, 5),
            answer_token_ids=(10, 11),
            answer_sequence_logp=-2.0,
            answer_gradients=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            clean_residuals=torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]),
            clean_residual_norms=torch.tensor([5.0, 2.0]),
            process_layer=3,
        )

    def test_autoregressive_predictor_positions(self) -> None:
        self.assertEqual(answer_predictor_positions(5, 3), (4, 5, 6))

    def test_targeted_sign_and_normalized_norm(self) -> None:
        specs = build_interventions(
            self.bundle, family="targeted", strength=0.2,
            campaign_seed=42, item_id="x", max_abs_cosine=0.1,
        )
        self.assertAlmostEqual(cosine(specs[4].delta, self.bundle.answer_gradients[0]), -1.0)
        self.assertAlmostEqual(float(specs[4].delta.norm()), 1.0)
        self.assertAlmostEqual(float(specs[5].delta.norm()), 0.4)

    def test_same_layer_random_controls_are_orthogonal(self) -> None:
        for family in ("random", "alternative_random"):
            specs = build_interventions(
                self.bundle, family=family, strength=0.2,
                campaign_seed=42, item_id="x", max_abs_cosine=0.1,
            )
            for index, position in enumerate((4, 5)):
                self.assertLessEqual(
                    abs(cosine(specs[position].delta, self.bundle.answer_gradients[index])),
                    1e-6,
                )

    def test_alternative_targeted_uses_same_support_reducing_direction(self) -> None:
        specs = build_interventions(
            self.bundle, family="alternative_targeted", strength=0.2,
            campaign_seed=42, item_id="x", max_abs_cosine=0.1,
        )
        for index, position in enumerate((4, 5)):
            self.assertAlmostEqual(
                cosine(specs[position].delta, self.bundle.answer_gradients[index]),
                -1.0,
            )

    def test_seed_and_schedule_are_deterministic_and_fail_closed(self) -> None:
        self.assertEqual(
            deterministic_seed(42, "7", 10, "random"),
            deterministic_seed(42, "7", 10, "random"),
        )
        specs = build_interventions(
            self.bundle, family="random", strength=0.2,
            campaign_seed=42, item_id="x", max_abs_cosine=0.1,
        )
        schedule = InterventionSchedule(process_layer=1, positions=specs)
        self.assertIsNotNone(schedule.delta_for(4))
        with self.assertRaisesRegex(AssertionError, "intervention positions differ"):
            schedule.assert_complete()
        self.assertIsNotNone(schedule.delta_for(5))
        schedule.assert_complete()

    def test_functional_cache_override_does_not_bind_layer_to_itself(self) -> None:
        class Layer:
            pass

        layer = Layer()
        layer.recurrent_states = {}
        layer.is_recurrent_states_initialized = {}
        layer.device = None
        layer.dtype = None
        adapter = types.SimpleNamespace(
            new_cache=lambda: types.SimpleNamespace(layers=[layer])
        )
        cache = _make_differentiable_recurrent_cache(adapter)
        self.assertIsNone(getattr(cache.layers[0].update_recurrent_state, "__self__", None))
        value = torch.ones(2)
        returned = cache.layers[0].update_recurrent_state(value, state_idx=3)
        self.assertIs(returned, value)
        self.assertIs(cache.layers[0].recurrent_states[3], value)
        self.assertTrue(cache.layers[0].is_recurrent_states_initialized[3])


if __name__ == "__main__":
    unittest.main()
