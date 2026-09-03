from __future__ import annotations

import unittest

import torch

from experiments.process_sensitive_replay.cache_state import (
    assert_hybrid_cache_integrity,
    assert_process_propagated,
)
from experiments.process_sensitive_replay.gradient_intervention import (
    InterventionSchedule,
    build_interventions,
    compute_clean_gradients,
)
from experiments.process_sensitive_replay.replay import (
    QwenReplayAdapter,
    replay_teacher_forced,
)


class ActualQwenReplayIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import jlens
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        config = Qwen3_5TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            layer_types=[
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
            ],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
            },
        )
        torch.manual_seed(1)
        model = Qwen3_5ForCausalLM(config).eval()
        lens_model = jlens.from_hf(model, object(), compile=False, force_bos=False)
        cls.adapter = QwenReplayAdapter(model, lens_model)

    def test_gradient_targeted_replay_and_hybrid_state_propagation(self) -> None:
        post_answer = [1, 2, 3, 4, 5]
        question_prefix = [1, 2, 3]
        answer = [4]
        bundle = compute_clean_gradients(
            self.adapter,
            post_answer,
            prefix_length=len(question_prefix),
            answer_token_ids=answer,
            process_layer=3,
        )
        clean = replay_teacher_forced(
            self.adapter,
            post_answer_token_ids=post_answer,
            question_prefix_token_ids=question_prefix,
            answer_token_ids=answer,
        )
        schedule = InterventionSchedule(
            process_layer=3,
            positions=build_interventions(
                bundle,
                family="targeted",
                strength=0.05,
                campaign_seed=42,
                item_id="tiny",
                max_abs_cosine=0.1,
            ),
        )
        targeted = replay_teacher_forced(
            self.adapter,
            post_answer_token_ids=post_answer,
            question_prefix_token_ids=question_prefix,
            answer_token_ids=answer,
            intervention=schedule,
        )
        self.assertAlmostEqual(clean.answer_sequence_logp, bundle.answer_sequence_logp, places=5)
        self.assertGreater(clean.answer_sequence_logp - targeted.answer_sequence_logp, 0)
        layer_types = self.adapter.text_config.layer_types
        assert_hybrid_cache_integrity(
            clean.cache, layer_types=layer_types,
            expected_sequence_length=len(post_answer),
        )
        assert_hybrid_cache_integrity(
            targeted.cache, layer_types=layer_types,
            expected_sequence_length=len(post_answer),
        )
        assert_process_propagated(
            clean.cache_audit, targeted.cache_audit, process_layer=3
        )


if __name__ == "__main__":
    unittest.main()

