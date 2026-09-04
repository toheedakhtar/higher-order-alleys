"""Synthetic-weight Qwen integration; never experimental behavioral data."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from experiments.process_sensitive_replay.cache_state import audit_cache, release_cache_storage
from experiments.process_sensitive_replay.replay import QwenReplayAdapter
from .precision_smoke import capture_branch


class NativeForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jlens
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)  # Tiny CPU-only synthetic model, not CUDA runtime.
        config = Qwen3_5TextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=64,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=2,
            layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 16,
            rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": .25},
        )
        torch.manual_seed(37)
        model = Qwen3_5ForCausalLM(config).eval().bfloat16()
        cls.lens_model = jlens.from_hf(model, object(), compile=False, force_bos=False)
        cls.adapter = QwenReplayAdapter(model, cls.lens_model)

    @classmethod
    def tearDownClass(cls):
        del cls.adapter, cls.lens_model
        torch.set_num_threads(cls.threads)

    def test_sham_and_native_causal_write(self):
        adapter = self.adapter
        source = adapter.new_cache()
        try:
            adapter.step(1, source, expected_position=0)
            before = audit_cache(source)
            suffix = SimpleNamespace(token_ids=(2, 3), question_position=1)
            h, baseline, unpatched = capture_branch(adapter, source, suffix)
            hs, sham, sham_audit = capture_branch(adapter, source, suffix, replacement=h.clone())
            self.assertTrue(torch.equal(h, hs))
            torch.testing.assert_close(baseline, sham, atol=1e-5, rtol=1e-5)
            self.assertEqual(unpatched["boundary_cache_digest"], sham_audit["boundary_cache_digest"])
            replacement = h.clone()
            replacement[0] += 1
            hp, _, patch = capture_branch(adapter, source, suffix, replacement=replacement)
            self.assertTrue(torch.equal(hp, replacement))
            self.assertEqual(patch["patch_positions"], [1])
            self.assertNotEqual(patch["boundary_cache_digest"], unpatched["boundary_cache_digest"])
            self.assertEqual(patch["cache_tensor_dtypes"], unpatched["cache_tensor_dtypes"])
            self.assertEqual(before.digest, audit_cache(source).digest)
            self.assertEqual(patch["process_hook_calls"], 0)
            self.assertEqual(hp.dtype, torch.bfloat16)
            self.assertTrue(all(p.dtype == torch.bfloat16 for p in adapter.hf_model.parameters()))
            self.assertFalse(adapter.layers[42]._forward_hooks)
        finally:
            release_cache_storage(source)

    def test_float32_write_rejected_and_hook_removed(self):
        source = self.adapter.new_cache()
        try:
            self.adapter.step(1, source, expected_position=0)
            suffix = SimpleNamespace(token_ids=(2,), question_position=1)
            with self.assertRaisesRegex(AssertionError, "native BF16"):
                capture_branch(self.adapter, source, suffix, replacement=torch.zeros(32))
            self.assertFalse(self.adapter.layers[42]._forward_hooks)
        finally:
            release_cache_storage(source)

    def test_pinned_jlens_score_is_not_raw_projection(self):
        from jlens import JacobianLens
        lens = JacobianLens({42: torch.eye(32)}, n_prompts=1, d_model=32)
        model = self.lens_model
        with torch.no_grad():
            # Construct a known numerical relationship using the actual pinned
            # transport/unembed implementation and synthetic weights.
            model._lm_head.weight[0].zero_()
            model._lm_head.weight[0, 0] = 1
            model._final_norm.weight.zero_()
            h = torch.ones(32, dtype=torch.bfloat16)
            q = h.clone()
            q[0] = 2
            before = float(model.unembed(lens.transport(h.float(), 42))[0])
            after = float(model.unembed(lens.transport(q.float(), 42))[0])
            self.assertEqual(float(q[0] - h[0]), 1.0)
            self.assertGreater(after, before)
            self.assertNotAlmostEqual(after - before, 1.0, places=3)
            normalized = model._final_norm(q)
            self.assertAlmostEqual(after, float(normalized[0]), places=5)


if __name__ == "__main__":
    unittest.main()
