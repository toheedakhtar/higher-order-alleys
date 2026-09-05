"""Synthetic geometry and real Qwen equations, with tiny random CPU weights."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from experiments.process_sensitive_replay.cache_state import audit_cache, release_cache_storage
from .mixed_precision import FP32Tail, candidate_patch, realize_patch, tensor_sha
from .mixed_smoke import continuation, patch_geometry
from . import test_precision_forward as native_tests


class GeometryTests(unittest.TestCase):
    def test_complete_geometry_matrix_reports_controls_and_shams(self):
        rng = torch.Generator().manual_seed(224)
        v = torch.randn(5120, generator=rng)
        v /= v.norm()
        h = torch.randn(5120, generator=rng).bfloat16()
        primary, alternative = h.clone(), h.clone()
        primary[0] += .5
        alternative[1] += .5
        # Readout values here are placeholders, not scientific outcomes.
        with tempfile.TemporaryDirectory() as directory, patch(
                "experiments.causal_mediation.mixed_smoke.candidate_score", return_value=0.0):
            rows = patch_geometry({"clean": h, "primary": primary, "alternative": alternative},
                v, "synthetic", "geometry", None, None, Path(directory))
        self.assertEqual(len(rows), 19)
        self.assertTrue(all(r["precision_gate_passed"] for r in rows))
        self.assertTrue(all(not r["nonzero_judgment_computed"] for r in rows))
        self.assertEqual(sum(r["kind"] == "sham" for r in rows), 3)

    def test_coordinate_noise_and_true_sham(self):
        rng = torch.Generator().manual_seed(123)
        h = torch.randn(5120, generator=rng).bfloat16()
        donor = h.clone()
        donor[0] += .5
        v = torch.randn(5120, generator=rng)
        v /= v.norm()
        q, report = candidate_patch(h, donor, v)
        self.assertTrue(report["precision_gate_passed"], report)
        self.assertLess(report["orthogonal_leakage_over_intended_coordinate"], .001)
        sham, report = candidate_patch(h, h, v)
        self.assertTrue(torch.equal(sham, h.float()))
        self.assertEqual(report["total_patch_l2"], 0)
        self.assertTrue(report["precision_gate_passed"])

    def test_full_restoration_and_random_norm_match(self):
        from .precision import random_direction
        rng = torch.Generator().manual_seed(312)
        h = torch.randn(5120, generator=rng).bfloat16()
        donor = torch.randn(5120, generator=rng).bfloat16()
        v = torch.randn(5120, generator=rng)
        v /= v.norm()
        restored, report = realize_patch(h, donor.double()-h.double(), v)
        self.assertTrue(torch.equal(restored, donor.float()))
        candidate, report = candidate_patch(h, donor, v)
        target = report["total_patch_l2"]
        u, _ = random_direction(v, seed=1729, item_id="synthetic", branch="geometry", donor="synthetic")
        for sign in (-1, 1):
            q, report = realize_patch(h, sign * target * u, v)
            self.assertTrue(report["noise_bound_passed"])
            self.assertLessEqual(abs(report["total_patch_l2"]-target), report["floating_point_noise_l2_bound"])
            self.assertLessEqual(abs(report["realized_candidate_coordinate_change"]), report["floating_point_noise_l2_bound"])


class MixedForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native_tests.NativeForwardTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        native_tests.NativeForwardTests.tearDownClass.__func__(cls)

    def setUp(self):
        self.cache = self.adapter.new_cache()
        self.adapter.step(1, self.cache, expected_position=0)

    def tearDown(self):
        release_cache_storage(self.cache)

    def test_switch_sham_and_downstream_patch(self):
        adapter = self.adapter
        before = audit_cache(self.cache)
        weights = {n: tensor_sha(p) for n, p in adapter.hf_model.named_parameters()}
        math_before = torch.backends.cuda.matmul.fp32_precision
        h, _, bf, _ = continuation(adapter, self.cache, (2, 3), fp32=False)
        observed = []
        with FP32Tail(adapter):
            def spy(_module, args):
                observed.append((args[0].dtype, torch.backends.cuda.matmul.fp32_precision))
            handle = adapter.layers[0].register_forward_pre_hook(spy)
            try:
                hf, _, fp, baseline = continuation(adapter, self.cache, (2, 3), fp32=True)
                hs, _, sham, sham_audit = continuation(adapter, self.cache, (2, 3), fp32=True, sham=True)
                replacement = h.float().clone()
                replacement[0] += 1
                hp, post, changed, patched = continuation(adapter, self.cache, (2, 3), fp32=True, replacement=replacement)
            finally:
                handle.remove()
            self.assertTrue(all(p.dtype == torch.float32 for p in adapter.layers[43].parameters()))
            self.assertTrue(all(p.dtype == torch.bfloat16 for p in adapter.layers[42].parameters()))
        self.assertEqual(observed, [(torch.bfloat16, math_before)] * 6)
        self.assertEqual(weights, {n: tensor_sha(p) for n, p in adapter.hf_model.named_parameters()})
        self.assertTrue(torch.equal(h, hf) and torch.equal(h, hs) and torch.equal(h, hp))
        self.assertTrue(torch.equal(post, replacement))
        torch.testing.assert_close(fp, sham, atol=1e-5, rtol=1e-5)
        self.assertEqual(baseline["state"]["digest"], sham_audit["state"]["digest"])
        self.assertNotEqual(baseline["state"]["digest"], patched["state"]["digest"])
        self.assertFalse(torch.equal(changed, fp))
        self.assertEqual(before.digest, audit_cache(self.cache).digest)
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, math_before)

    def test_exception_restores_weights_math_and_hooks(self):
        adapter = self.adapter
        weights = {n: tensor_sha(p) for n, p in adapter.hf_model.named_parameters()}
        precision = torch.backends.cudnn.conv.fp32_precision
        with self.assertRaisesRegex(AssertionError, "replacement"):
            with FP32Tail(adapter):
                continuation(adapter, self.cache, (2,), fp32=True, replacement=torch.zeros(32, dtype=torch.bfloat16))
        self.assertEqual(weights, {n: tensor_sha(p) for n, p in adapter.hf_model.named_parameters()})
        self.assertEqual(precision, torch.backends.cudnn.conv.fp32_precision)
        for layer in adapter.layers:
            self.assertFalse(layer._forward_hooks)
            self.assertFalse(layer._forward_pre_hooks)

    def test_complete_multitoken_scores_follow_tail(self):
        class Tokenizer:
            eos_token_id = 63
            def __call__(self, label, add_special_tokens=False):
                return {"input_ids": [4, 5] if label == "YES" else [6, 7, 8]}
            def decode(self, ids, **kwargs):
                return "synthetic " + str(ids)
        with FP32Tail(self.adapter):
            _, _, _, result = continuation(self.adapter, self.cache, (2, 3), fp32=True,
                tokenizer=Tokenizer(), labels=("YES", "NO"), max_new_tokens=2)
        scores = result["judgments"]
        self.assertEqual(len(scores["YES"]["relevant_token_logits"]), 2)
        self.assertEqual(len(scores["NO"]["relevant_token_logits"]), 3)
        self.assertAlmostEqual(scores["margin"], sum(scores["YES"]["token_logprobs"])-sum(scores["NO"]["token_logprobs"]))


if __name__ == "__main__":
    unittest.main()
