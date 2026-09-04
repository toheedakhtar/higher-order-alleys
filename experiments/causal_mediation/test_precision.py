from __future__ import annotations

import unittest

import torch

from .precision import POLICY, lattice_bound, realize_coordinate, realize_random_pair
from .upstream import load_upstream


class PrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vector, _, _, cls.identity = load_upstream()

    def test_exact_frozen_identity(self):
        self.assertEqual(self.vector.shape, (5120,))
        self.assertEqual(self.identity["smoke_item_ids"], ["0", "2"])

    def test_sham_is_bitwise_noop(self):
        h = torch.randn(5120, generator=torch.Generator().manual_seed(7)).bfloat16()
        q, audit = realize_coordinate(h, self.vector, 0.0)
        self.assertTrue(torch.equal(h.view(torch.int16), q.view(torch.int16)))
        self.assertTrue(audit["precision_gate_passed"])
        self.assertEqual(audit["orthogonal_leakage_l2"], 0)
        self.assertEqual(audit["neighbor_steps"], 0)

    def test_exact_axis_patch_and_input_immutability(self):
        h = torch.tensor([1., 2., -1.], dtype=torch.bfloat16)
        v = torch.tensor([1., 0., 0.])
        before = h.clone()
        q, report = realize_coordinate(h, v, 0.125)
        self.assertTrue(torch.equal(h, before))
        self.assertTrue(torch.equal(q, torch.tensor([1.125, 2., -1.], dtype=torch.bfloat16)))
        self.assertTrue(report["precision_gate_passed"])
        self.assertEqual(report["orthogonal_leakage_l2"], 0)

    def test_compensation_restores_projection_but_rejects_contamination(self):
        h = torch.randn(5120, generator=torch.Generator().manual_seed(42)).bfloat16()
        donor = h.clone()
        donor[0] += 0.125
        delta = float((donor.double() - h.double()).dot(self.vector.double()))
        q, audit = realize_coordinate(h, self.vector, delta)
        self.assertEqual(q.dtype, torch.bfloat16)
        self.assertLess(audit["absolute_coordinate_error"], audit["naive_rounding"]["absolute_coordinate_error"])
        self.assertTrue(audit["lattice_bound"]["certified_infeasible"])
        self.assertFalse(audit["precision_gate_passed"])
        self.assertLessEqual(audit["relative_coordinate_error"], POLICY.coordinate_relative_tolerance)

    def test_lattice_lower_bound_against_enumerated_residuals(self):
        h = torch.tensor([1., 1.], dtype=torch.bfloat16)
        v = torch.tensor([0.6, 0.8], dtype=torch.float64)
        target = 0.001
        bound = lattice_bound(h, v, target)
        self.assertTrue(bound["certified_infeasible"])
        grid = torch.arange(0.99, 1.011, .0001).bfloat16().double().unique()
        for x in grid:
            for y in grid:
                d = torch.stack((x, y)) - h.double()
                error = float(d.dot(v)) - target
                leakage = float((d - d.dot(v) * v).norm())
                self.assertGreaterEqual(error**2 + leakage**2 + 1e-16, bound["nearest_lattice_distance"]**2)

    def test_random_realization_uses_actual_candidate_norm(self):
        h = torch.zeros(5120, dtype=torch.bfloat16)
        _, candidate = realize_coordinate(h, self.vector, 0.01)
        controls = realize_random_pair(h, self.vector, candidate["total_patch_l2"],
            seed=42, item_id="0", branch="confidence", donor="clean")
        self.assertTrue(candidate["precision_gate_passed"])
        for q, audit in controls:
            self.assertTrue(audit["control_precision_gate_passed"])
            self.assertEqual(audit["matched_realized_candidate_l2"], candidate["total_patch_l2"])
            self.assertEqual(audit["policy_sha256"], candidate["policy_sha256"])
            self.assertEqual(q.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(controls[0][0], -controls[1][0]))

    def test_determinism_and_bidirectional_symmetry_at_zero(self):
        h = torch.zeros(5120, dtype=torch.bfloat16)
        p, a = realize_coordinate(h, self.vector, 0.01)
        q, b = realize_coordinate(h, self.vector, 0.01)
        n, _ = realize_coordinate(h, self.vector, -0.01)
        self.assertTrue(torch.equal(p, q))
        self.assertEqual(a, b)
        self.assertTrue(torch.equal(p, -n))

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            realize_coordinate(torch.zeros(5120), self.vector, 0.01)
        with self.assertRaises(ValueError):
            realize_coordinate(torch.zeros(5120).bfloat16(), self.vector, float("nan"))
        with self.assertRaises(ValueError):
            realize_coordinate(torch.zeros(5120).bfloat16(), self.vector * 2, 0.01)


if __name__ == "__main__":
    unittest.main()
