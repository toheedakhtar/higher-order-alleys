"""Outcome-blind, CPU/float64 geometry; BF16 input and output only.

No model, labels, confidence margins, or correctness outcomes enter this API.
The solver is deterministic and heuristic, not a global integer optimizer.
An independent lattice-distance bound can certify infeasibility.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class PrecisionPolicy:
    version: str = "bf16_compensated_v1_proposed"
    # Proposed, geometry-based budgets; never adapted per item or outcome.
    coordinate_relative_tolerance: float = 0.01
    orthogonal_relative_tolerance: float = 0.10
    random_norm_relative_tolerance: float = 0.01
    random_candidate_projection_relative_tolerance: float = 0.01
    bracket_steps: int = 64
    bisection_steps: int = 80
    correction_steps: int = 512
    refinement_steps: int = 512

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


POLICY = PrecisionPolicy()


def vector_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def _vector(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().cpu().double().reshape(-1)
    if not value.numel() or not bool(torch.isfinite(value).all()):
        raise ValueError("vectors must be nonempty and finite")
    return value


def _nearest(value: torch.Tensor) -> torch.Tensor:
    """Nearest finite BF16 neighbor, checked in float64 (including ties)."""
    middle = value.to(torch.bfloat16)
    minus = torch.nextafter(middle, torch.full_like(middle, -math.inf))
    plus = torch.nextafter(middle, torch.full_like(middle, math.inf))
    choices = torch.stack((middle.double(), minus.double(), plus.double()))
    errors = (choices - value).abs()
    errors[~torch.isfinite(choices)] = math.inf
    selected = errors.argmin(dim=0)
    rounded = choices.gather(0, selected.unsqueeze(0))[0]
    if not bool(torch.isfinite(rounded).all()):
        raise ValueError("no finite BF16 representation")
    return rounded


def _geometry(h: torch.Tensor, q: torch.Tensor, w: torch.Tensor, target: float) -> dict:
    d = q - h
    w2 = float(w.dot(w))
    realized = float(d.dot(w))
    leakage = float((d - realized / w2 * w).norm())
    norm = float(d.norm())
    error = abs(realized - target)
    return {
        "intended_coordinate_change": target,
        "realized_coordinate_change": realized,
        "absolute_coordinate_error": error,
        "relative_coordinate_error": error / abs(target) if target else (0.0 if error == 0 else None),
        "total_patch_l2": norm,
        "orthogonal_leakage_l2": leakage,
        "orthogonal_leakage_to_intended_coordinate": leakage / abs(target) if target else (0.0 if leakage == 0 else None),
        "ideal_patch_l2": abs(target) / math.sqrt(w2),
        "norm_error": abs(norm - abs(target) / math.sqrt(w2)),
        "resulting_residual_dtype": "torch.bfloat16",
    }


def lattice_bound(h: torch.Tensor, w: torch.Tensor, target: float, policy=POLICY) -> dict:
    """Certificate valid for EVERY BF16 residual, not only this solver.

    With d=q-h, e=w.d-target and d_perp perpendicular to w:
      ||q-(h+target*w/||w||²)||² = e²/||w||² + ||d_perp||².
    Coordinatewise nearest BF16 rounding supplies a global lower bound on the
    left side. If it exceeds both budgets combined, no BF16 patch can pass.
    """
    h, w = _vector(h), _vector(w)
    w2 = float(w.dot(w))
    if h.shape != w.shape or w2 <= 0 or not math.isfinite(target):
        raise ValueError("invalid bound inputs")
    ideal = h + target / w2 * w
    lower = float((_nearest(ideal) - ideal).norm())
    error_budget = policy.coordinate_relative_tolerance * abs(target)
    leakage_budget = policy.orthogonal_relative_tolerance * abs(target)
    combined = math.hypot(error_budget / math.sqrt(w2), leakage_budget)
    return {
        "nearest_lattice_distance": lower,
        "coordinate_error_budget": error_budget,
        "orthogonal_leakage_budget": leakage_budget,
        "combined_budget": combined,
        # Numerical margin protects the certificate from float64 roundoff.
        "certified_infeasible": lower > combined + 1e-12 * max(1.0, float(h.norm())),
        "minimum_orthogonal_leakage_at_coordinate_budget": math.sqrt(max(0.0, lower**2 - error_budget**2 / w2)),
    }


def _neighbors(q: torch.Tensor) -> torch.Tensor:
    b = q.to(torch.bfloat16)
    return torch.stack((
        torch.nextafter(b, torch.full_like(b, -math.inf)).double(),
        torch.nextafter(b, torch.full_like(b, math.inf)).double(),
    ))


def _correct(h, q, w, target, policy):
    """Single-neighbor compensation, then leakage descent within error budget."""
    q = q.clone()
    budget = abs(target) * policy.coordinate_relative_tolerance
    count = 0
    for _ in range(policy.correction_steps):
        error = target - float((q - h).dot(w))
        if abs(error) <= budget:
            break
        moves = _neighbors(q) - q
        projection = moves * w
        new_error = (error - projection).abs()
        allowed = torch.isfinite(moves) & (new_error < abs(error))
        if not bool(allowed.any()):
            break
        energy = 2 * (q - h) * moves + moves.square()
        reaching = allowed & (new_error <= budget)
        if bool(reaching.any()):
            costs = torch.where(reaching, energy, math.inf)
        else:
            # Prefer efficient non-overshooting changes; exact ties resolve by
            # flattened order: negative neighbor first, then ascending index.
            toward = allowed & (projection.abs() <= abs(error))
            eligible = toward if bool(toward.any()) else allowed
            costs = torch.where(eligible, energy / projection.abs().clamp_min(1e-300), math.inf)
        flat = int(costs.reshape(-1).argmin())
        side, index = divmod(flat, q.numel())
        q[index] += moves[side, index]
        count += 1
    for _ in range(policy.refinement_steps):
        realized = float((q - h).dot(w))
        moves = _neighbors(q) - q
        energy = 2 * (q - h) * moves + moves.square()
        dp = moves * w
        leakage_change = energy - (2 * realized * dp + dp.square()) / float(w.dot(w))
        allowed = (torch.isfinite(moves) & (leakage_change < -1e-30)
                   & ((realized + moves * w - target).abs() <= budget))
        if not bool(allowed.any()):
            break
        costs = torch.where(allowed, leakage_change, math.inf)
        side, index = divmod(int(costs.reshape(-1).argmin()), q.numel())
        q[index] += moves[side, index]
        count += 1
    return q, count


@torch.no_grad()
def realize_coordinate(recipient: torch.Tensor, direction: torch.Tensor, target: float,
                       *, policy=POLICY) -> tuple[torch.Tensor, dict]:
    """Return a PROPOSAL and diagnostics. A false gate MUST NOT be applied.

    Float64 is used solely for CPU patch design/auditing, never model compute.
    The original vector is not renormalized or replaced.
    """
    if recipient.dtype != torch.bfloat16:
        raise ValueError("recipient must be native BF16")
    h, w = _vector(recipient), _vector(direction)
    if h.shape != w.shape or abs(float(w.norm()) - 1.0) > 1e-5:
        raise ValueError("direction must match residual width and be unit norm")
    if not math.isfinite(target):
        raise ValueError("target must be finite")
    # Canonical reflection removes sign-dependent neighbor tie-breaking.
    # This is arithmetic symmetry, never selection based on outcomes.
    if target < 0:
        reflected, metrics = realize_coordinate(-recipient, direction, -target, policy=policy)
        result = -reflected
        metrics.update(_geometry(h, _vector(result), w, target))
        metrics["naive_rounding"] = _geometry(
            h, _nearest(h + target / float(w.dot(w)) * w), w, target)
        return result, metrics
    bound = lattice_bound(h, w, target, policy)
    naive = _nearest(h + target / float(w.dot(w)) * w)
    if target == 0:
        q, steps = h.clone(), 0
    else:
        sign = 1 if target > 0 else -1
        goal = abs(target)
        oriented = w * sign
        low, high = 0.0, goal / float(w.dot(w))
        for _ in range(policy.bracket_steps):
            if float((_nearest(h + high * oriented) - h).dot(oriented)) >= goal:
                break
            high *= 2
        else:
            raise RuntimeError("BF16 projection bracket exhausted")
        for _ in range(policy.bisection_steps):
            mid = (low + high) / 2
            if float((_nearest(h + mid * oriented) - h).dot(oriented)) < goal:
                low = mid
            else:
                high = mid
        # Norm-minimizing Lagrangian rounding + discrete compensation from both
        # sides of its discontinuity and from naive rounding. No outcome input.
        proposals = [_correct(h, start, w, target, policy) for start in (
            naive, _nearest(h + low * oriented), _nearest(h + high * oriented))]

        def key(pair):
            a = _geometry(h, pair[0], w, target)
            ok = a["absolute_coordinate_error"] <= bound["coordinate_error_budget"]
            # Coordinate-feasible proposals: minimize orthogonal leakage.
            return (not ok, a["orthogonal_leakage_l2"] if ok else a["absolute_coordinate_error"],
                    a["absolute_coordinate_error"], a["total_patch_l2"])

        q, steps = min(proposals, key=key)
    metrics = _geometry(h, q, w, target)
    passed = (metrics["absolute_coordinate_error"] <= bound["coordinate_error_budget"]
              and metrics["orthogonal_leakage_l2"] <= bound["orthogonal_leakage_budget"])
    metrics.update({
        "policy_sha256": policy.digest(), "policy": asdict(policy),
        "coordinate_direction_sha256": vector_hash(direction),
        "neighbor_steps": steps, "lattice_bound": bound,
        "naive_rounding": _geometry(h, naive, w, target),
        "precision_gate_passed": bool(passed),
        "status": "passed" if passed else ("certified_infeasible" if bound["certified_infeasible"] else "solver_did_not_qualify"),
        "behavioral_outcomes_used": False,
    })
    return q.reshape(recipient.shape).to(device=recipient.device, dtype=torch.bfloat16), metrics


def random_direction(candidate: torch.Tensor, *, seed: int, item_id: str,
                     branch: str, donor: str) -> tuple[torch.Tensor, int]:
    encoded = json.dumps([seed, item_id, branch, donor], separators=(",", ":")).encode()
    derived = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63 - 1)
    v = _vector(candidate)
    u = torch.randn(v.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(derived))
    u -= u.dot(v) / v.dot(v) * v
    if float(u.norm()) == 0:
        raise ValueError("no orthogonal random direction in this residual space")
    return u / u.norm(), derived


def realize_random_pair(recipient, candidate, realized_candidate_norm, *, seed, item_id,
                        branch, donor, policy=POLICY):
    """Same solver and accuracy budgets; match to REALIZED candidate L2.

    A control additionally must match total norm and have little projection
    onto the original candidate. Never silently rescale after realization.
    """
    if not math.isfinite(realized_candidate_norm) or realized_candidate_norm < 0:
        raise ValueError("invalid realized candidate norm")
    u, derived = random_direction(candidate, seed=seed, item_id=item_id, branch=branch, donor=donor)
    v, h = _vector(candidate), _vector(recipient)
    results = []
    for sign in (1, -1):
        q, report = realize_coordinate(recipient, u, sign * realized_candidate_norm, policy=policy)
        d = _vector(q) - h
        projection = float(d.dot(v))
        norm_error = abs(float(d.norm()) - realized_candidate_norm)
        scale = realized_candidate_norm
        report.update({
            "random_seed": derived,
            "random_vector_sha256": hashlib.sha256(u.contiguous().numpy().tobytes()).hexdigest(),
            "random_vector_hash_dtype": "float64", "random_sign": sign,
            "random_vector_candidate_cosine": float(u.dot(v) / v.norm()),
            "matched_realized_candidate_l2": scale,
            "realized_norm_match_absolute_error": norm_error,
            "realized_norm_match_relative_error": norm_error / scale if scale else 0.0,
            "realized_candidate_projection": projection,
            "realized_candidate_projection_relative_to_target_norm": abs(projection) / scale if scale else 0.0,
            "control_precision_gate_passed": bool(
                report["precision_gate_passed"]
                and norm_error <= policy.random_norm_relative_tolerance * scale
                and abs(projection) <= policy.random_candidate_projection_relative_tolerance * scale),
        })
        results.append((q, report))
    return results
