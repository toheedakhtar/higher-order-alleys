"""Answer-support gradients and normalized intervention schedules."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


def hidden_tensor(output: Any) -> torch.Tensor:
    return output if torch.is_tensor(output) else output[0]


def replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    raise TypeError(f"unsupported decoder-layer output {type(output).__name__}")


def answer_predictor_positions(prefix_length: int, answer_length: int) -> tuple[int, ...]:
    if prefix_length < 1 or answer_length < 1:
        raise ValueError("prefix and answer must both contain tokens")
    return tuple(range(prefix_length - 1, prefix_length + answer_length - 1))


def normalized(vector: torch.Tensor, *, epsilon: float = 1e-12) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector.float())
    if not torch.isfinite(norm) or float(norm.item()) <= epsilon:
        raise RuntimeError("cannot normalize a degenerate direction")
    return vector.float() / norm


def project_orthogonal(vector: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    reference_hat = normalized(reference)
    value = vector.float()
    return value - torch.dot(value, reference_hat) * reference_hat


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.dot(normalized(first), normalized(second)).item())


def deterministic_seed(campaign_seed: int, item_id: str, position: int, family: str) -> int:
    payload = f"{campaign_seed}|{item_id}|{position}|{family}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF


def seeded_orthogonal_random(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    candidate = torch.randn(reference.numel(), generator=generator, dtype=torch.float32)
    candidate = candidate.to(reference.device).reshape_as(reference)
    return normalized(project_orthogonal(candidate, reference))


@dataclass(frozen=True)
class GradientBundle:
    predictor_positions: tuple[int, ...]
    answer_token_ids: tuple[int, ...]
    answer_sequence_logp: float
    answer_gradients: torch.Tensor
    entropy_gradients: torch.Tensor
    clean_residuals: torch.Tensor
    clean_residual_norms: torch.Tensor


@dataclass(frozen=True)
class PositionIntervention:
    position: int
    token_id: int
    family: str
    strength: float
    delta: torch.Tensor
    residual_norm: float
    answer_gradient_norm: float
    alternative_gradient_norm: float | None
    direction_cosine: float
    rng_seed: int | None
    used_fallback: bool


@dataclass
class InterventionSchedule:
    process_layer: int
    positions: dict[int, PositionIntervention]
    enabled: bool = True
    calls: int = 0
    applied_positions: list[int] | None = None

    def __post_init__(self) -> None:
        if self.applied_positions is None:
            self.applied_positions = []

    def delta_for(self, position: int) -> torch.Tensor | None:
        self.calls += 1
        if not self.enabled:
            return None
        spec = self.positions.get(int(position))
        if spec is None:
            return None
        self.applied_positions.append(int(position))
        return spec.delta

    def assert_complete(self) -> None:
        expected = sorted(self.positions)
        observed = sorted(self.applied_positions or [])
        if observed != expected:
            raise AssertionError(f"intervention positions differ: expected {expected}, observed {observed}")


def _model_logits(adapter: Any, final_hidden: torch.Tensor) -> torch.Tensor:
    return adapter.lm_head(final_hidden.to(adapter.lm_head.weight.device))


def _objective_gradient(
    adapter: Any,
    token_ids: torch.Tensor,
    predictor_positions: Sequence[int],
    answer_token_ids: Sequence[int],
    process_layer: int,
    objective: str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    captured: dict[str, torch.Tensor] = {}

    def root_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = hidden_tensor(output)
        if not hidden.requires_grad:
            hidden.requires_grad_(True)
        hidden.retain_grad()
        captured["hidden"] = hidden
        return output

    handle = adapter.layers[process_layer].register_forward_hook(root_hook)
    try:
        outputs = adapter.text_module(
            input_ids=token_ids.to(adapter.input_device),
            use_cache=False,
        )
        final_hidden = outputs.last_hidden_state[:, list(predictor_positions), :]
        logits = _model_logits(adapter, final_hidden).float()
        targets = torch.tensor(answer_token_ids, device=logits.device, dtype=torch.long)
        log_probs = logits.log_softmax(dim=-1)
        support = log_probs[0, torch.arange(len(targets), device=logits.device), targets].sum()
        if objective == "support":
            scalar = support
        elif objective == "entropy":
            probabilities = log_probs.exp()
            scalar = -(probabilities * log_probs).sum(dim=-1).sum()
        else:
            raise ValueError(f"unknown objective {objective!r}")
        root = captured.get("hidden")
        if root is None:
            raise RuntimeError("process-layer gradient hook did not fire")
        gradient = torch.autograd.grad(scalar, root, retain_graph=False)[0]
        selected_gradient = gradient[0, list(predictor_positions), :].detach().float().cpu()
        selected_residual = root[0, list(predictor_positions), :].detach().float().cpu()
        return selected_gradient, selected_residual, float(support.detach().item())
    finally:
        handle.remove()


def compute_clean_gradients(
    adapter: Any,
    post_answer_token_ids: Sequence[int],
    *,
    prefix_length: int,
    answer_token_ids: Sequence[int],
    process_layer: int,
) -> GradientBundle:
    positions = answer_predictor_positions(prefix_length, len(answer_token_ids))
    ids = torch.tensor([list(post_answer_token_ids)], dtype=torch.long)
    answer_gradient, residuals, support = _objective_gradient(
        adapter, ids, positions, answer_token_ids, process_layer, "support"
    )
    entropy_gradient, entropy_residuals, entropy_support = _objective_gradient(
        adapter, ids, positions, answer_token_ids, process_layer, "entropy"
    )
    if not torch.allclose(residuals, entropy_residuals, atol=0, rtol=0):
        raise AssertionError("clean residuals changed between frozen gradient passes")
    if abs(support - entropy_support) > 1e-5:
        raise AssertionError("clean support changed between frozen gradient passes")
    if not torch.isfinite(torch.tensor(support)):
        raise RuntimeError("clean answer support is non-finite")
    gradient_norms = torch.linalg.vector_norm(answer_gradient, dim=-1)
    if not torch.isfinite(gradient_norms).all() or bool((gradient_norms <= 1e-12).any()):
        raise RuntimeError("answer-support gradient is non-finite or degenerate")
    return GradientBundle(
        predictor_positions=positions,
        answer_token_ids=tuple(int(value) for value in answer_token_ids),
        answer_sequence_logp=support,
        answer_gradients=answer_gradient,
        entropy_gradients=entropy_gradient,
        clean_residuals=residuals,
        clean_residual_norms=torch.linalg.vector_norm(residuals, dim=-1),
    )


def build_interventions(
    bundle: GradientBundle,
    *,
    family: str,
    strength: float,
    campaign_seed: int,
    item_id: str,
    max_abs_cosine: float,
) -> dict[int, PositionIntervention]:
    positions: dict[int, PositionIntervention] = {}
    for index, (position, token_id) in enumerate(
        zip(bundle.predictor_positions, bundle.answer_token_ids, strict=True)
    ):
        gradient = bundle.answer_gradients[index]
        residual_norm = float(bundle.clean_residual_norms[index].item())
        if not torch.isfinite(torch.tensor(residual_norm)) or residual_norm <= 0:
            raise RuntimeError("clean residual norm is non-finite or zero")
        seed: int | None = None
        alternative_gradient_norm: float | None = None
        used_fallback = False
        if family == "targeted":
            direction = -normalized(gradient)
        elif family == "random":
            seed = deterministic_seed(campaign_seed, item_id, position, family)
            direction = seeded_orthogonal_random(gradient, seed)
        elif family == "alternative":
            entropy_gradient = bundle.entropy_gradients[index]
            alternative_gradient_norm = float(torch.linalg.vector_norm(entropy_gradient.float()).item())
            if not torch.isfinite(entropy_gradient).all() or not torch.isfinite(torch.tensor(alternative_gradient_norm)):
                raise RuntimeError("alternative entropy gradient is non-finite")
            projected = project_orthogonal(entropy_gradient, gradient)
            try:
                direction = normalized(projected)
            except RuntimeError:
                seed = deterministic_seed(campaign_seed, item_id, position, "alternative_fallback")
                direction = seeded_orthogonal_random(gradient, seed)
                used_fallback = True
        else:
            raise ValueError(f"unknown intervention family {family!r}")
        direction_cosine = cosine(direction, gradient)
        if family != "targeted" and abs(direction_cosine) > float(max_abs_cosine) + 1e-6:
            raise AssertionError(
                f"{family} direction cosine {direction_cosine} exceeds {max_abs_cosine}"
            )
        delta = float(strength) * residual_norm * direction
        expected_norm = abs(float(strength)) * residual_norm
        observed_norm = float(torch.linalg.vector_norm(delta.float()).item())
        if not torch.isfinite(delta).all() or not torch.isfinite(torch.tensor(observed_norm)):
            raise RuntimeError("perturbation contains non-finite values")
        if abs(observed_norm - expected_norm) > max(1e-5, 1e-5 * expected_norm):
            raise AssertionError("normalized perturbation norm contract failed")
        positions[position] = PositionIntervention(
            position=position,
            token_id=token_id,
            family=family,
            strength=float(strength),
            delta=delta.detach().cpu(),
            residual_norm=residual_norm,
            answer_gradient_norm=float(torch.linalg.vector_norm(gradient).item()),
            alternative_gradient_norm=alternative_gradient_norm,
            direction_cosine=direction_cosine,
            rng_seed=seed,
            used_fallback=used_fallback,
        )
    return positions


def intervention_log_rows(
    schedule: Mapping[int, PositionIntervention], *, item_id: str, condition: str,
    process_layer: int,
) -> list[dict[str, Any]]:
    return [
        {
            "item_id": item_id,
            "condition": condition,
            "process_layer": int(process_layer),
            "token_index": spec.position,
            "token_id": spec.token_id,
            "family": spec.family,
            "strength": spec.strength,
            "h_norm": spec.residual_norm,
            "grad_norm": spec.answer_gradient_norm,
            "alternative_gradient_norm": spec.alternative_gradient_norm,
            "perturbation_norm": float(torch.linalg.vector_norm(spec.delta.float()).item()),
            "direction_cosine": spec.direction_cosine,
            "rng_seed": spec.rng_seed,
            "used_fallback": spec.used_fallback,
        }
        for spec in schedule.values()
    ]
