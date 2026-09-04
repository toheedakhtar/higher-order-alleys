"""Answer-support gradients and normalized intervention schedules."""

from __future__ import annotations

import hashlib
import math
import weakref
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from .cache_state import release_cache_storage


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
    clean_residuals: torch.Tensor
    clean_residual_norms: torch.Tensor
    process_layer: int = -1
    token_logprobs: tuple[float, ...] = ()
    parity: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PositionIntervention:
    position: int
    token_id: int
    family: str
    strength: float
    delta: torch.Tensor
    residual_norm: float
    answer_gradient_norm: float
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


def _make_differentiable_recurrent_cache(adapter: Any) -> Any:
    """Keep Qwen's recurrent kernels while replacing autograd-breaking state copies."""
    cache = adapter.new_cache()
    functionalized_layers = 0
    for layer in cache.layers:
        if not hasattr(layer, "recurrent_states"):
            continue
        functionalized_layers += 1

        layer_ref = weakref.ref(layer)

        def functional_recurrent_update(
            recurrent_states: torch.Tensor,
            state_idx: int = 0,
            _layer_ref: Any = layer_ref,
            **_kwargs: Any,
        ) -> torch.Tensor:
            current_layer = _layer_ref()
            if current_layer is None:
                raise RuntimeError("functional recurrent cache layer was released early")
            if current_layer.device is None:
                current_layer.dtype = recurrent_states.dtype
                current_layer.device = recurrent_states.device
            current_layer.recurrent_states[state_idx] = recurrent_states
            current_layer.is_recurrent_states_initialized[state_idx] = True
            return recurrent_states

        # An ordinary function stored on an instance is not descriptor-bound.
        # The weak reference avoids a layer -> bound method -> layer cycle that
        # can retain an entire recurrent autograd graph between items.
        layer.update_recurrent_state = functional_recurrent_update
    if functionalized_layers == 0:
        raise RuntimeError("Qwen hybrid cache exposes no recurrent layers to functionalize")
    return cache


def _clone_convolution_states_for_autograd(cache: Any) -> None:
    """Give each recurrent step fresh conv storage before Qwen updates it in place."""
    for layer in cache.layers:
        states = getattr(layer, "conv_states", None)
        if not isinstance(states, dict):
            continue
        for state_idx, value in tuple(states.items()):
            if torch.is_tensor(value):
                states[state_idx] = value.clone()


def _max_relative_difference(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = torch.maximum(first.abs(), second.abs()).clamp_min(1e-12)
    return float(((first - second).abs() / denominator).max().item())


def _recurrent_gradient_pass(
    adapter: Any,
    token_ids: torch.Tensor,
    predictor_positions: Sequence[int],
    answer_token_ids: Sequence[int],
    process_layer: int,
    *,
    atol: float,
    rtol: float,
) -> tuple[torch.Tensor, torch.Tensor, float, tuple[float, ...], dict[str, Any]]:
    positions = tuple(int(value) for value in predictor_positions)
    targets = tuple(int(value) for value in answer_token_ids)
    if len(positions) != len(targets) or not positions:
        raise ValueError("predictor positions and answer tokens must be non-empty and aligned")
    target_by_position = dict(zip(positions, targets, strict=True))
    functional_cache = _make_differentiable_recurrent_cache(adapter)
    reference_cache = adapter.new_cache()
    roots: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    support_terms: list[torch.Tensor] = []
    functional_logprob_values: list[float] = []
    reference_logprobs: list[float] = []
    hook_positions: list[int] = []
    reference_capture_positions: list[int] = []
    logit_max_abs_diffs: list[float] = []
    logit_max_rel_diffs: list[float] = []
    residual_max_abs_diffs: list[float] = []
    residual_max_rel_diffs: list[float] = []
    sequence = [int(value) for value in token_ids[0].tolist()]

    # Later tokens cannot causally affect any answer logit. Stopping at the
    # final predictor avoids retaining an unnecessary autograd graph.
    for position, token_id in enumerate(sequence[: positions[-1] + 1]):
        intended = position in target_by_position
        captured: dict[str, torch.Tensor] = {}
        handle = None
        if intended:
            _clone_convolution_states_for_autograd(functional_cache)

            def root_hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = hidden_tensor(output)
                residuals.append(hidden[:, -1, :].detach().float().cpu())
                root = hidden.detach().requires_grad_(True)
                roots.append(root)
                hook_positions.append(position)
                captured["root"] = root
                return replace_hidden(output, root)

            handle = adapter.layers[process_layer].register_forward_hook(root_hook)
        try:
            inputs = torch.tensor([[token_id]], dtype=torch.long, device=adapter.input_device)
            context = torch.enable_grad() if intended else torch.no_grad()
            with context:
                outputs = adapter.text_module(
                    input_ids=inputs,
                    past_key_values=functional_cache,
                    use_cache=True,
                )
                functional_logits = _model_logits(
                    adapter, outputs.last_hidden_state[:, -1, :]
                ).float()[0]
        finally:
            if handle is not None:
                handle.remove()
        if adapter.cache_length(functional_cache) != position + 1:
            raise AssertionError(
                f"differentiable recurrent cache failed to advance at position {position}"
            )

        reference_logits, reference_captures = adapter.step(
            token_id,
            reference_cache,
            expected_position=position,
            capture_layers=(process_layer,) if intended else (),
        )
        if not intended:
            continue
        reference_capture_positions.append(position)
        if "root" not in captured:
            raise RuntimeError(f"gradient hook did not fire at intended position {position}")

        observed_logits = functional_logits.detach()
        expected_logits = reference_logits.to(observed_logits.device)
        logit_abs = float((observed_logits - expected_logits).abs().max().item())
        logit_rel = _max_relative_difference(observed_logits, expected_logits)
        logit_max_abs_diffs.append(logit_abs)
        logit_max_rel_diffs.append(logit_rel)
        if not torch.allclose(observed_logits, expected_logits, atol=atol, rtol=rtol):
            raise AssertionError(
                "recurrent gradient per-token logit parity failed at "
                f"position={position} max_abs={logit_abs:.9g} max_rel={logit_rel:.9g}"
            )

        observed_residual = residuals[-1].to(reference_captures[process_layer].device)
        expected_residual = reference_captures[process_layer].float()
        residual_abs = float((observed_residual - expected_residual).abs().max().item())
        residual_rel = _max_relative_difference(observed_residual, expected_residual)
        residual_max_abs_diffs.append(residual_abs)
        residual_max_rel_diffs.append(residual_rel)
        if not torch.allclose(observed_residual, expected_residual, atol=atol, rtol=rtol):
            raise AssertionError(
                "recurrent gradient intervention-layer residual parity failed at "
                f"position={position} max_abs={residual_abs:.9g} max_rel={residual_rel:.9g}"
            )

        target = target_by_position[position]
        functional_log_probs = functional_logits.log_softmax(dim=-1)
        reference_log_probs = reference_logits.log_softmax(dim=-1)
        support_terms.append(functional_log_probs[target])
        functional_logprob_values.append(float(functional_log_probs[target].detach().item()))
        reference_logprobs.append(float(reference_log_probs[target].item()))

    if tuple(hook_positions) != positions:
        raise AssertionError(
            f"gradient hook scope failed: expected {positions}, observed {tuple(hook_positions)}"
        )
    if tuple(reference_capture_positions) != positions:
        raise AssertionError(
            "ordinary cached reference hook scope failed: "
            f"expected {positions}, observed {tuple(reference_capture_positions)}"
        )
    support = torch.stack(support_terms).sum()
    reference_support = float(sum(reference_logprobs))
    support_value = float(sum(functional_logprob_values))
    if not math.isclose(support_value, reference_support, abs_tol=atol, rel_tol=rtol):
        raise AssertionError(
            "recurrent gradient total answer-support parity failed: "
            f"gradient={support_value:.9g} cached={reference_support:.9g} "
            f"abs_diff={abs(support_value - reference_support):.9g}"
        )

    support_gradients = torch.autograd.grad(support, roots, retain_graph=False)
    support_gradient = torch.stack(
        [value[0, -1, :].detach().float().cpu() for value in support_gradients]
    )
    norms = torch.linalg.vector_norm(support_gradient, dim=-1)
    if not torch.isfinite(support_gradient).all() or bool((norms <= 1e-12).any()):
        raise RuntimeError(
            "answer-support gradient is non-finite or zero at an intended position"
        )

    parity = {
        "method": "differentiable_token_by_token_recurrent",
        "reference_method": "ordinary_cached_token_by_token_recurrent",
        "atol": float(atol),
        "rtol": float(rtol),
        "hook_positions": list(hook_positions),
        "ordinary_reference_capture_hook_positions": list(reference_capture_positions),
        "hook_calls_outside_declared_positions": 0,
        "per_token_logit_parity": True,
        "total_answer_support_parity": True,
        "intervention_layer_residual_parity": True,
        "finite_nonzero_answer_gradients": True,
        "reference_answer_sequence_logp": reference_support,
        "gradient_answer_sequence_logp": support_value,
        "support_absolute_difference": abs(support_value - reference_support),
        "per_token_logit_max_abs_differences": logit_max_abs_diffs,
        "per_token_logit_max_rel_differences": logit_max_rel_diffs,
        "per_token_residual_max_abs_differences": residual_max_abs_diffs,
        "per_token_residual_max_rel_differences": residual_max_rel_diffs,
    }
    result = (
        support_gradient,
        torch.cat(residuals, dim=0),
        support_value,
        tuple(reference_logprobs),
        parity,
    )
    # Every returned tensor is already detached onto CPU. Release both GPU
    # caches deterministically after the parity and gradient checks finish.
    release_cache_storage(functional_cache)
    release_cache_storage(reference_cache)
    return result


def compute_clean_gradients(
    adapter: Any,
    post_answer_token_ids: Sequence[int],
    *,
    prefix_length: int,
    answer_token_ids: Sequence[int],
    process_layer: int,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> GradientBundle:
    positions = answer_predictor_positions(prefix_length, len(answer_token_ids))
    ids = torch.tensor([list(post_answer_token_ids)], dtype=torch.long)
    adapter.hf_model.requires_grad_(False)
    (
        answer_gradient,
        residuals,
        support,
        token_logprobs,
        parity,
    ) = _recurrent_gradient_pass(
        adapter,
        ids,
        positions,
        answer_token_ids,
        process_layer,
        atol=float(atol),
        rtol=float(rtol),
    )
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
        clean_residuals=residuals,
        clean_residual_norms=torch.linalg.vector_norm(residuals, dim=-1),
        process_layer=int(process_layer),
        token_logprobs=token_logprobs,
        parity=parity,
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
        used_fallback = False
        if family in {"targeted", "alternative_targeted"}:
            direction = -normalized(gradient)
        elif family in {"random", "alternative_random"}:
            seed = deterministic_seed(campaign_seed, item_id, position, family)
            direction = seeded_orthogonal_random(gradient, seed)
        else:
            raise ValueError(f"unknown intervention family {family!r}")
        direction_cosine = cosine(direction, gradient)
        if (
            family in {"targeted", "alternative_targeted"}
            and abs(direction_cosine + 1.0) > 1e-5
        ):
            raise AssertionError(
                f"{family} direction is not the negative answer-support gradient"
            )
        if family in {"random", "alternative_random"} and abs(direction_cosine) > float(max_abs_cosine) + 1e-6:
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
            "perturbation_norm": float(torch.linalg.vector_norm(spec.delta.float()).item()),
            "direction_cosine": spec.direction_cosine,
            "rng_seed": spec.rng_seed,
            "used_fallback": spec.used_fallback,
        }
        for spec in schedule.values()
    ]
