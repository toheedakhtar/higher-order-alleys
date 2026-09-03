"""J-Lens directions, residual hooks, readouts, generation, and label scoring."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


def sha256_tensor(tensor: torch.Tensor) -> str:
    data = tensor.detach().float().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _hidden(output: Any) -> torch.Tensor:
    return output if torch.is_tensor(output) else output[0]


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    raise TypeError(f"unsupported block output: {type(output).__name__}")


@dataclass(frozen=True)
class SteeringSpec:
    mode: str
    layer: int
    direction: torch.Tensor
    requested_strength: float
    direction_source_position: int
    prompt_length: int
    selected_position: int | None = None
    localized_residual_scale: float | None = None
    max_injection_fraction: float | None = None
    steer_generated: bool = False
    skip_positions: tuple[int, ...] = ()

    @property
    def intervention_scope(self) -> str:
        if self.mode == "neuronpedia_global":
            return (
                "all_prompt_positions_and_generated_tokens"
                if self.steer_generated
                else "all_prompt_positions"
            )
        if self.mode == "single_position":
            return "one_selected_prompt_position"
        raise ValueError(f"unsupported intervention mode: {self.mode!r}")

    @property
    def effective_strength_after_cap(self) -> float:
        if self.mode != "neuronpedia_global" or self.max_injection_fraction is None:
            return float(self.requested_strength)
        cap = float(self.max_injection_fraction)
        return max(-cap, min(cap, float(self.requested_strength)))

    @property
    def applied_prompt_position_count(self) -> int:
        if self.mode == "single_position":
            return 1
        skipped = sum(1 for position in set(self.skip_positions) if 0 <= position < self.prompt_length)
        return self.prompt_length - skipped


@dataclass
class HookAudit:
    calls: int = 0
    sequence_lengths: list[int] = field(default_factory=list)
    changed_positions_per_call: list[int] = field(default_factory=list)

    def record(self, sequence_length: int, changed: int) -> None:
        self.calls += 1
        self.sequence_lengths.append(int(sequence_length))
        self.changed_positions_per_call.append(int(changed))


class ResidualHooks:
    """Capture block outputs and apply one centralized intervention contract."""

    def __init__(
        self,
        layers: Sequence[Any],
        *,
        capture_layers: Iterable[int],
        capture_position: int,
        capture_scale_layers: Iterable[int] = (),
        steering: SteeringSpec | None = None,
    ) -> None:
        self.layers = layers
        self.capture_layers = sorted(set(int(layer) for layer in capture_layers))
        self.capture_position = int(capture_position)
        self.capture_scale_layers = set(int(layer) for layer in capture_scale_layers)
        self.steering = steering
        self.captured: dict[int, torch.Tensor] = {}
        self.mean_residual_scales: dict[int, float] = {}
        self.audit = HookAudit()
        self.handles: list[Any] = []

    def _apply(self, hidden: torch.Tensor) -> tuple[torch.Tensor, int]:
        spec = self.steering
        if spec is None:
            return hidden, 0
        direction = spec.direction.to(device=hidden.device, dtype=hidden.dtype)
        direction = direction / direction.float().norm().clamp_min(1e-12).to(direction.dtype)
        changed = hidden.clone()
        if spec.mode == "single_position":
            if spec.selected_position is None or spec.localized_residual_scale is None:
                raise ValueError("single_position requires position and residual scale")
            if not 0 <= spec.selected_position < hidden.shape[1]:
                raise IndexError("localized intervention position outside sequence")
            delta = (
                direction
                * float(spec.localized_residual_scale)
                * float(spec.requested_strength)
            )
            changed[:, spec.selected_position, :] += delta
            return changed, 1
        if spec.mode != "neuronpedia_global":
            raise ValueError(f"unsupported intervention mode: {spec.mode!r}")
        stop = hidden.shape[1] if spec.steer_generated else min(hidden.shape[1], spec.prompt_length)
        target = hidden[:, :stop, :]
        norms = torch.linalg.vector_norm(target.float(), dim=-1, keepdim=True).to(target.dtype)
        delta = float(spec.requested_strength) * norms * direction
        if spec.max_injection_fraction is not None:
            injected_norms = torch.linalg.vector_norm(
                delta.float(), dim=-1, keepdim=True
            ).to(delta.dtype)
            maximum_norms = float(spec.max_injection_fraction) * norms
            factors = torch.where(
                injected_norms > maximum_norms,
                maximum_norms / injected_norms.clamp_min(1e-12),
                torch.ones_like(injected_norms),
            )
            delta = delta * factors
        for position in spec.skip_positions:
            if 0 <= position < stop:
                delta[:, position, :] = 0
        changed[:, :stop, :] = target + delta
        applied = stop - sum(1 for position in set(spec.skip_positions) if 0 <= position < stop)
        return changed, applied

    def _hook(self, layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = _hidden(output)
            modified = False
            if self.steering is not None and layer == self.steering.layer:
                hidden, count = self._apply(hidden)
                self.audit.record(hidden.shape[1], count)
                output = _replace_hidden(output, hidden)
                modified = True
            if layer in self.capture_layers:
                if not 0 <= self.capture_position < hidden.shape[1]:
                    raise IndexError("capture position outside sequence")
                self.captured[layer] = hidden[:, self.capture_position, :].detach()
            if layer in self.capture_scale_layers:
                self.mean_residual_scales[layer] = float(
                    hidden.detach()[0].float().norm(dim=-1).mean().item()
                )
            return output if modified else None

        return hook

    def __enter__(self) -> "ResidualHooks":
        indices = set(self.capture_layers) | self.capture_scale_layers
        if self.steering is not None:
            indices.add(self.steering.layer)
        for layer in sorted(indices):
            self.handles.append(self.layers[layer].register_forward_hook(self._hook(layer)))
        return self

    def __exit__(self, *_exc: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@dataclass
class Candidate:
    label: str
    token_id: int
    feature_id: str
    layer: int
    direction: torch.Tensor
    direction_sha256: str
    direction_path: str
    direction_method: str = "normalize(J_l.T @ lm_head.weight[token_id])"

    def metadata(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("direction")
        return result


def candidate_direction(
    *,
    label: str,
    token_id: int,
    layer: int,
    lens_model: Any,
    lens: Any,
    checkpoint_dir: Path,
) -> Candidate:
    if layer not in lens.jacobians:
        raise ValueError(f"layer {layer} is absent from fitted lens")
    matrix = lens.jacobians[layer]
    unembedding = lens_model._lm_head.weight[token_id].detach().to(
        device=matrix.device, dtype=torch.float32
    )
    direction = (matrix.float().T @ unembedding).cpu()
    norm = direction.norm()
    if not torch.isfinite(norm) or norm.item() == 0:
        raise RuntimeError("candidate direction has invalid norm")
    direction /= norm
    digest = sha256_tensor(direction)
    target = checkpoint_dir / "directions"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"layer{layer}_token{token_id}_{digest[:12]}.pt"
    if not path.exists():
        torch.save(
            {
                "direction": direction,
                "label": label,
                "token_id": token_id,
                "layer": layer,
                "method": "normalize(J_l.T @ lm_head.weight[token_id])",
                "sha256": digest,
            },
            path,
        )
    return Candidate(
        label=label,
        token_id=token_id,
        feature_id=f"vocab_token:{token_id}",
        layer=layer,
        direction=direction,
        direction_sha256=digest,
        direction_path=str(path.resolve()),
    )


def resolve_candidate(
    spec: dict[str, Any],
    layer: int,
    tokenizer: Any,
    lens_model: Any,
    lens: Any,
    checkpoint_dir: Path,
) -> Candidate:
    encoded = tokenizer(spec["label"], add_special_tokens=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    if len(encoded) != 1 or int(encoded[0]) != int(spec["token_id"]):
        raise ValueError(
            f"candidate {spec['label']!r} expected token {spec['token_id']}, got {encoded}"
        )
    return candidate_direction(
        label=spec["label"],
        token_id=int(spec["token_id"]),
        layer=int(layer),
        lens_model=lens_model,
        lens=lens,
        checkpoint_dir=checkpoint_dir,
    )


def token_is_wordlike(text: str) -> bool:
    """Match jlens.vis._meaningful_token_mask for UI-filtered ranking."""
    stripped = text.strip()
    if not stripped:
        return False
    if "<|" in stripped or (stripped.startswith("<") and stripped.endswith(">")):
        return False
    return all(
        char.isalnum()
        or (0 < position < len(stripped) - 1 and char in "'-’")
        for position, char in enumerate(stripped)
    )


def word_token_ids(tokenizer: Any) -> torch.Tensor:
    ids = [
        token_id
        for token_id in range(len(tokenizer))
        if token_is_wordlike(
            tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        )
    ]
    if not ids:
        raise RuntimeError("word-token filter selected no vocabulary entries")
    return torch.tensor(ids, dtype=torch.long)


def _vocab_logits(logits: torch.Tensor, context: str) -> torch.Tensor:
    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 1 or logits.numel() == 0:
        raise RuntimeError(f"{context} expected [vocab], got {tuple(logits.shape)}")
    return logits


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[token_id]).sum().item()) + 1


def filtered_rank_of(logits: torch.Tensor, token_id: int, word_ids: torch.Tensor) -> int:
    selected = logits[word_ids]
    return int((selected > logits[token_id]).sum().item()) + 1


@torch.no_grad()
def readout_across_layers(
    lens_model: Any,
    lens: Any,
    input_ids: torch.Tensor,
    *,
    position: int,
    candidates: Sequence[Candidate],
    top_k: int,
    word_ids: torch.Tensor,
    residual_path: Path,
) -> dict[str, Any]:
    final_layer = lens_model.n_layers - 1
    layers = sorted(set(int(layer) for layer in lens.source_layers) | {final_layer})
    with ResidualHooks(
        lens_model.layers,
        capture_layers=layers,
        capture_position=position,
        capture_scale_layers=lens.source_layers,
    ) as hooks:
        lens_model.forward(input_ids.to(lens_model.input_device))
    residual_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "position": position,
            "token_id": int(input_ids[0, position]),
            "residuals": {
                layer: tensor[0].to(device="cpu", dtype=torch.float16)
                for layer, tensor in hooks.captured.items()
            },
            "layer_indexing": "zero-based decoder block output",
        },
        residual_path,
    )
    metrics: dict[str, dict[str, dict[str, Any]]] = {
        candidate.feature_id: {} for candidate in candidates
    }
    layer_rows = []
    for layer in layers:
        residual = hooks.captured[layer][0].float()
        if layer in lens.jacobians:
            residual = lens.transport(residual, layer)
        logits = _vocab_logits(
            lens_model.unembed(residual).float().detach().cpu(),
            f"readout layer {layer}",
        )
        values, indices = logits.topk(min(top_k, logits.numel()))
        for candidate in candidates:
            metrics[candidate.feature_id][str(layer)] = {
                "score": float(logits[candidate.token_id].item()),
                "raw_rank": rank_of(logits, candidate.token_id),
                "word_filtered_rank": filtered_rank_of(
                    logits, candidate.token_id, word_ids
                ),
            }
        layer_rows.append(
            {
                "layer": layer,
                "top_k": [
                    {
                        "rank": rank,
                        "token_id": int(token_id),
                        "label": lens_model.tokenizer.decode(
                            [int(token_id)], clean_up_tokenization_spaces=False
                        ),
                        "score": float(value),
                    }
                    for rank, (value, token_id) in enumerate(
                        zip(values.tolist(), indices.tolist(), strict=True), start=1
                    )
                ],
            }
        )
    return {
        "layers": layer_rows,
        "candidate_metrics": metrics,
        "mean_residual_scales": {
            str(layer): value for layer, value in hooks.mean_residual_scales.items()
        },
        "residual_state": {"path": str(residual_path.resolve())},
        "rank_policy": {
            "raw_rank": "all vocabulary entries",
            "word_filtered_rank": "jlens.vis meaningful-token display filter",
        },
    }


def build_steering_spec(
    *,
    mode: str,
    candidate: Candidate,
    requested_strength: float,
    direction_source_position: int,
    input_ids: torch.Tensor,
    tokenizer: Any,
    selected_position: int | None = None,
    localized_residual_scale: float | None = None,
    max_injection_fraction: float | None = None,
    steer_generated: bool = False,
) -> SteeringSpec:
    prompt_ids = [int(value) for value in input_ids[0].tolist()]
    bos_id = getattr(tokenizer, "bos_token_id", None)
    skipped = tuple(
        index
        for index, token_id in enumerate(prompt_ids)
        if bos_id is not None and token_id == int(bos_id)
    )
    if mode == "neuronpedia_global":
        if max_injection_fraction is None or max_injection_fraction <= 0:
            raise ValueError("global mode requires a positive cap")
        return SteeringSpec(
            mode=mode,
            layer=candidate.layer,
            direction=candidate.direction,
            requested_strength=float(requested_strength),
            direction_source_position=direction_source_position,
            prompt_length=len(prompt_ids),
            max_injection_fraction=float(max_injection_fraction),
            steer_generated=bool(steer_generated),
            skip_positions=skipped,
        )
    if mode == "single_position":
        return SteeringSpec(
            mode=mode,
            layer=candidate.layer,
            direction=candidate.direction,
            requested_strength=float(requested_strength),
            direction_source_position=direction_source_position,
            prompt_length=len(prompt_ids),
            selected_position=selected_position,
            localized_residual_scale=localized_residual_scale,
        )
    raise ValueError(f"unsupported intervention mode: {mode!r}")


def _last_logits(
    lens_model: Any, input_ids: torch.Tensor, steering: SteeringSpec | None
) -> tuple[torch.Tensor, HookAudit]:
    final_layer = lens_model.n_layers - 1
    with ResidualHooks(
        lens_model.layers,
        capture_layers=[final_layer],
        capture_position=input_ids.shape[1] - 1,
        steering=steering,
    ) as hooks:
        lens_model.forward(input_ids.to(lens_model.input_device))
    logits = _vocab_logits(
        lens_model.unembed(hooks.captured[final_layer][0]).float().detach().cpu(),
        "next-token scoring",
    )
    return logits, hooks.audit


def _eos_ids(tokenizer: Any) -> set[int]:
    value = tokenizer.eos_token_id
    if value is None:
        return set()
    if isinstance(value, list):
        return {int(item) for item in value}
    return {int(value)}


@torch.no_grad()
def generate_greedy(
    lens_model: Any,
    prefix_ids: torch.Tensor,
    tokenizer: Any,
    *,
    max_new_tokens: int,
    steering: SteeringSpec | None = None,
) -> dict[str, Any]:
    ids = prefix_ids.clone().cpu()
    generated: list[int] = []
    token_logprobs: list[float] = []
    entropies: list[float] = []
    hook_calls = 0
    max_sequence = 0
    for _ in range(max_new_tokens):
        logits, audit = _last_logits(lens_model, ids, steering)
        hook_calls += audit.calls
        max_sequence = max([max_sequence, *audit.sequence_lengths])
        log_probs = logits.log_softmax(dim=-1)
        probabilities = log_probs.exp()
        next_id = int(logits.argmax().item())
        generated.append(next_id)
        token_logprobs.append(float(log_probs[next_id].item()))
        entropies.append(float(-(probabilities * log_probs).sum().item()))
        ids = torch.cat([ids, torch.tensor([[next_id]], dtype=ids.dtype)], dim=1)
        decoded = tokenizer.decode(generated, skip_special_tokens=False)
        if next_id in _eos_ids(tokenizer) or (generated and "\n" in decoded.strip("\r")):
            break
    return {
        "raw": tokenizer.decode(generated, skip_special_tokens=True),
        "token_ids": generated,
        "token_logprobs": token_logprobs,
        "sequence_logprob": float(sum(token_logprobs)),
        "mean_entropy": float(sum(entropies) / len(entropies)) if entropies else None,
        "hook_audit": {
            "calls": hook_calls,
            "maximum_sequence_length": max_sequence,
            "generated_positions_steered": bool(
                steering is not None
                and steering.mode == "neuronpedia_global"
                and steering.steer_generated
                and max_sequence > steering.prompt_length
            ),
        },
    }


def _label_ids(tokenizer: Any, label: str) -> list[int]:
    ids = tokenizer(label, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if not ids:
        raise RuntimeError(f"label {label!r} tokenized to no tokens")
    return [int(item) for item in ids]


@torch.no_grad()
def sequence_logprob(
    lens_model: Any,
    prefix_ids: torch.Tensor,
    tokenizer: Any,
    label: str,
    steering: SteeringSpec | None = None,
) -> dict[str, Any]:
    ids = prefix_ids.clone().cpu()
    token_ids = _label_ids(tokenizer, label)
    values = []
    audits = []
    for token_id in token_ids:
        logits, audit = _last_logits(lens_model, ids, steering)
        values.append(float(logits.log_softmax(dim=-1)[token_id].item()))
        audits.append(asdict(audit))
        ids = torch.cat([ids, torch.tensor([[token_id]], dtype=ids.dtype)], dim=1)
    return {
        "label": label,
        "token_ids": token_ids,
        "token_logprobs": values,
        "sequence_logprob": float(sum(values)),
        "hook_audits": audits,
    }


def score_label_pair(
    lens_model: Any,
    prefix_ids: torch.Tensor,
    tokenizer: Any,
    labels: Sequence[str],
    steering: SteeringSpec | None = None,
) -> dict[str, Any]:
    first = sequence_logprob(lens_model, prefix_ids, tokenizer, labels[0], steering)
    second = sequence_logprob(lens_model, prefix_ids, tokenizer, labels[1], steering)
    return {
        labels[0]: first,
        labels[1]: second,
        "margin": first["sequence_logprob"] - second["sequence_logprob"],
        "margin_definition": f"logP({labels[0]}) - logP({labels[1]})",
    }


@torch.no_grad()
def candidate_score_after_steering(
    lens_model: Any,
    lens: Any,
    input_ids: torch.Tensor,
    candidate: Candidate,
    source_position: int,
    steering: SteeringSpec,
) -> float:
    with ResidualHooks(
        lens_model.layers,
        capture_layers=[candidate.layer],
        capture_position=source_position,
        steering=steering,
    ) as hooks:
        lens_model.forward(input_ids.to(lens_model.input_device))
    residual = lens.transport(hooks.captured[candidate.layer][0].float(), candidate.layer)
    logits = _vocab_logits(
        lens_model.unembed(residual).float().detach().cpu(),
        "post-steering candidate readout",
    )
    return float(logits[candidate.token_id].item())
