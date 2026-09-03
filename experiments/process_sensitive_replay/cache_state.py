"""Clone, hash, and audit Transformers hybrid DynamicCache instances."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class TensorAudit:
    path: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    sha256: str
    finite: bool
    data_ptr: int


@dataclass(frozen=True)
class CacheAudit:
    digest: str
    structure_digest: str
    tensors: tuple[TensorAudit, ...]
    layer_digests: tuple[str, ...]


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _finite(tensor: torch.Tensor) -> bool:
    return not (tensor.is_floating_point() or tensor.is_complex()) or bool(torch.isfinite(tensor).all().item())


def _walk_tensors(value: Any, path: str, seen: set[int]) -> Iterable[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        yield path, value
        return
    object_id = id(value)
    if object_id in seen:
        return
    if isinstance(value, Mapping):
        seen.add(object_id)
        for key in sorted(value, key=lambda item: str(item)):
            yield from _walk_tensors(value[key], f"{path}[{key!r}]", seen)
    elif isinstance(value, (list, tuple)):
        seen.add(object_id)
        for index, item in enumerate(value):
            yield from _walk_tensors(item, f"{path}[{index}]", seen)
    elif hasattr(value, "__dict__"):
        seen.add(object_id)
        for name in sorted(vars(value)):
            if name in {"prefetch_stream"}:
                continue
            yield from _walk_tensors(getattr(value, name), f"{path}.{name}", seen)


def cache_tensors(cache: Any) -> list[tuple[str, torch.Tensor]]:
    return list(_walk_tensors(cache, "cache", set()))


def clone_hybrid_cache(cache: Any) -> Any:
    """Deep-copy every cache tensor; fail if any storage remains shared."""
    cloned = copy.deepcopy(cache)
    assert_storage_disjoint(cache, cloned)
    return cloned


def _structure(value: Any, seen: set[int]) -> Any:
    if torch.is_tensor(value):
        return {
            "kind": "tensor", "shape": list(value.shape),
            "dtype": str(value.dtype), "device": str(value.device),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    object_id = id(value)
    if object_id in seen:
        return {"kind": "cycle", "type": type(value).__name__}
    if isinstance(value, Mapping):
        seen.add(object_id)
        return {str(key): _structure(value[key], seen) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        seen.add(object_id)
        return [_structure(item, seen) for item in value]
    if hasattr(value, "__dict__"):
        seen.add(object_id)
        return {
            "kind": "object",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "attributes": {
                name: _structure(attribute, seen)
                for name, attribute in sorted(vars(value).items())
                if name not in {"prefetch_stream"}
            },
        }
    return {"kind": "opaque", "type": type(value).__name__, "repr": repr(value)}


def audit_cache(cache: Any) -> CacheAudit:
    tensor_audits = tuple(
        TensorAudit(
            path=path,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            sha256=_tensor_digest(tensor),
            finite=_finite(tensor),
            data_ptr=int(tensor.data_ptr()),
        )
        for path, tensor in cache_tensors(cache)
    )
    structure_json = json.dumps(_structure(cache, set()), sort_keys=True, separators=(",", ":"))
    structure_digest = hashlib.sha256(structure_json.encode("utf-8")).hexdigest()
    combined = hashlib.sha256()
    combined.update(structure_digest.encode("ascii"))
    for record in tensor_audits:
        combined.update(record.path.encode("utf-8"))
        combined.update(record.sha256.encode("ascii"))
    layer_digests = tuple(audit_layer(layer) for layer in getattr(cache, "layers", ()))
    return CacheAudit(combined.hexdigest(), structure_digest, tensor_audits, layer_digests)


def audit_layer(layer: Any) -> str:
    combined = hashlib.sha256()
    structure_json = json.dumps(_structure(layer, set()), sort_keys=True, separators=(",", ":"))
    combined.update(structure_json.encode("utf-8"))
    for path, tensor in _walk_tensors(layer, "layer", set()):
        combined.update(path.encode("utf-8"))
        combined.update(_tensor_digest(tensor).encode("ascii"))
    return combined.hexdigest()


def assert_storage_disjoint(first: Any, second: Any) -> None:
    first_tensors = dict(cache_tensors(first))
    second_tensors = dict(cache_tensors(second))
    if set(first_tensors) != set(second_tensors):
        raise AssertionError("cache clones have different tensor topology")
    aliases = []
    for path in first_tensors:
        left, right = first_tensors[path], second_tensors[path]
        if left.device == right.device and left.data_ptr() == right.data_ptr() and left.numel():
            aliases.append(path)
    if aliases:
        raise AssertionError(f"cache clones share tensor storage: {aliases[:5]}")


def assert_cache_unchanged(before: CacheAudit, cache: Any, context: str) -> None:
    after = audit_cache(cache)
    if before.digest != after.digest:
        raise AssertionError(f"source cache mutated during {context}")


def assert_hybrid_cache_integrity(
    cache: Any,
    *,
    layer_types: Sequence[str],
    expected_sequence_length: int,
) -> CacheAudit:
    layers = getattr(cache, "layers", None)
    if layers is None or len(layers) != len(layer_types):
        raise AssertionError("cache layer topology does not match model layer_types")
    for index, (layer, layer_type) in enumerate(zip(layers, layer_types, strict=True)):
        if layer_type == "full_attention":
            keys, values = getattr(layer, "keys", None), getattr(layer, "values", None)
            if not torch.is_tensor(keys) or not torch.is_tensor(values) or not keys.numel() or not values.numel():
                raise AssertionError(f"full-attention cache layer {index} is uninitialized")
            if keys.shape[-2] != expected_sequence_length or values.shape[-2] != expected_sequence_length:
                raise AssertionError(f"full-attention cache layer {index} has wrong sequence length")
        elif layer_type == "linear_attention":
            conv = getattr(layer, "conv_states", None)
            recurrent = getattr(layer, "recurrent_states", None)
            if not isinstance(conv, dict) or not isinstance(recurrent, dict):
                raise AssertionError(f"linear-attention cache layer {index} lacks hybrid states")
            if not all(torch.is_tensor(value) and value.numel() for value in conv.values()):
                raise AssertionError(f"linear-attention conv state {index} is uninitialized")
            if not all(torch.is_tensor(value) and value.numel() for value in recurrent.values()):
                raise AssertionError(f"linear-attention recurrent state {index} is uninitialized")
            if not all(getattr(layer, "has_previous_state", {}).values()):
                raise AssertionError(f"linear-attention layer {index} has no previous-state flag")
        else:
            raise AssertionError(f"unexpected layer type {layer_type!r} at {index}")
    audit = audit_cache(cache)
    nonfinite = [record.path for record in audit.tensors if not record.finite]
    if nonfinite:
        raise AssertionError(f"cache contains non-finite tensors: {nonfinite[:5]}")
    return audit


def assert_process_propagated(
    clean: CacheAudit,
    perturbed: CacheAudit,
    *,
    process_layer: int,
) -> None:
    if len(clean.layer_digests) != len(perturbed.layer_digests):
        raise AssertionError("cache layer counts differ")
    changed_before_or_at = [
        index for index in range(process_layer + 1)
        if clean.layer_digests[index] != perturbed.layer_digests[index]
    ]
    if changed_before_or_at:
        raise AssertionError(f"upstream cache unexpectedly changed at layers {changed_before_or_at}")
    downstream = [
        index for index in range(process_layer + 1, len(clean.layer_digests))
        if clean.layer_digests[index] != perturbed.layer_digests[index]
    ]
    if not downstream:
        raise AssertionError("perturbation did not propagate into downstream persistent state")


def assert_numeric_parity(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    context: str,
) -> None:
    if left.shape != right.shape or not torch.allclose(left.float(), right.float(), atol=atol, rtol=rtol):
        maximum = math.inf
        if left.shape == right.shape and left.numel():
            maximum = float((left.float() - right.float()).abs().max().item())
        raise AssertionError(f"reset parity failed for {context}; max_abs_difference={maximum}")

