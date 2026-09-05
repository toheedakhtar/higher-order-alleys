"""One explicit FP32 tail; no alternate model equations or attention dispatch.

Zero-based block 42 finishes in BF16. Blocks 43..63, final norm, and head
use promoted *loaded BF16 values*. Prefix processing must finish before entry.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import torch

from experiments.process_sensitive_replay.cache_state import audit_cache, cache_tensors, _structure


@dataclass(frozen=True)
class MixedPolicy:
    name: str = "bf16_process_fp32_tail_v1"
    patch_layer: int = 42
    first_fp32_block: int = 43
    last_fp32_block: int = 63
    coordinate_relative_tolerance: float = .01
    sham_atol: float = 1e-5
    sham_rtol: float = 1e-5
    random_seed: int = 1729
    # Conservative forward error bound for multiply + add, including scalar
    # conversion. Not an empirically selected percentage leakage allowance.
    rounding_bound_eps_multiplier: float = 4.0


POLICY = MixedPolicy()


def tensor_sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def realize_patch(h, displacement, vector, *, intended_coordinate=None):
    """Single FP32 add; geometry measured in FP64, without behavioral inputs."""
    if h.dtype != torch.bfloat16 or h.ndim != 1:
        raise AssertionError("recipient must be the original flat BF16 residual")
    h = h.detach().cpu()
    v = vector.detach().cpu().double()
    d = displacement.detach().cpu().double()
    if d.shape != h.shape or v.shape != h.shape or not all(torch.isfinite(x).all() for x in (h, d, v)):
        raise AssertionError("invalid patch geometry")
    # Zero is deliberately a true no-op following lossless promotion.
    q = h.float() if not bool(d.any()) else h.float() + d.float()
    realized = q.double() - h.double()
    coordinate = float(v @ realized)
    intended = float(v @ d) if intended_coordinate is None else float(intended_coordinate)
    error = abs(coordinate - intended)
    target_norm = float(d.norm())
    # Noise orthogonal to the intended displacement; candidate-axis leakage
    # is reported separately for random/full-residual patches.
    if target_norm:
        orth = realized - d * ((realized @ d) / (d @ d))
    else:
        orth = realized
    bound = POLICY.rounding_bound_eps_multiplier * torch.finfo(torch.float32).eps * float((h.double().abs() + d.abs()).norm())
    full_error = float((realized - d).norm())
    return q, {
        "intended_candidate_coordinate_change": intended,
        "realized_candidate_coordinate_change": coordinate,
        "absolute_coordinate_error": error,
        "relative_coordinate_error": error / abs(intended) if intended else (0.0 if error == 0 else None),
        "intended_patch_l2": target_norm,
        "total_patch_l2": float(realized.norm()),
        "orthogonal_leakage_l2": float(orth.norm()),
        "orthogonal_leakage_over_intended_coordinate": float(orth.norm()) / abs(intended) if intended else None,
        "orthogonal_leakage_over_intended_patch_norm": float(orth.norm()) / target_norm if target_norm else 0.0,
        "rounding_error_l2": full_error,
        "floating_point_noise_l2_bound": bound,
        "noise_bound_passed": full_error <= bound and float(orth.norm()) <= bound,
        "resulting_residual_dtype": str(q.dtype),
        "realized_residual_sha256": tensor_sha(q),
    }


def candidate_patch(h, donor, vector):
    # Frozen v is not renormalized; its recorded norm differs from one only
    # at FP32 representation error. Use exactly the user's delta*v formula.
    v = vector.detach().cpu().double()
    delta = float(v @ (donor.detach().cpu().double() - h.detach().cpu().double()))
    q, report = realize_patch(h, delta * v, vector, intended_coordinate=delta)
    report["coordinate_accuracy_passed"] = report["absolute_coordinate_error"] <= .01 * abs(delta)
    report["precision_gate_passed"] = report["coordinate_accuracy_passed"] and report["noise_bound_passed"]
    return q, report


def promote_tail_cache(cache):
    """Promote isolated cache storage losslessly; preserve flags and positions."""
    if len(cache.layers) != 64:
        raise AssertionError("expected complete 64-layer hybrid state")
    before = audit_cache(cache)
    before_structure = _structure(cache, set())
    for index, layer in enumerate(cache.layers):
        if index < 43:
            continue
        def convert(value):
            if torch.is_tensor(value):
                if value.is_floating_point():
                    if value.dtype not in (torch.bfloat16, torch.float32):
                        raise AssertionError("unexpected cache dtype")
                    changed = value.float()
                    if not torch.equal(changed.to(value.dtype), value):
                        raise AssertionError("cache promotion changed values")
                    return changed
                return value
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, tuple):
                return tuple(convert(v) for v in value)
            if isinstance(value, list):
                return [convert(v) for v in value]
            if isinstance(value, torch.dtype) and value == torch.bfloat16:
                return torch.float32
            return value
        for key, value in tuple(vars(layer).items()):
            setattr(layer, key, convert(value))
    after = audit_cache(cache)
    if before.layer_digests[:43] != after.layer_digests[:43]:
        raise AssertionError("upstream cache changed during promotion")
    def dtype_neutral(value):
        if isinstance(value, dict):
            return {k: dtype_neutral(v) for k, v in value.items()}
        if isinstance(value, list):
            return [dtype_neutral(v) for v in value]
        if isinstance(value, str):
            return value.replace("torch.bfloat16", "promoted_float").replace("torch.float32", "promoted_float")
        return value
    if dtype_neutral(before_structure) != dtype_neutral(_structure(cache, set())):
        raise AssertionError("cache flags, positions, or topology changed during promotion")
    return {"before": asdict(before), "after": asdict(after), "lossless": True}


class FP32Tail:
    """Temporary parameter representations; original BF16 storage saved on CPU.

    The actual model modules and forward methods remain in place. All earlier
    tokens must be computed outside this context. Never call J-Lens inside it.
    IEEE math settings are active only from block 43 through lm_head, so Qwen's
    FP32 recurrent arithmetic in blocks 0..42 keeps its upstream settings.
    """
    def __init__(self, adapter):
        self.adapter = adapter
        self.saved = []
        self.handles = []
        self.weight_audit = []
        self.precision_settings = []
        self.backend = adapter.text_config._attn_implementation

    def _ieee(self, *_args):
        for owner, _ in self.precision_settings:
            owner.fp32_precision = "ieee"

    def _restore_math(self, *_args):
        for owner, value in self.precision_settings:
            owner.fp32_precision = value

    def _tail_inputs(self, _module, args, kwargs):
        if any(t.dtype != torch.float32 for t in _module.parameters() if t.is_floating_point()):
            raise AssertionError("tail weight representation changed during execution")
        self._ieee()
        def promote(x):
            if torch.is_tensor(x) and x.is_floating_point():
                return x.float()
            if isinstance(x, tuple):
                return tuple(promote(y) for y in x)
            return x
        return tuple(promote(x) for x in args), {k: promote(v) for k, v in kwargs.items()}

    def __enter__(self):
        if getattr(self.adapter, "_mixed_tail_active", False):
            raise AssertionError("nested FP32 tails are unsupported")
        if len(self.adapter.layers) != 64:
            raise AssertionError("unexpected model architecture")
        modules = [(f"layer_{i}", self.adapter.layers[i]) for i in range(43, 64)]
        modules += [("final_norm", self.adapter.text_module.norm), ("lm_head", self.adapter.lm_head)]
        targets = [(f"{name}.{key}", t) for name, m in modules
                   for key, t in list(m.named_parameters()) + list(m.named_buffers()) if t.is_floating_point()]
        target_ids = {id(t) for _, t in targets}
        pointers = {t.untyped_storage().data_ptr() for _, t in targets}
        for t in list(self.adapter.hf_model.parameters()) + list(self.adapter.hf_model.buffers()):
            if id(t) not in target_ids and t.untyped_storage().data_ptr() in pointers:
                raise AssertionError("tail storage aliases upstream computation")
        # An identity alias (e.g. tied input embeddings/output head) is also
        # forbidden, even when named_parameters removes duplicate names.
        for module in [self.adapter.text_module.embed_tokens, *self.adapter.layers[:43]]:
            if any(id(t) in target_ids for t in list(module.parameters()) + list(module.buffers())):
                raise AssertionError("tail parameters tied to upstream computation")
        try:
            for owner in (torch.backends.cuda.matmul, torch.backends.cudnn.conv):
                self.precision_settings.append((owner, owner.fp32_precision))
            seen = set()
            for name, tensor in targets:
                if id(tensor) in seen:
                    continue
                seen.add(id(tensor))
                if tensor.dtype not in (torch.bfloat16, torch.float32):
                    raise AssertionError("unsupported original weight dtype")
                if tensor.dtype == torch.float32:
                    continue
                saved = tensor.detach().cpu().clone()
                self.saved.append((tensor, saved, tensor.device))
                self.weight_audit.append({"name": name, "source_dtype": str(saved.dtype), "source_sha256": tensor_sha(saved)})
                tensor.data = tensor.detach().float()
                if not torch.equal(tensor.detach().cpu(), saved.float()):
                    raise AssertionError("FP32 weights do not equal loaded BF16 values")
            for _, module in modules[:21]:
                self.handles.append(module.register_forward_pre_hook(self._tail_inputs, with_kwargs=True))
            self.handles.append(self.adapter.lm_head.register_forward_hook(self._restore_math))
            self.adapter._mixed_tail_active = True
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._restore_math()
        for tensor, saved, device in reversed(self.saved):
            tensor.data = saved.to(device)
        self.saved.clear()
        self.adapter._mixed_tail_active = False
        if self.adapter.text_config._attn_implementation != self.backend:
            raise AssertionError("attention implementation changed")


def state_topology(cache):
    """Shape/path comparison excludes deliberate dtype changes, not finiteness."""
    return [{"path": p, "shape": list(t.shape), "dtype": str(t.dtype),
             "finite": bool(torch.isfinite(t).all())} for p, t in cache_tensors(cache)]
