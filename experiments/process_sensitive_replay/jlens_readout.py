"""J-Lens scoring for residuals captured during cached Qwen replay."""

from __future__ import annotations

from typing import Any, Iterable

import torch


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[int(token_id)]).sum().item()) + 1


@torch.no_grad()
def readout_residual(
    lens_model: Any,
    lens: Any,
    residual: torch.Tensor,
    *,
    layer: int,
    top_k: int,
    explicit_token_ids: Iterable[int],
) -> dict[str, Any]:
    if layer not in lens.jacobians:
        raise RuntimeError(f"J-Lens checkpoint has no fitted layer {layer}")
    value = residual.float()
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1 or value.numel() != lens_model.d_model:
        raise ValueError(f"expected residual [{lens_model.d_model}], got {tuple(value.shape)}")
    transported = lens.transport(value.to(lens.jacobians[layer].device), layer)
    logits = lens_model.unembed(transported).float().detach().cpu()
    if logits.ndim != 1:
        logits = logits.reshape(-1)
    count = min(int(top_k), logits.numel())
    values, ids = logits.topk(count)
    explicit = {
        str(int(token_id)): {
            "token_id": int(token_id),
            "score": float(logits[int(token_id)].item()),
            "raw_rank": rank_of(logits, int(token_id)),
            "label": lens_model.tokenizer.decode(
                [int(token_id)], clean_up_tokenization_spaces=False
            ),
        }
        for token_id in explicit_token_ids
    }
    return {
        "layer": int(layer),
        "top_k": [
            {
                "rank": index,
                "token_id": int(token_id),
                "label": lens_model.tokenizer.decode(
                    [int(token_id)], clean_up_tokenization_spaces=False
                ),
                "score": float(score),
            }
            for index, (score, token_id) in enumerate(
                zip(values.tolist(), ids.tolist(), strict=True), start=1
            )
        ],
        "explicit": explicit,
    }


def readout_layers(
    lens_model: Any,
    lens: Any,
    residuals: dict[int, torch.Tensor],
    *,
    layers: Iterable[int],
    top_k: int,
    explicit_token_ids: Iterable[int],
) -> dict[str, Any]:
    requested = [int(layer) for layer in layers]
    missing = sorted(set(requested) - set(residuals))
    if missing:
        raise RuntimeError(f"missing captured residuals for layers {missing}")
    return {
        str(layer): readout_residual(
            lens_model,
            lens,
            residuals[layer],
            layer=layer,
            top_k=top_k,
            explicit_token_ids=explicit_token_ids,
        )
        for layer in requested
    }

