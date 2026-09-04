"""Deterministic execution profiles for full and exploratory campaigns."""

from __future__ import annotations

import copy
from typing import Any, Mapping


PROFILE_NAMES = ("full", "quick")

# These IDs were chosen before quick-profile execution.  They retain every item
# family and put one known incorrect calibration answer in each split.  Long
# prospective/knowledge-boundary answers remain represented, but their
# differentiable objective is bounded by the quick profile's token window.
QUICK_DISCOVERY_ITEM_IDS = (
    "1", "14", "43", "44", "48", "65", "66", "83",
)
QUICK_HELDOUT_ITEM_IDS = (
    "0", "2", "3", "4", "57", "67", "68", "82",
)


def resolve_execution_profile(
    base_config: Mapping[str, Any], profile: str
) -> dict[str, Any]:
    """Return a copied, fully resolved config for ``profile``."""
    name = str(profile)
    if name not in PROFILE_NAMES:
        raise ValueError(f"unknown execution profile {name!r}")
    config = copy.deepcopy(dict(base_config))
    if name == "full":
        return config

    selected = [*QUICK_DISCOVERY_ITEM_IDS, *QUICK_HELDOUT_ITEM_IDS]
    config["execution_profile"] = {
        "name": "quick",
        "exploratory": True,
        "selected_item_ids": selected,
        "gradient_answer_token_limit": 32,
        "claim_ceiling": "exploratory mechanistic evidence; not confirmatory",
    }
    config["split"] = {
        **config["split"],
        "discovery_counts": {
            "calibration": 6,
            "prospective": 1,
            "knowledge_boundary": 1,
        },
        "heldout_items": len(QUICK_HELDOUT_ITEM_IDS),
        "explicit_discovery_item_ids": list(QUICK_DISCOVERY_ITEM_IDS),
        "explicit_heldout_item_ids": list(QUICK_HELDOUT_ITEM_IDS),
    }
    config["strengths"] = {
        **config["strengths"],
        # psr-v7 already located the useful primary transition at 0.10/0.11.
        "alpha_grid": [0.10, 0.11],
        "beta_grid": [0.10, 0.20, 0.30],
        "weak_min_positive_items": 6,
    }
    config["layers"] = {
        **config["layers"],
        "readout": [38, 40, 42],
    }
    config["smoke"] = {**config["smoke"], "item_count": 2}
    config["readout"] = {**config["readout"], "top_k": 25}
    config["candidate_selection"] = {
        **config["candidate_selection"],
        "max_candidates": 1,
    }
    return config


def profile_name(config: Mapping[str, Any]) -> str:
    return str(config.get("execution_profile", {}).get("name", "full"))


def gradient_answer_token_limit(config: Mapping[str, Any]) -> int | None:
    value = config.get("execution_profile", {}).get("gradient_answer_token_limit")
    return None if value is None else int(value)

