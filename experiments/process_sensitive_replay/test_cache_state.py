from __future__ import annotations

import types
import unittest

import torch

from experiments.process_sensitive_replay.cache_state import (
    assert_cache_unchanged,
    assert_hybrid_cache_integrity,
    assert_process_propagated,
    assert_storage_disjoint,
    audit_cache,
    clone_hybrid_cache,
)


def fake_cache() -> types.SimpleNamespace:
    full0 = types.SimpleNamespace(
        keys=torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4),
        values=torch.ones(1, 1, 3, 4),
        is_initialized=True,
        cumulative_length=3,
    )
    full1 = types.SimpleNamespace(
        keys=torch.full((1, 1, 3, 4), 2.0),
        values=torch.full((1, 1, 3, 4), 3.0),
        is_initialized=True,
        cumulative_length=3,
    )
    linear = types.SimpleNamespace(
        conv_states={0: torch.ones(1, 8, 4)},
        recurrent_states={0: torch.ones(1, 2, 4, 4)},
        is_conv_states_initialized={0: True},
        is_recurrent_states_initialized={0: True},
        has_previous_state={0: True},
        conv_kernel_size={0: 4},
        record_past=False,
    )
    return types.SimpleNamespace(layers=[full0, linear, full1])


class CacheStateTests(unittest.TestCase):
    def test_actual_transformers_hybrid_dynamic_cache_clones_cleanly(self) -> None:
        from transformers import DynamicCache
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

        config = Qwen3_5TextConfig(
            num_hidden_layers=4,
            layer_types=[
                "linear_attention", "linear_attention",
                "linear_attention", "full_attention",
            ],
        )
        cache = DynamicCache(config=config)
        cache.update_conv_state(torch.ones(1, 8, 3), 0, conv_kernel_size=4)
        cache.update_recurrent_state(torch.ones(1, 2, 4, 4), 0)
        cache.update(torch.ones(1, 1, 3, 4), torch.ones(1, 1, 3, 4), 3)
        clone = clone_hybrid_cache(cache)
        self.assertEqual(type(cache.layers[0]).__name__, "LinearAttentionLayer")
        self.assertEqual(type(cache.layers[3]).__name__, "DynamicLayer")
        self.assertEqual(audit_cache(cache).digest, audit_cache(clone).digest)
        assert_storage_disjoint(cache, clone)

    def test_clone_is_equal_but_storage_disjoint(self) -> None:
        source = fake_cache()
        clone = clone_hybrid_cache(source)
        self.assertEqual(audit_cache(source).digest, audit_cache(clone).digest)
        assert_storage_disjoint(source, clone)
        clone.layers[0].keys.add_(1)
        self.assertNotEqual(audit_cache(source).digest, audit_cache(clone).digest)

    def test_source_mutation_is_detected(self) -> None:
        source = fake_cache()
        before = audit_cache(source)
        source.layers[1].conv_states[0].zero_()
        with self.assertRaisesRegex(AssertionError, "source cache mutated"):
            assert_cache_unchanged(before, source, "sibling branch")

    def test_hybrid_integrity_checks_all_state_families(self) -> None:
        source = fake_cache()
        audit = assert_hybrid_cache_integrity(
            source,
            layer_types=["full_attention", "linear_attention", "full_attention"],
            expected_sequence_length=3,
        )
        self.assertEqual(len(audit.layer_digests), 3)
        source.layers[1].recurrent_states[0] = None
        with self.assertRaisesRegex(AssertionError, "recurrent state"):
            assert_hybrid_cache_integrity(
                source,
                layer_types=["full_attention", "linear_attention", "full_attention"],
                expected_sequence_length=3,
            )

    def test_hybrid_integrity_requires_initialized_state_flags(self) -> None:
        for attribute in (
            "is_conv_states_initialized",
            "is_recurrent_states_initialized",
        ):
            source = fake_cache()
            getattr(source.layers[1], attribute)[0] = False
            with self.assertRaisesRegex(AssertionError, "initialization flags"):
                assert_hybrid_cache_integrity(
                    source,
                    layer_types=["full_attention", "linear_attention", "full_attention"],
                    expected_sequence_length=3,
                )

    def test_process_must_change_only_downstream_state(self) -> None:
        clean_cache = fake_cache()
        changed_cache = clone_hybrid_cache(clean_cache)
        changed_cache.layers[2].keys.add_(1)
        assert_process_propagated(
            audit_cache(clean_cache), audit_cache(changed_cache), process_layer=1
        )
        changed_cache.layers[0].keys.add_(1)
        with self.assertRaisesRegex(AssertionError, "upstream cache"):
            assert_process_propagated(
                audit_cache(clean_cache), audit_cache(changed_cache), process_layer=1
            )


if __name__ == "__main__":
    unittest.main()
