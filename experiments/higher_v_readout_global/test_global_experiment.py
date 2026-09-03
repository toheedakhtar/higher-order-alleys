"""CPU-only regression tests for the new global experiment package."""

from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from . import analysis, protocol, runner, steering


class TinyTokenizer:
    eos_token_id = 0
    bos_token_id = 1

    def __len__(self) -> int:
        return 32

    def apply_chat_template(
        self, messages, *, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    ):
        text = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        special = {"评价": [7], "评估": [8], "PASS": [2], "FAIL": [3]}
        ids = special.get(text, [10 + ord(char) % 22 for char in text])
        result = {
            "input_ids": torch.tensor([ids], dtype=torch.long)
            if kwargs.get("return_tensors") else ids
        }
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = torch.tensor(
                [[(index, index + 1) for index in range(len(text))]], dtype=torch.long
            )
        return result

    @staticmethod
    def decode(token_ids, **_kwargs):
        ids = [int(value) for value in token_ids]
        if ids == [0]:
            return "<eos>"
        if ids == [2]:
            return "PASS"
        if ids == [3]:
            return "FAIL"
        if ids == [7]:
            return "评价"
        if ids == [8]:
            return "评估"
        if len(ids) == 1:
            return chr((ids[0] - 10) % 22 + 97)
        return "".join(TinyTokenizer.decode([value]) for value in ids)


class TinyModel:
    def __init__(self) -> None:
        torch.manual_seed(4)
        self.n_layers = 3
        self.d_model = 4
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.embedding = nn.Embedding(32, 4)
        self.norm = nn.LayerNorm(4)
        self._lm_head = nn.Linear(4, 32, bias=False)
        self.tokenizer = TinyTokenizer()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def parameters(self):
        for module in (self.layers, self.embedding, self.norm, self._lm_head):
            yield from module.parameters()

    @property
    def input_device(self):
        return torch.device("cpu")

    def forward(self, input_ids):
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
        return hidden

    def unembed(self, residual):
        return self._lm_head(self.norm(residual))


class TinyLens:
    def __init__(self):
        self.jacobians = {0: torch.eye(4), 1: torch.eye(4)}
        self.source_layers = [0, 1]
        self.d_model = 4

    def transport(self, residual, layer):
        return residual @ self.jacobians[layer].T


def fake_candidate(token_id=7, layer=0):
    return steering.Candidate(
        label="评价" if token_id == 7 else "评估",
        token_id=token_id,
        feature_id=f"vocab_token:{token_id}",
        layer=layer,
        direction=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        direction_sha256="synthetic",
        direction_path="synthetic.pt",
    )


class GlobalExperimentTests(unittest.TestCase):
    def test_static_config_and_all_dataset_rows(self):
        config = runner.load_config(runner.DEFAULT_CONFIG)
        rows = protocol.load_dataset_rows(runner.REPO_ROOT / "dataset/metacognition.csv")
        report = runner.validate_config(config, rows)
        self.assertEqual(report["all_sample_count"], 90)
        self.assertEqual(report["primary"]["candidate_token_id"], 97817)
        self.assertEqual(report["analysis_families"][0], "frozen_primary")

    def test_moved_gate_accepts_relocated_identical_dataset(self):
        config = runner.load_config(runner.DEFAULT_CONFIG)
        stored_config = copy.deepcopy(config)
        stored_config["dataset"]["path"] = "datasets/MMB/metacognition.csv"
        dataset_path = runner.resolve_repo_path(config["dataset"]["path"])
        with tempfile.TemporaryDirectory() as temporary:
            gate = Path(temporary)
            (gate / "run_manifest.json").write_text(json.dumps({
                "run_id": "moved-pilot",
                "phase": "pilot",
                "config": stored_config,
                "dataset": {"sha256": runner.sha256_file(dataset_path)},
            }), encoding="utf-8")
            (gate / "pilot_summary.json").write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )
            evidence = runner.validate_gate_run(gate, "pilot", config)
        self.assertEqual(evidence["run_id"], "moved-pilot")

    def test_moved_gate_rejects_different_dataset_content(self):
        config = runner.load_config(runner.DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            gate = Path(temporary)
            (gate / "run_manifest.json").write_text(json.dumps({
                "run_id": "wrong-dataset",
                "phase": "pilot",
                "config": config,
                "dataset": {"sha256": "not-the-current-dataset"},
            }), encoding="utf-8")
            (gate / "pilot_summary.json").write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "dataset content mismatch"):
                runner.validate_gate_run(gate, "pilot", config)

    def test_protocol_preserves_original_judgment(self):
        rows = protocol.load_dataset_rows(runner.REPO_ROOT / "dataset/metacognition.csv")
        item = next(row for row in rows if row["item_id"] == "73")
        trial = protocol.build_trial_protocol(item)
        self.assertEqual(
            trial.judgment_prompt,
            "Did your answer PASS or FAIL? Reply with exactly `PASS` or `FAIL`.",
        )
        self.assertTrue(trial.factual_prompt.endswith(item["detail"]))

    def test_question_mark_and_meaningful_predecessor_use_offsets(self):
        tokenizer = TinyTokenizer()
        messages = [{"role": "user", "content": "Did it PASS or FAIL? Reply PASS or FAIL."}]
        ids, rendered, _offsets, positions = protocol.locate_judgment_positions(tokenizer, messages)
        question = positions["question_mark"]
        previous = positions["meaningful_token_before_question_mark"]
        self.assertEqual(rendered[question.char_index], "?")
        self.assertLess(previous.index, question.index)
        self.assertEqual(ids.ndim, 2)

    def test_global_changes_every_non_bos_position_with_per_position_norm(self):
        layers = nn.ModuleList([nn.Identity()])
        hidden = torch.tensor([[[3.0, 4.0], [0.0, 2.0], [6.0, 8.0]]])
        spec = steering.SteeringSpec(
            mode="neuronpedia_global", layer=0, direction=torch.tensor([1.0, 0.0]),
            requested_strength=-1.7, direction_source_position=1, prompt_length=2,
            max_injection_fraction=1.0, steer_generated=True, skip_positions=(0,),
        )
        with steering.ResidualHooks(
            layers, capture_layers=[0], capture_position=1, steering=spec
        ) as hooks:
            output = layers[0](hidden)
        self.assertTrue(torch.equal(output[0, 0], hidden[0, 0]))
        self.assertTrue(torch.allclose(output[0, 1], torch.tensor([-2.0, 2.0])))
        self.assertTrue(torch.allclose(output[0, 2], torch.tensor([-4.0, 8.0])))
        self.assertEqual(hooks.audit.changed_positions_per_call, [2])

    def test_global_generated_positions_can_be_disabled(self):
        layers = nn.ModuleList([nn.Identity()])
        hidden = torch.ones(1, 3, 2)
        spec = steering.SteeringSpec(
            mode="neuronpedia_global", layer=0, direction=torch.tensor([1.0, 0.0]),
            requested_strength=0.5, direction_source_position=1, prompt_length=2,
            max_injection_fraction=1.0, steer_generated=False,
        )
        with steering.ResidualHooks(
            layers, capture_layers=[0], capture_position=1, steering=spec
        ):
            output = layers[0](hidden)
        self.assertFalse(torch.equal(output[0, 1], hidden[0, 1]))
        self.assertTrue(torch.equal(output[0, 2], hidden[0, 2]))

    def test_cap_makes_negative_1_7_and_1_8_identical(self):
        layers = nn.ModuleList([nn.Identity()])
        hidden = torch.tensor([[[0.0, 2.0], [3.0, 4.0]]])
        outputs = []
        for strength in (-1.7, -1.8):
            spec = steering.SteeringSpec(
                mode="neuronpedia_global", layer=0,
                direction=torch.tensor([1.0, 0.0]), requested_strength=strength,
                direction_source_position=1, prompt_length=2,
                max_injection_fraction=1.0, steer_generated=True,
            )
            with steering.ResidualHooks(
                layers, capture_layers=[0], capture_position=1, steering=spec
            ):
                outputs.append(layers[0](hidden))
            self.assertEqual(spec.effective_strength_after_cap, -1.0)
        # The intended capped magnitude is identical; tiny differences are
        # permitted because the uncapped multiply happens before float rounding.
        self.assertTrue(torch.allclose(outputs[0], outputs[1], atol=1e-6, rtol=1e-6))

    def test_localized_mode_changes_one_cell_only(self):
        layers = nn.ModuleList([nn.Identity()])
        hidden = torch.zeros(1, 4, 3)
        spec = steering.SteeringSpec(
            mode="single_position", layer=0, direction=torch.tensor([1.0, 0.0, 0.0]),
            requested_strength=-1.5, direction_source_position=1, prompt_length=4,
            selected_position=2, localized_residual_scale=2.0,
        )
        with steering.ResidualHooks(
            layers, capture_layers=[0], capture_position=2, steering=spec
        ) as hooks:
            output = layers[0](hidden)
        self.assertEqual(float(output[0, 2, 0]), -3.0)
        self.assertEqual(int(torch.count_nonzero(output[:, [0, 1, 3], :])), 0)
        self.assertEqual(hooks.audit.changed_positions_per_call, [1])

    def test_central_builder_separates_source_from_scope(self):
        tokenizer = TinyTokenizer()
        candidate = fake_candidate()
        ids = torch.tensor([[1, 4, 5, 6]])
        spec = steering.build_steering_spec(
            mode="neuronpedia_global", candidate=candidate, requested_strength=-1.7,
            direction_source_position=3, input_ids=ids, tokenizer=tokenizer,
            max_injection_fraction=1.0, steer_generated=True,
        )
        self.assertIsNone(spec.selected_position)
        self.assertEqual(spec.direction_source_position, 3)
        self.assertEqual(spec.applied_prompt_position_count, 3)
        self.assertEqual(spec.intervention_scope, "all_prompt_positions_and_generated_tokens")

    def test_raw_and_word_filtered_ranks_are_distinct(self):
        logits = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0])
        word_ids = torch.tensor([2, 3, 4])
        self.assertEqual(steering.rank_of(logits, 3), 4)
        self.assertEqual(steering.filtered_rank_of(logits, 3, word_ids), 2)

    def test_candidate_direction_uses_explicit_token_identity(self):
        model, lens = TinyModel(), TinyLens()
        with tempfile.TemporaryDirectory() as directory:
            candidate = steering.resolve_candidate(
                {"label": "评价", "token_id": 7}, 0, model.tokenizer,
                model, lens, Path(directory),
            )
            self.assertEqual(candidate.feature_id, "vocab_token:7")
            self.assertAlmostEqual(float(candidate.direction.norm()), 1.0, places=6)
            self.assertTrue(Path(candidate.direction_path).is_file())

    def test_readout_records_both_rank_types(self):
        model, lens = TinyModel(), TinyLens()
        candidate = fake_candidate()
        with tempfile.TemporaryDirectory() as directory:
            result = steering.readout_across_layers(
                model, lens, torch.tensor([[1, 2, 3, 4]]), position=2,
                candidates=[candidate], top_k=3,
                word_ids=torch.arange(32),
                residual_path=Path(directory) / "residual.pt",
            )
        metric = result["candidate_metrics"][candidate.feature_id]["0"]
        self.assertIn("raw_rank", metric)
        self.assertIn("word_filtered_rank", metric)

    def test_adaptive_plan_excludes_frozen_primary_condition(self):
        readout = {"candidate_metrics": {
            "vocab_token:97817": {
                "10": {"word_filtered_rank": 2}, "40": {"word_filtered_rank": 1}
            },
            "vocab_token:99973": {
                "20": {"word_filtered_rank": 3}, "40": {"word_filtered_rank": 4}
            },
        }}
        specs = [{"token_id": 97817}, {"token_id": 99973}]
        plan = runner.adaptive_layer_plan(
            readout, specs, primary_token_id=97817, primary_layer=40,
            threshold=50, maximum=2,
        )
        keys = [(int(spec["token_id"]), layer) for spec, layer, _reason in plan]
        self.assertNotIn((97817, 40), keys)
        self.assertEqual(keys[0], (97817, 10))
        self.assertIn((99973, 40), keys)

    def test_global_primary_is_not_duplicated_across_local_positions(self):
        config = runner.load_config(runner.DEFAULT_CONFIG)
        plan = runner.primary_condition_plan(config)
        self.assertEqual([row["requested_strength"] for row in plan], [-1.7, 1.8])
        self.assertTrue(all(row["target_selector"] is None for row in plan))
        self.assertTrue(all(row["mode"] == "neuronpedia_global" for row in plan))

    def test_execute_primary_writes_two_global_rows_and_one_summary_baseline(self):
        model, lens, tokenizer = TinyModel(), TinyLens(), TinyTokenizer()
        candidate = fake_candidate(token_id=7, layer=0)
        position = protocol.SelectedPosition(
            index=1, token_id=2, token="?", char_index=1, char_span=(1, 2),
            selector="question_mark", conversation_position="test", surrounding_tokens=[],
        )
        readout = {
            "candidate_metrics": {candidate.feature_id: {
                "0": {"score": 1.0, "raw_rank": 2, "word_filtered_rank": 2}
            }},
            "mean_residual_scales": {"0": 1.0},
        }
        prepared = runner.PreparedTrial(
            item={"item_id": "x", "item_type": "prospective", "domain": "test",
                  "difficulty": "hard", "answer_key": "x", "prompt": "?"},
            condition="self", messages=[{"role": "user", "content": "?"}],
            input_ids=torch.tensor([[1, 2, 4]]), positions={"question_mark": position},
            readouts={"question_mark": readout}, labels=("PASS", "FAIL"),
            expected_judgment="PASS", factual_correct=True, factual_answer="x",
            factual_invalid=False, factual_scoring={"scoring_method": "test"},
            pre_generation=None, pre_normalized=None, pre_valid=None,
            baseline_generation={"raw": "PASS"}, baseline_normalized="PASS",
            baseline_valid=True, baseline_scores={"margin": 0.5},
        )
        config = {
            "rank_policy": {"appearance_rank": "word_filtered_rank", "appearance_rank_threshold": 50},
            "generation": {"max_choice_tokens": 1},
            "candidates": [{"label": "评价", "token_id": 7, "enabled": True}],
            "interventions": {
                "primary": {"candidate_token_id": 7, "layer": 0,
                            "direction_source_selector": "question_mark",
                            "strengths": [0, -1.7, 1.8],
                            "max_injection_fraction": 1.0, "steer_generated": True},
                "localized_control": {"enabled": False},
                "adaptive_rescue": {"enabled": False},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "checkpoints").mkdir()
            recorder = runner.Recorder(run_dir, "synthetic")
            result = runner.execute_prepared_trial(
                recorder, model, lens, tokenizer, prepared, [candidate], config,
                run_localized=False, run_adaptive=False,
            )
            with (run_dir / "trial_summary.csv").open("r", encoding="utf-8") as handle:
                summaries = list(csv.DictReader(handle))
            recorder.close()
        self.assertEqual(len(result["primary_rows"]), 2)
        self.assertEqual(len(summaries), 1)
        self.assertTrue(all(row["intervention_mode"] == "neuronpedia_global" for row in result["primary_rows"]))
        self.assertTrue(all(row["localized_target_selector"] is None for row in result["primary_rows"]))

    def test_primary_analysis_rejects_duplicate_sample_strength(self):
        rows = pd.DataFrame([
            {"item_id": "1", "requested_strength": -1.7, "analysis_family": "frozen_primary", "is_primary_estimand": True, "intervention_mode": "neuronpedia_global"},
            {"item_id": "1", "requested_strength": -1.7, "analysis_family": "frozen_primary", "is_primary_estimand": True, "intervention_mode": "neuronpedia_global"},
        ])
        with self.assertRaisesRegex(ValueError, "sample-unique"):
            analysis.primary_rows(rows)

    def test_bootstrap_clusters_attempts_by_item(self):
        frame = pd.DataFrame({
            "item_id": ["a", "a", "a", "b"],
            "value": [0.0, 0.0, 0.0, 10.0],
        })
        rng = __import__("numpy").random.default_rng(1)
        low, high = analysis.bootstrap_item_interval(frame, "value", rng, iterations=1000)
        self.assertLessEqual(low, 5.0)
        self.assertGreaterEqual(high, 5.0)

    def test_export_validation_preserves_231_ids_and_position_214(self):
        messages = [
            {"role": "user", "content": "predict"},
            {"role": "assistant", "content": "WILL_PASS"},
            {"role": "user", "content": "Crewther question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "Did it PASS or FAIL?"},
            {"role": "assistant", "content": "FAIL"},
        ]
        tokens = [
            {"position": index, "id": index % 32, "token": "?" if index == 214 else "x"}
            for index in range(231)
        ]
        payload = {
            "kind": "chat", "modelId": "qwen3.6-27b", "messages": messages,
            "meta": {"prompt_len": 231}, "tokens": tokens,
            "steer": {"config": {
                "token": "评价", "type": "JACOBIAN_LENS", "layers": [40],
                "strength": -1.7, "ablate": False, "mode": "steer",
                "steerGenerated": True,
            }, "tokens": tokens},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "export.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = protocol.load_neuronpedia_export(path, {
                "expected_prompt_tokens": 231, "expected_source_position": 214
            })
        self.assertEqual(result["prompt_len"], 231)
        self.assertEqual(result["source_position"], 214)
        self.assertEqual(len(result["input_token_ids"]), 231)

    def test_intervention_schema_contains_raw_normalized_and_scope_fields(self):
        required = {
            "baseline_output_raw", "baseline_output_normalized",
            "intervened_output_raw", "intervened_output_normalized",
            "direction_source_position", "intervention_scope", "raw_rank",
            "word_filtered_rank", "effective_strength_after_cap",
        }
        self.assertTrue(required.issubset(runner.INTERVENTION_FIELDS))

    def test_mode_aware_analysis_uses_only_frozen_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "plots").mkdir()
            (run_dir / "run_manifest.json").write_text(
                json.dumps({"run_id": "synthetic", "seed": 42}), encoding="utf-8"
            )
            summary = {field: "" for field in runner.SUMMARY_FIELDS}
            summary.update({
                "item_id": "1", "item_type": "prospective", "condition": "self",
                "difficulty": "hard", "baseline_valid": True,
                "baseline_output_normalized": "FAIL", "factual_correct": False,
            })
            with (run_dir / "trial_summary.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=runner.SUMMARY_FIELDS)
                writer.writeheader(); writer.writerow(summary)
            rows = []
            for order, (family, mode, target, strength, delta) in enumerate([
                ("frozen_primary", "neuronpedia_global", "", -1.7, 1.0),
                ("frozen_primary", "neuronpedia_global", "", 1.8, 2.0),
                ("localized_control", "single_position", "question_mark", -1.7, 0.2),
                ("localized_control", "single_position", "meaningful_token_before_question_mark", -1.7, 0.1),
                ("adaptive_rescue", "neuronpedia_global", "", 1.8, 3.0),
            ], start=1):
                row = {field: "" for field in runner.INTERVENTION_FIELDS}
                row.update({
                    "item_id": "1", "analysis_family": family,
                    "is_primary_estimand": family == "frozen_primary", "attempt_order": order,
                    "intervention_mode": mode, "localized_target_selector": target,
                    "requested_strength": strength, "layer": 40, "token_id": 97817,
                    "baseline_oriented_margin": 1.0, "delta_oriented_margin": delta,
                    "delta_margin": delta, "flipped": strength > 0,
                    "flip_effect": "improved" if strength > 0 else "no_flip",
                    "intervened_valid": True,
                })
                rows.append(row)
            with (run_dir / "intervention_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=runner.INTERVENTION_FIELDS)
                writer.writeheader(); writer.writerows(rows)
            plotted = analysis.analyze(run_dir)
            self.assertTrue(plotted["primary global rescue and harm"])
            self.assertTrue(plotted["primary global margin movement"])
            self.assertTrue(plotted["global versus localized"])
            self.assertTrue((run_dir / "results.md").is_file())
            self.assertTrue((run_dir / "primary_stratified_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
