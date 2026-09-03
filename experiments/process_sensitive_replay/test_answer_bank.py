from __future__ import annotations

import types
import unittest

import torch

from experiments.process_sensitive_replay.answer_bank import discover_answer
from experiments.process_sensitive_replay.replay import (
    DISABLED_THINKING_SUFFIX,
    encode_chat,
    verify_thinking_disabled,
)


class TinyChatTokenizer:
    eos_token_id = 250
    im_end_id = 249
    endoftext_id = 250
    _special = {
        "<|im_start|>": 248,
        "<|im_end|>": im_end_id,
        "<|endoftext|>": endoftext_id,
    }

    def __init__(self) -> None:
        self.thinking_flags: list[bool] = []

    def apply_chat_template(
        self, messages, *, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    ):
        if tokenize:
            raise AssertionError("tests require rendered text before tokenization")
        self.thinking_flags.append(enable_thinking)
        rendered = ""
        for message in messages:
            rendered += f"<|im_start|>{message['role']}\n"
            if message["role"] == "assistant":
                rendered += "<think>\n\n</think>\n\n"
            rendered += message["content"] + "<|im_end|>\n"
        if add_generation_prompt:
            rendered += DISABLED_THINKING_SUFFIX
        return rendered

    def _encode(self, text: str) -> list[int]:
        result: list[int] = []
        index = 0
        specials = sorted(self._special, key=len, reverse=True)
        while index < len(text):
            special = next((value for value in specials if text.startswith(value, index)), None)
            if special is not None:
                result.append(self._special[special])
                index += len(special)
            else:
                result.append(ord(text[index]) + 1)
                index += 1
        return result

    def __call__(self, text, **kwargs):
        ids = self._encode(text)
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, token_ids, **_kwargs):
        reverse = {value: key for key, value in self._special.items()}
        return "".join(
            reverse[int(token_id)] if int(token_id) in reverse else chr(int(token_id) - 1)
            for token_id in token_ids
        )


class TinyGenerationAdapter:
    def __init__(self, response: list[int], prefix_length: int) -> None:
        self.response = response
        self.prefix_length = prefix_length
        self.hf_model = types.SimpleNamespace(
            generation_config=types.SimpleNamespace(eos_token_id=[249, 250])
        )

    @staticmethod
    def new_cache() -> list[int]:
        return []

    @staticmethod
    def cache_length(cache: list[int]) -> int:
        return len(cache)

    def step(self, token_id, cache, *, expected_position):
        if len(cache) != expected_position:
            raise AssertionError("position mismatch")
        cache.append(int(token_id))
        response_index = len(cache) - self.prefix_length
        next_token = self.response[response_index] if response_index >= 0 else 1
        logits = torch.full((512,), -20.0)
        logits[next_token] = 20.0
        return logits, {}


class AnswerBankTests(unittest.TestCase):
    row = {
        "item_id": "66",
        "item_type": "prospective",
        "prompt": "unused",
        "detail": "Q?",
        "answer_key": "AB",
    }

    def adapter(self, tokenizer: TinyChatTokenizer, response: list[int]) -> TinyGenerationAdapter:
        _, prefix_ids = encode_chat(
            tokenizer,
            [{"role": "user", "content": "Q?"}],
            add_generation_prompt=True,
        )
        return TinyGenerationAdapter(response, len(prefix_ids))

    def test_generated_content_is_exact_and_only_terminal_is_canonicalized(self) -> None:
        tokenizer = TinyChatTokenizer()
        content = tokenizer("AB", add_special_tokens=False)["input_ids"]
        adapter = self.adapter(tokenizer, [*content, tokenizer.endoftext_id])
        record = discover_answer(adapter, tokenizer, self.row, max_answer_tokens=256)

        self.assertFalse(record["invalid"])
        self.assertEqual(record["answer_token_ids"], content)
        self.assertEqual(record["generated_turn_end_token_ids"], [tokenizer.endoftext_id])
        self.assertEqual(record["canonical_turn_end_token_ids"], [tokenizer.im_end_id])
        self.assertEqual(record["post_answer_token_ids"][-1], tokenizer.im_end_id)
        self.assertFalse(record["generation"]["terminal_was_already_canonical"])

    def test_cap_without_termination_is_invalid_and_not_canonicalized(self) -> None:
        tokenizer = TinyChatTokenizer()
        content = tokenizer("ABC", add_special_tokens=False)["input_ids"]
        adapter = self.adapter(tokenizer, content)
        record = discover_answer(adapter, tokenizer, self.row, max_answer_tokens=2)

        self.assertTrue(record["invalid"])
        self.assertTrue(
            record["invalid_reasons"]["reached_token_cap_without_valid_turn_termination"]
        )
        self.assertEqual(record["answer_token_ids"], content[:2])
        self.assertEqual(record["generated_turn_end_token_ids"], [])
        self.assertEqual(record["canonical_turn_end_token_ids"], [])

    def test_three_family_render_checks_keep_thinking_disabled(self) -> None:
        tokenizer = TinyChatTokenizer()
        results = [
            verify_thinking_disabled(
                tokenizer, [{"role": "user", "content": f"question {index}"}]
            )
            for index in range(3)
        ]
        self.assertTrue(all(result["closed_empty_thinking_block"] for result in results))
        self.assertTrue(all(flag is False for flag in tokenizer.thinking_flags))


if __name__ == "__main__":
    unittest.main()
