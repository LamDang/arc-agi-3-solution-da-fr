"""Offline format fixtures; real pinned processor still needs its data preflight."""
import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from common import LABELS
from data import encode, hf_messages, labels_from_offsets


class CharTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i+1) for i in range(len(text))]}


class FixtureProcessor:
    """Transparent character IDs and synthetic image expansion expose shifts."""
    tokenizer = CharTokenizer()

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False, **kwargs):
        assert "strict" not in str(tools)
        text, images = "", 0
        for m in messages:
            text += f"<{m['role']}>"
            content = m.get("content", "") or ""
            if isinstance(content, list):
                for part in content:
                    if part["type"] == "image":
                        text += "\ufffc"
                        images += 1
                    else:
                        text += part["text"]
            elif m["role"] == "assistant":
                text += "<think>\n" + m.get("reasoning_content", "").strip("\n") + "\n</think>\n"
                text += content
                for call in m.get("tool_calls", []):
                    f = call["function"]
                    text += f"<tool_call><function={f['name']}><parameter=code>{f['arguments']['code']}</parameter></function></tool_call>"
            else:
                text += content
            text += "<end>"
        if add_generation_prompt:
            text += "<assistant><think>\n"
        if not tokenize:
            return text
        ids = [x for c in text for x in ([248056]*4 if c == "\ufffc" else [ord(c)])]
        out = {"input_ids": torch.tensor([ids]), "mm_token_type_ids": torch.tensor([[int(x==248056) for x in ids]])}
        if images:
            out.update(image_grid_thw=torch.tensor([[1, 4, 4]]*images), pixel_values=torch.zeros(images*16, 12))
        return out


def sample():
    stream = io.BytesIO()
    Image.new("RGB", (8, 8)).save(stream, format="PNG")
    image = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
    return dict(messages=[{"role": "system", "content": "instructions"},
                           {"role": "assistant", "reasoning_content": "old reasoning", "content": "old text"},
                           {"role": "user", "content": [{"type": "text", "text": "level 2"},
                                                         {"type": "image_url", "image_url": {"url": image}}]},
                           {"role": "assistant", "reasoning_content": "\nMove → goal\n", "content": "",
                            "tool_calls": [{"function": {"name": "python", "arguments": {
                                "code": "print('<think><tool_call>literal</tool_call>')"}}}]}],
                tools=[{"type": "function", "function": {"name": "python", "parameters": {}}}],
                chat_template_kwargs={"preserve_thinking": True})


def test_final_only_images_unicode_and_literal_tags():
    s, processor = sample(), FixtureProcessor()
    enc, meta = encode(processor, s)
    assert meta["images"] == 1
    assert meta["positions"][0] == meta["prompt_tokens"]
    assert meta["target_ids"] == enc["input_ids"][0, meta["prompt_tokens"]:].tolist()
    assert "old reasoning" not in meta["target_text"]
    assert sum(meta["category_counts"].values()) == meta["target_tokens"]
    assert meta["category_counts"]["tool_code"] == len(s["messages"][-1]["tool_calls"][0]["function"]["arguments"]["code"])
    assert meta["category_counts"]["thinking"] == len("Move → goal")
    assert meta["category_counts"]["tool_format"] > 0


def test_plain_final_text_and_empty_categories():
    s = sample()
    s["messages"][-1] = dict(role="assistant", reasoning_content="Done", content="Finished.")
    _, meta = encode(FixtureProcessor(), s)
    assert meta["category_counts"]["assistant_text"] == len("Finished.")
    assert meta["category_counts"]["tool_code"] == 0


def test_template_rationale_trimming_preserves_code_whitespace():
    s = sample()
    code = "\n  print('preserve leading space')\n\n"
    s["messages"][-1]["tool_calls"][0]["function"]["arguments"]["code"] = code
    _, meta = encode(FixtureProcessor(), s)
    assert meta["category_counts"]["thinking"] == len("Move → goal")
    assert meta["category_counts"]["tool_code"] == len(code)


def test_strict_arguments_and_boundary_tokens():
    s = sample()
    s["messages"][-1]["tool_calls"][0]["function"]["arguments"] = '{"code":"x"}'
    with pytest.raises(ValueError, match="mapping"):
        hf_messages(s["messages"])
    labels = np.array([0, 0, 1, 1], dtype=np.uint8)
    got = labels_from_offsets([(0, 2), (1, 3), (0, 0)], labels)
    assert got.tolist() == [0, LABELS.index("boundary"), LABELS.index("boundary")]


def test_token_prefix_mismatch_is_fatal():
    class BadProcessor(FixtureProcessor):
        def apply_chat_template(self, *args, **kwargs):
            result = super().apply_chat_template(*args, **kwargs)
            if kwargs.get("tokenize") and kwargs.get("add_generation_prompt"):
                result["input_ids"][0, 0] += 1
            return result
    with pytest.raises(ValueError, match="prefix"):
        encode(BadProcessor(), sample())
