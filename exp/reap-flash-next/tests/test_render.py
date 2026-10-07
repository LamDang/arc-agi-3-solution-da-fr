"""Rendered request lengths equal the prompt_tokens SGLang logged.

Needs a real request log and the model's processor files:
  REAP_TEST_LOG        a Kaggle-run *_requests.jsonl[.xz] (served by SGLang)
  REAP_TEST_PROCESSOR  a directory with tokenizer.json, chat_template.jinja,
                       preprocessor_config.json, processor_config.json, ...
Skipped when either is missing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import render  # noqa: E402
import traces  # noqa: E402

LOG = os.environ.get("REAP_TEST_LOG")
PROCESSOR = os.environ.get("REAP_TEST_PROCESSOR")
pytestmark = pytest.mark.skipif(not (LOG and PROCESSOR and Path(LOG).exists() and Path(PROCESSOR).exists()),
                                reason="REAP_TEST_LOG / REAP_TEST_PROCESSOR not set")


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(PROCESSOR)


def test_every_request_renders_to_the_logged_length(processor):
    path = Path(LOG)
    requests = traces.scan_log(path)
    assert requests
    for request in requests:
        row = traces.load_request(path, request.line)
        enc = render.render(processor, row["messages"], row["tools"], add_generation_prompt=True)
        assert enc["input_ids"].shape[1] == request.usage["prompt_tokens"], request.line


def test_sample_categories(processor):
    path = Path(LOG)
    samples, _ = traces.samples_from_log(path, "test", "game", 0)
    sample = samples[0]
    messages, tools, _ = traces.load_messages(sample)
    enc = render.render(processor, messages, tools, add_generation_prompt=False)
    image_token = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    cats = render.token_categories(enc["input_ids"], processor.tokenizer, image_token)
    n_images = sum(isinstance(m.get("content"), list) and sum(p.get("type") == "image_url" for p in m["content"])
                   for m in messages)
    image_tokens = int(enc["image_grid_thw"].prod(-1).sum()) // 4
    assert int((cats == render.IMAGE).sum()) == image_tokens and enc["image_grid_thw"].shape[0] == n_images
    assert int((cats == render.GENERATED).sum()) > 0
    # the sequence is the logged prompt plus the reply
    assert enc["input_ids"].shape[1] > sample.logged_prompt_tokens
    ids = enc["input_ids"][0]
    end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    generated_ends = int(((ids == end) & (cats[0] == render.GENERATED)).sum())
    n_assistant = sum(m["role"] == "assistant" for m in messages)
    assert generated_ends == n_assistant
    assert torch.all(cats[0, :10] == render.CONTEXT)
