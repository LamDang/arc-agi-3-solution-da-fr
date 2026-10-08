"""Exercise the real HF multimodal processor with an offline synthetic tokenizer.

This verifies API wiring and image expansion; it does not replace the pinned
production tokenizer/template + real sol payload preflight.
"""
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3VLProcessor
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from data import encode, load_processor
from test_data import sample

TEMPLATE = """{% for m in messages %}{{ '<|im_start|>' + m.role + '\n' }}{% if m.role == 'assistant' %}{{ '<think>\n' + m.get('reasoning_content', '').strip('\n') + '\n</think>\n' }}{{ m.get('content', '') }}{% for c in m.get('tool_calls', []) %}{{ '<tool_call><function=' + c.function.name + '><parameter=code>' + c.function.arguments.code + '</parameter></function></tool_call>' }}{% endfor %}{% elif m.content is string %}{{ m.content }}{% else %}{% for p in m.content %}{% if p.type == 'image' %}{{ '<|vision_start|><|image_pad|><|vision_end|>' }}{% else %}{{ p.text }}{% endif %}{% endfor %}{% endif %}{{ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n' }}{% endif %}"""


def test_real_processor_image_expansion_roundtrip_and_reload(tmp_path):
    special = ['<|image_pad|>', '<|video_pad|>', '<|vision_start|>', '<|vision_end|>', '<|im_start|>', '<|im_end|>']
    alphabet = ['[UNK]'] + [chr(i) for i in range(9, 127)] + ['→'] + special
    vocab = {c: i for i, c in enumerate(alphabet)}
    raw = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    raw.pre_tokenizer = pre_tokenizers.Split('', behavior='isolated')
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', additional_special_tokens=special)
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessorPil(patch_size=16, temporal_patch_size=2, merge_size=2,
                                                size={"shortest_edge": 4096, "longest_edge": 4096}),
        tokenizer=tokenizer, video_processor=Qwen3VLVideoProcessor(), chat_template=TEMPLATE)
    enc, annotation = encode(processor, sample())
    assert enc["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert int((enc["input_ids"] == processor.image_token_id).sum()) == 4
    assert annotation["category_counts"]["tool_code"] > 0
    processor.save_pretrained(tmp_path)
    restored = load_processor(tmp_path)
    _, after = encode(restored, sample())
    assert annotation == after
