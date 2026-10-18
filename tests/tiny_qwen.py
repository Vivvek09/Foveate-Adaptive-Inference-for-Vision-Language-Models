"""A randomly initialised, tiny Qwen2.5-VL built offline (no Hub download).

Lets CPU tests exercise the real Transformers code path: processor, chat
template, video placeholder expansion, vision hooks and generate().
"""

from __future__ import annotations

SPECIALS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
    "<|image_pad|>", "<|video_pad|>",
]

CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}{{ message['content'] }}{% else %}"
    "{% for c in message['content'] %}"
    "{% if c['type'] == 'video' %}<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif c['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
    "{% else %}{{ c['text'] }}{% endif %}{% endfor %}{% endif %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def build():
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2_5_VLConfig,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLProcessor,
    )
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

    chars = [chr(i) for i in range(32, 127)] + ["\n"]
    vocab = {t: i for i, t in enumerate(SPECIALS + chars + ["[UNK]"])}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>",
                                  pad_token="<|endoftext|>", unk_token="[UNK]")
    tok.add_special_tokens({"additional_special_tokens": SPECIALS})
    tok.chat_template = CHAT_TEMPLATE

    cfg = Qwen2_5_VLConfig(
        text_config=dict(
            vocab_size=len(vocab) + 8, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=4096,
            rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        ),
        vision_config=dict(
            depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
            fullatt_block_indexes=[1], window_size=56, patch_size=14,
            spatial_merge_size=2, temporal_patch_size=2,
        ),
        image_token_id=vocab["<|image_pad|>"],
        video_token_id=vocab["<|video_pad|>"],
        vision_start_token_id=vocab["<|vision_start|>"],
        vision_end_token_id=vocab["<|vision_end|>"],
    )
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(cfg).eval()
    proc = Qwen2_5_VLProcessor(
        image_processor=Qwen2VLImageProcessor(), tokenizer=tok,
        video_processor=Qwen2VLVideoProcessor(), chat_template=CHAT_TEMPLATE,
    )
    return model, proc
