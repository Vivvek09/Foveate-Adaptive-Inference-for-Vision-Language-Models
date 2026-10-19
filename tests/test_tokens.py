from foveate.config import VisualConfig
from foveate.tokens import (
    MODEL_SHAPES,
    kv_bytes_per_token,
    smart_resize,
    target_size,
    visual_tokens,
)


def test_smart_resize_multiple_of_28():
    for h, w in [(480, 640), (720, 1280), (100, 3000), (1080, 1920)]:
        rh, rw = smart_resize(h, w)
        assert rh % 28 == 0 and rw % 28 == 0


def test_target_size_budget():
    for res in (224, 336, 448, 672, 896):
        h, w = target_size(720, 1280, res)
        assert h * w <= res * res
        assert h * w >= 0.6 * res * res  # rounding down never loses too much


def test_square_token_counts_match_plan():
    assert visual_tokens(448, 448, 1, "images") == 256
    assert visual_tokens(672, 672, 1, "images") == 576
    assert visual_tokens(448, 448, 8, "video") == 1024
    assert visual_tokens(448, 448, 8, "images") == 2048
    assert visual_tokens(672, 672, 16, "video") == 4608
    assert visual_tokens(448, 448, 5, "video") == 3 * 256  # odd count padded


def test_kv_cache_bytes_3b():
    # 2 (K,V) * 36 layers * 2 KV heads * 128 dims * 2 bytes = 36,864 B/token
    assert kv_bytes_per_token(MODEL_SHAPES["Qwen/Qwen2.5-VL-3B-Instruct"]) == 36864


def test_config_parse():
    c = VisualConfig.parse("3b:8x448")
    assert (c.model, c.frames, c.res) == ("3b", 8, 448)
    assert VisualConfig.parse("16x224", default_model="7b").name == "7b:16x224"
