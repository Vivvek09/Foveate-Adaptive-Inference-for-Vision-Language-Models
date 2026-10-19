"""Real Transformers code path on a tiny random Qwen2.5-VL (CPU, no download)."""

import pytest

pytest.importorskip("torchvision")
pytest.importorskip("transformers")

from foveate.data import load_samples
from foveate.data.frames import FrameStore
from foveate.tokens import visual_tokens


def test_profile_and_predict(tmp_env):
    from foveate.backends.hf_backend import HFBackend
    from foveate.profile import run_profile, summarize

    import tiny_qwen

    model, proc = tiny_qwen.build()
    b = HFBackend("tiny", model=model, processor=proc, device="cpu", dtype="float32")
    s = load_samples("synthetic", "full")[0]
    st = FrameStore(None)
    for n, res in [(4, 224), (8, 448), (5, 224)]:
        fb = st.load(s, n, res)
        t = b.profile_one(s, fb, out_tokens=3)
        assert t.visual_tokens == visual_tokens(fb.height, fb.width, n)
        assert t.out_tokens == 3 and t.vision_ms > 0 and t.decode_ms > 0
    row = b.predict(s, st.load(s, 4, 224))
    assert row["pred"] in "ABCD" and 0 < row["conf"] <= 1
    rows = run_profile(b, "tiny", [s], [2, 4], [224], 2, [1, 4], 1, 2,
                       tmp_env / "p.jsonl")
    summ = summarize(rows, "tiny")
    assert {r["sweep"] for r in summ} == {"cold", "visual", "decode"}
