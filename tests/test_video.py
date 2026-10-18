import numpy as np
import pytest

av = pytest.importorskip("av")

from foveate.data import Sample  # noqa: E402
from foveate.data.frames import FrameStore, decode_indices, probe_video  # noqa: E402


def _write_video(path, n=40, fps=10, h=120, w=160):
    with av.open(str(path), "w") as c:
        st = c.add_stream("mpeg4", rate=fps)
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        for i in range(n):
            img = np.full((h, w, 3), i * 6, dtype=np.uint8)  # brightness encodes the index
            for p in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in st.encode():
            c.mux(p)


def test_decode_and_cache(tmp_path):
    v = tmp_path / "clip.mp4"
    _write_video(v)
    info = probe_video(str(v))
    assert info.total_frames == 40 and abs(info.fps - 10) < 0.01
    fr = decode_indices(str(v), [0, 20, 39, 55])
    assert abs(fr[20].mean() - 120) < 8 and abs(fr[39].mean() - 234) < 8
    assert np.array_equal(fr[55], fr[39])  # past the end -> last frame
    s = Sample(qid="q", dataset="x", task="t", question="?", answer="A", options=["a", "b"],
               video=str(v), start=1.0, end=3.0)
    store = FrameStore(tmp_path / "cache", frame_counts=[2, 4])
    fb = store.load(s, 4, 224)
    assert fb.meta["total_num_frames"] == 21  # frames 10..30
    assert len(list((tmp_path / "cache").rglob("*.jpg"))) >= 4
    fb2 = store.load(s, 2, 224)  # served from cache
    assert len(fb2.frames) == 2
