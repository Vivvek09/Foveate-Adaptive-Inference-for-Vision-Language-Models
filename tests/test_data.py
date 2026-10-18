from foveate.data import Sample, load_samples, pilot_subset, shard
from foveate.data.frames import FrameStore, image_stats, sample_indices
from foveate.data.corrupt import apply_corruption


def _samples(n_tasks=4, per=50):
    return [Sample(qid=f"t{t}/{i}", dataset="x", task=f"t{t}", question="q", answer="A",
                   options=["a", "b"]) for t in range(n_tasks) for i in range(per)]


def test_pilot_is_30pct_stratified_and_stable():
    s = _samples()
    p1, p2 = pilot_subset(s, 0.3, seed=0), pilot_subset(list(reversed(s)), 0.3, seed=0)
    assert [x.qid for x in p1] == [x.qid for x in p2]
    assert len(p1) == 60
    assert {x.task for x in p1} == {"t0", "t1", "t2", "t3"}
    assert all(sum(x.task == t for x in p1) == 15 for t in ["t0", "t1", "t2", "t3"])
    assert [x.qid for x in pilot_subset(s, 0.3, seed=1)] != [x.qid for x in p1]


def test_shards_partition():
    s = _samples()
    parts = [shard(s, i, 2) for i in range(2)]
    assert sorted(x.qid for p in parts for x in p) == sorted(x.qid for x in s)


def test_sample_indices():
    assert sample_indices(0, 99, 4) == [12, 37, 62, 87]
    assert sample_indices(0, 2, 6) == [0, 0, 1, 1, 2, 2]
    assert sample_indices(10, 19, 2) == [12, 17]


def test_synthetic_load_and_frames(tmp_env):
    s = load_samples("synthetic", "full")
    assert len(s) == 24 and all(x.exists() for x in s)
    fb = FrameStore(None).load(s[0], 8, 448)
    assert len(fb.frames) == 8 and fb.height % 28 == 0 and fb.height * fb.width <= 448 * 448
    assert fb.meta["frames_indices"] == sample_indices(0, 31, 8)
    assert set(fb.stats) == {"brightness", "contrast", "sharpness", "saturation"}


def test_corruptions_change_stats(tmp_env):
    s = load_samples("synthetic", "full")[0]
    fb = FrameStore(None).load(s, 4, 448)
    base = image_stats(fb.frames)
    fog = image_stats([apply_corruption(f, "fog:4", seed=1) for f in fb.frames])
    night = image_stats([apply_corruption(f, "night:4", seed=1) for f in fb.frames])
    blur = image_stats([apply_corruption(f, "blur:4", seed=1) for f in fb.frames])
    assert fog["contrast"] < base["contrast"] and fog["brightness"] > base["brightness"]
    assert night["brightness"] < base["brightness"]
    assert blur["sharpness"] < base["sharpness"]
    a = apply_corruption(fb.frames[0], "fog:3", seed=7)
    b = apply_corruption(fb.frames[0], "fog:3", seed=7)
    assert a.tobytes() == b.tobytes()  # deterministic
