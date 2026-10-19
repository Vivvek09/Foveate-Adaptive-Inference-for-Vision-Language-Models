import json

import pytest

from foveate import cli
from foveate.config import grid_configs, ladder, load_config
from foveate.data import load_samples
from foveate.grid import run_grid

from fake_backend import FakeBackend


def test_help_for_every_command(capsys):
    for cmd in cli.COMMANDS:
        with pytest.raises(SystemExit) as e:
            cli.main([cmd, "--help"])
        assert e.value.code == 0


def test_default_config_is_consistent():
    cfg = load_config()
    for ds in cfg["ladder"]:
        grid = {c.name for m in cfg["grid"][ds] for c in grid_configs(cfg, ds, m)}
        assert {c.name for c in ladder(cfg, ds)} <= grid, ds
        assert set(cfg["shift"]["configs"][ds]) <= grid, ds


def test_replay_and_report_cli(tmp_env):
    cfg = load_config()
    samples = load_samples("synthetic", "full")
    for m in ("3b", "7b"):
        run_grid(FakeBackend(), samples, grid_configs(cfg, "synthetic", m), dataset="synthetic",
                 subset="full", batch_size=8, probe=2)
        run_grid(FakeBackend(), samples, [c for c in grid_configs(cfg, "synthetic", m)
                                          if c.name in cfg["shift"]["configs"]["synthetic"]],
                 dataset="synthetic", subset="full", corruption="night:3", batch_size=8, probe=0)
    cli.main(["replay", "--dataset", "synthetic", "--full"])
    pol = json.loads((tmp_env / "results" / "synthetic_full" / "replay" / "policy.json").read_text())
    assert pol["ladder"] == cfg["ladder"]["synthetic"] and 0 <= pol["tau"] <= 1.01
    assert (tmp_env / "results" / "synthetic_full" / "replay" / "shift.json").exists()
    cli.main(["report"])
    assert (tmp_env / "results" / "REPORT.md").exists()
