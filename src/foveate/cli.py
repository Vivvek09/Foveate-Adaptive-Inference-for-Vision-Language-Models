"""Command-line entry point: `foveate <command> ...`.

Every command that touches data takes --subset pilot (fixed-seed 30%, the
default) or --subset full (also spelled --pilot / --full).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from . import config as C
from .utils import capture_env, setup_logging

log = logging.getLogger("foveate")


def _subset_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--subset", choices=["pilot", "full"], default="pilot",
                   help="pilot = fixed-seed 30%% stratified subset (default); full = everything")
    g.add_argument("--pilot", dest="subset", action="store_const", const="pilot")
    g.add_argument("--full", dest="subset", action="store_const", const="full")
    p.add_argument("--tasks", default=None,
                   help="MVBench tasks: all (default), lite (skip the 11 GB perception zip), "
                        "or a comma list")
    p.add_argument("--limit", type=int, default=None, help="cap questions (smoke tests)")


def _samples(args, cfg):
    from .data import load_samples

    tasks = args.tasks.split(",") if args.tasks and "," in args.tasks else args.tasks
    return load_samples(args.dataset, args.subset, fraction=cfg["pilot_fraction"],
                        seed=cfg["seed"], tasks=tasks, limit=args.limit)


def _store_res(cfg, dataset: str) -> int:
    return max(C.VisualConfig.parse(s, m).res for m, specs in cfg["grid"][dataset].items() for s in specs)


# -- commands ----------------------------------------------------------------


def cmd_env(args, cfg):
    print(json.dumps(capture_env(), indent=1))


def cmd_download(args, cfg):
    root = C.data_root()
    if args.dataset == "mvbench":
        from .data import mvbench

        mvbench.download(root, args.tasks, keep_zips=args.keep_zips)
    elif args.dataset == "lingoqa":
        from .data import lingoqa

        lingoqa.download(root)
    elif args.dataset == "synthetic":
        from .data import synthetic

        synthetic.generate(root)
    log.info("data is under %s", root)


def cmd_profile(args, cfg):
    from .backends.hf_backend import HFBackend
    from .data import load_samples
    from .profile import run_profile, summarize

    pcfg = cfg["profile"]
    mid = C.model_id(cfg, args.model)
    samples = load_samples(args.source, "full", tasks=args.tasks, limit=1)
    backend = HFBackend(mid, dtype=args.dtype, attn_implementation=args.attn)
    out = C.results_root() / "profile" / f"{args.model}_{args.source}.jsonl"
    rows = run_profile(
        backend, mid, samples,
        frames=args.frames or pcfg["frames"], resolutions=args.res or pcfg["res"],
        out_tokens=args.out_tokens or pcfg["out_tokens"],
        decode_sweep=pcfg["decode_sweep_out_tokens"],
        warmup=args.warmup if args.warmup is not None else pcfg["warmup"],
        repeats=args.repeats or pcfg["repeats"], out_path=out,
        cache_dir=C.cache_root() / "profile",
    )
    summ = summarize(rows, mid, args.peak_tflops, args.mem_bw)
    out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))
    log.info("wrote %s and %s", out, out.with_suffix(".summary.json"))


def _spawn_shards(args, n_gpus: int) -> None:
    """One child process per GPU, each running a deterministic shard of the questions."""
    argv = [a for a in sys.argv[1:]]
    if "--gpus" in argv:
        i = argv.index("--gpus")
        del argv[i:i + 2]
    procs = []
    for g in range(n_gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        cmd = [sys.executable, "-m", "foveate.cli", *argv, "--shard", str(g), "--num-shards", str(n_gpus)]
        log.info("GPU %d: %s", g, " ".join(cmd))
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"shard processes exited with {codes}")


def cmd_grid(args, cfg):
    if args.gpus > 1 and args.shard is None:
        return _spawn_shards(args, args.gpus)
    from .grid import HFGridAdapter, run_grid

    samples = _samples(args, cfg)
    configs = ([C.VisualConfig.parse(s, args.model) for s in args.configs]
               if args.configs else C.grid_configs(cfg, args.dataset, args.model))
    configs = [c for c in configs if c.model == args.model]
    mid = C.model_id(cfg, args.model)
    backend = None
    if args.backend in ("vllm", "auto"):
        try:
            from .backends.vllm_backend import VLLMBackend

            backend = VLLMBackend(
                mid, quantization=C.model_quantization(cfg, args.model), dtype=args.dtype,
                max_res=max(c.res for c in configs), max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_mem, enforce_eager=args.enforce_eager,
                tensor_parallel_size=args.tp, max_num_seqs=args.max_num_seqs, seed=cfg["seed"],
            )
        except Exception as e:  # noqa: BLE001 - any vLLM start-up failure
            if args.backend == "vllm":
                raise
            log.warning("vLLM unavailable (%s: %s); falling back to the Transformers backend "
                        "(batch 1, slower)", type(e).__name__, str(e).splitlines()[0][:200])
            _free_gpu()
    if backend is None:
        from .backends.hf_backend import HFBackend

        backend = HFGridAdapter(HFBackend(mid, dtype=args.dtype, attn_implementation=args.attn))
    corruptions = args.corruption or [None]
    for corr in corruptions:
        run_grid(backend, samples, configs, dataset=args.dataset, subset=args.subset,
                 corruption=corr, shard_id=args.shard or 0, num_shards=args.num_shards,
                 batch_size=args.batch_size, probe=0 if corr else args.probe,
                 workers=args.workers, store_res=_store_res(cfg, args.dataset))


def _free_gpu():
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def cmd_shift(args, cfg):
    """Re-run this model's shift configs on fog / night / blur copies."""
    sh = cfg["shift"]
    args.configs = [s for s in sh["configs"][args.dataset] if s.startswith(args.model + ":")]
    if not args.configs:
        log.info("no shift configs for model %s", args.model)
        return
    args.corruption = args.corruption or [f"{c}:{sh['severity']}" for c in sh["corruptions"]]
    return cmd_grid(args, cfg)


def cmd_judge(args, cfg):
    from .grid import run_dir
    from .judge import judge_dir

    samples = {s.qid: s for s in _samples(args, cfg)}
    judge_dir(run_dir(args.dataset, args.subset), samples, device=args.device)


def cmd_replay(args, cfg):
    from .drift import shift_report
    from .grid import run_dir
    from .policy import build_table, load_costs, load_rows, replay

    rd = run_dir(args.dataset, args.subset)
    df = load_rows(rd)
    costs = load_costs(rd)
    model_ids = {k: v["id"] for k, v in cfg["models"].items()}
    t = build_table(df, costs, cost_metric=args.cost, model_ids=model_ids)
    ladder = args.ladder or cfg["ladder"][args.dataset]
    rep = replay(t, ladder, ref=args.ref, tol=args.tol, seed=cfg["seed"])
    out = rd / "replay"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rep, indent=1))
    policy = {"dataset": args.dataset, "ladder": rep["ladder"], "tau": rep["tau"],
              "rules": rep["rules"], "costs": rep["costs"], "cost_metric": rep["cost_metric"]}
    (out / "policy.json").write_text(json.dumps(policy, indent=1))
    for p in rep["policies"]:
        log.info("%-34s acc %.3f [%.3f, %.3f]  cost %.1f", p["policy"], p["accuracy"],
                 p["acc_lo"], p["acc_hi"], p["mean_cost"])
    if (df["corruption"] != "clean").any():
        srep = shift_report(df, costs, rep["ladder"], rep["tau"], model_ids)
        (out / "shift.json").write_text(json.dumps(srep, indent=1))
        for r in srep["per_corruption"]:
            log.info("%s: escalation %.2f -> %.2f, cascade acc %.3f -> %.3f", r["corruption"],
                     r["clean_escalation"], r["shifted_escalation"], r["clean_cascade_acc"],
                     r["shifted_cascade_acc"])
    log.info("wrote %s", out)


def cmd_serve(args, cfg):
    import uvicorn

    from .serve.gateway import Gateway, build_app

    samples = {s.qid: s for s in _samples(args, cfg)}
    pol = json.loads(Path(args.policy_file).read_text()) if args.policy_file else {}
    ladder = [C.VisualConfig.parse(s) for s in (pol.get("ladder") or cfg["ladder"][args.dataset])]
    backends = dict(b.split("=", 1) for b in args.backends)
    gw = Gateway(samples, backends, {k: v["id"] for k, v in cfg["models"].items()}, ladder,
                 tau=args.tau if args.tau is not None else pol.get("tau", 0.6),
                 telemetry=C.results_root() / "serving" / "telemetry.jsonl",
                 store_res=_store_res(cfg, args.dataset))
    uvicorn.run(build_app(gw), host=args.host, port=args.port, log_level="warning")


def cmd_loadtest(args, cfg):
    import asyncio

    from .serve.loadgen import make_trace, run_trace

    samples = _samples(args, cfg)
    trace = make_trace(samples, args.n, args.rate, args.repeat, seed=cfg["seed"])
    tag = args.tag or f"{args.policy.replace(':', '-')}_r{args.rate}"
    out = C.results_root() / "serving" / f"{tag}.jsonl"
    summ = asyncio.run(run_trace(args.url, trace, args.policy, out))
    summ.update({"rate": args.rate, "repeat": args.repeat})
    out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def cmd_report(args, cfg):
    from .report import build_report

    print(build_report(C.results_root()))


# -- parser ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="foveate", description=__doc__)
    ap.add_argument("--config", default=None, help="YAML config (default: packaged default.yaml)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env", help="print GPU / library versions")

    p = sub.add_parser("download", help="fetch a dataset into $FOVEATE_DATA (default ./data)")
    p.add_argument("dataset", choices=["mvbench", "lingoqa", "synthetic"])
    p.add_argument("--tasks", default="all")
    p.add_argument("--keep-zips", action="store_true")

    p = sub.add_parser("profile", help="stage 1: batch-1 stage timing sweep (Transformers)")
    p.add_argument("--model", default="3b")
    p.add_argument("--source", default="synthetic", choices=["synthetic", "mvbench", "lingoqa"])
    p.add_argument("--tasks", default=None)
    p.add_argument("--frames", type=int, nargs="*")
    p.add_argument("--res", type=int, nargs="*")
    p.add_argument("--out-tokens", type=int)
    p.add_argument("--repeats", type=int)
    p.add_argument("--warmup", type=int)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--peak-tflops", type=float)
    p.add_argument("--mem-bw", type=float)

    def grid_args(p):
        p.add_argument("--dataset", default="mvbench", choices=["mvbench", "lingoqa", "synthetic"])
        _subset_args(p)
        p.add_argument("--model", default="3b")
        p.add_argument("--configs", nargs="*", help="e.g. 4x224 8x448 (default: from config)")
        p.add_argument("--corruption", nargs="*", help="e.g. fog:3 night:3 blur:3")
        p.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto",
                       help="auto = vLLM, falling back to Transformers if vLLM fails to start")
        p.add_argument("--gpus", type=int, default=1, help="data-parallel: one process per GPU")
        p.add_argument("--shard", type=int, default=None)
        p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--probe", type=int, default=16, help="batch-1 latency probe size")
        p.add_argument("--workers", type=int, default=4, help="frame-loading threads")
        p.add_argument("--dtype", default="auto")
        p.add_argument("--attn", default="sdpa")
        p.add_argument("--gpu-mem", type=float, default=0.90)
        p.add_argument("--max-model-len", type=int, default=8192)
        p.add_argument("--max-num-seqs", type=int, default=None)
        p.add_argument("--tp", type=int, default=1, help="tensor parallel size (e.g. fp32 on 2xT4)")
        p.add_argument("--enforce-eager", action="store_true",
                       help="skip CUDA graphs (faster start-up, less memory)")

    grid_args(sub.add_parser("grid", help="stage 2: run configs over every question (vLLM)"))
    grid_args(sub.add_parser("shift", help="stage 4: rerun shift configs on corrupted frames"))

    p = sub.add_parser("judge", help="score LingoQA answers with Lingo-Judge")
    p.add_argument("--dataset", default="lingoqa")
    _subset_args(p)
    p.add_argument("--device", default=0)

    p = sub.add_parser("replay", help="stages 3-4: evaluate policies + shift from the grid cache")
    p.add_argument("--dataset", default="mvbench")
    _subset_args(p)
    p.add_argument("--ladder", nargs="*")
    p.add_argument("--ref", default=None, help="reference config (default: most accurate)")
    p.add_argument("--cost", default="latency_b1_ms", choices=["latency_b1_ms", "amortized_ms"])
    p.add_argument("--tol", type=float, default=0.0, help="accuracy the cascade may give up")

    p = sub.add_parser("serve", help="stage 5: policy gateway in front of vllm serve")
    p.add_argument("--dataset", default="mvbench")
    _subset_args(p)
    p.add_argument("--backends", nargs="+", default=["3b=http://localhost:8001", "7b=http://localhost:8002"])
    p.add_argument("--policy-file", default=None)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)

    p = sub.add_parser("loadtest", help="stage 5: Poisson load against the gateway")
    p.add_argument("--dataset", default="mvbench")
    _subset_args(p)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--policy", default="cascade", help="cascade or static:3b:16x448")
    p.add_argument("--rate", type=float, default=1.0, help="requests per second")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--repeat", type=float, default=0.3, help="share of repeated-clip requests")
    p.add_argument("--tag", default=None)

    sub.add_parser("report", help="figures + results/REPORT.md")
    return ap


COMMANDS = {
    "env": cmd_env, "download": cmd_download, "profile": cmd_profile, "grid": cmd_grid,
    "shift": cmd_shift, "judge": cmd_judge, "replay": cmd_replay, "serve": cmd_serve,
    "loadtest": cmd_loadtest, "report": cmd_report,
}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    cfg = C.load_config(args.config)
    COMMANDS[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
