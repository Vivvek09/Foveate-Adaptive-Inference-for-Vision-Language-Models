"""Transformers backend: batch-1 inference with per-stage GPU timing.

vLLM is faster but does not expose per-request vision-encoder time. Here we
own the model object, so we can bracket the vision tower with CUDA events
(forward pre/post hooks) and timestamp every generation step with a logits
processor, which runs once per generated token, right after each forward pass.

    start ─ H2D ─┬─ [vision encoder] ─┬─ LLM prefill ─ step1 ─ decode ... ─ stepN
                 v0                    v1               (TTFT on GPU)

Times come from CUDA events, which are recorded in the GPU stream, so they
measure GPU execution, not Python dispatch. We synchronize once per request,
after generation, never inside the timed region.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

from ..data import Sample
from ..data.frames import FrameBatch
from ..prompts import chat_messages, mc_summary, option_letters, softmax_over_options

log = logging.getLogger(__name__)

# Frames arrive pre-resized to multiples of 28 px, so the processor must not
# resize again: allow everything from one token to a huge frame, no re-sampling.
NO_RESIZE_VIDEO_KWARGS = {
    "size": {"shortest_edge": 28 * 28, "longest_edge": 28 * 28 * 16384},
    "do_sample_frames": False,
    "cap_pixels_per_frame": False,
}


@dataclass
class StageTiming:
    preprocess_ms: float
    h2d_ms: float
    vision_ms: float
    prefill_ms: float
    ttft_gpu_ms: float
    decode_ms: float
    tpot_ms: float
    gpu_total_ms: float
    out_tokens: int
    prompt_tokens: int
    visual_tokens: int
    peak_mem_gb: float


class HFBackend:
    kind = "hf"

    def __init__(
        self,
        model_id: str,
        dtype: str = "auto",
        attn_implementation: str = "sdpa",
        device: str = "auto",
        model=None,
        processor=None,
    ):
        import torch
        from transformers import AutoProcessor

        from ..utils import resolve_dtype

        t_load = time.perf_counter()
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_id = model_id
        self.attn_implementation = attn_implementation
        self.dtype_name = resolve_dtype(dtype)  # fp16 on T4; explicit bf16 on T4 raises
        torch_dtype = getattr(torch, self.dtype_name)
        if model is None:
            from transformers import Qwen2_5_VLForConditionalGeneration

            kw = dict(attn_implementation=attn_implementation, device_map=device)
            try:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, dtype=torch_dtype, **kw)
            except TypeError:  # transformers < 4.56 calls it torch_dtype
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, torch_dtype=torch_dtype, **kw)
        self.model = model.eval()
        self.processor = processor or AutoProcessor.from_pretrained(model_id)
        self.tok = self.processor.tokenizer
        self.video_token_id = self.processor.video_token_id
        inner = getattr(self.model, "model", self.model)
        self.visual = getattr(inner, "visual", None) or self.model.visual
        self._events: dict[str, list] = {"v0": [], "v1": [], "steps": []}
        self.visual.register_forward_pre_hook(lambda m, a: self._mark("v0"))
        self.visual.register_forward_hook(lambda m, a, o: self._mark("v1"))
        self.letter_ids = {}
        for L in "ABCDEFGH":
            ids = self.tok.encode(L, add_special_tokens=False)
            if len(ids) == 1:
                self.letter_ids[L] = ids[0]
        self.load_s = time.perf_counter() - t_load

    def info(self) -> dict:
        import transformers

        return {"kind": self.kind, "model_id": self.model_id, "dtype": self.dtype_name,
                "attn_implementation": self.attn_implementation,
                "transformers": transformers.__version__, "load_s": self.load_s}

    # -- timing helpers ---------------------------------------------------
    @property
    def _cuda(self) -> bool:
        return self.device.startswith("cuda") and self.torch.cuda.is_available()

    def _event(self):
        if self._cuda:
            e = self.torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()  # CPU fallback (tests)

    @staticmethod
    def _ms(a, b) -> float:
        if isinstance(a, float):
            return (b - a) * 1e3
        return a.elapsed_time(b)

    def _mark(self, key: str) -> None:
        self._events[key].append(self._event())

    def _step_processor(self):
        backend = self

        class _StepTimer:  # a LogitsProcessor that only timestamps
            def __call__(self, input_ids, scores):
                backend._mark("steps")
                return scores

        return _StepTimer()

    # -- inputs -------------------------------------------------------------
    def build_inputs(self, s: Sample, fb: FrameBatch):
        text = self.processor.apply_chat_template(
            chat_messages(s), tokenize=False, add_generation_prompt=True
        )
        try:
            return self.processor(
                text=[text], videos=[fb.frames], video_metadata=[fb.meta],
                videos_kwargs=NO_RESIZE_VIDEO_KWARGS, return_tensors="pt",
            )
        except (TypeError, ValueError):
            # Older transformers: no video_metadata / size dict; pass pixel bounds and fps.
            fps = len(fb.frames) / fb.meta["duration"] if fb.meta.get("duration") else 2.0
            return self.processor(
                text=[text], videos=[fb.frames], fps=fps, return_tensors="pt",
                min_pixels=28 * 28, max_pixels=28 * 28 * 16384,
            )

    # -- profiling ----------------------------------------------------------
    def profile_one(self, s: Sample, fb: FrameBatch, out_tokens: int = 32) -> StageTiming:
        torch = self.torch
        for k in self._events:
            self._events[k].clear()
        if self._cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        inputs = self.build_inputs(s, fb)
        pre_ms = (time.perf_counter() - t0) * 1e3
        if self._cuda:
            torch.cuda.synchronize()
        e0 = self._event()
        inputs = inputs.to(self.model.device)
        e_h2d = self._event()
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=out_tokens, min_new_tokens=out_tokens,
                do_sample=False, logits_processor=[self._step_processor()],
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        e_end = self._event()
        if self._cuda:
            torch.cuda.synchronize()
        steps, v0, v1 = self._events["steps"], self._events["v0"], self._events["v1"]
        vision = self._ms(v0[0], v1[-1]) if v0 and v1 else float("nan")
        ttft_gpu = self._ms(e0, steps[0])
        h2d = self._ms(e0, e_h2d)
        decode = self._ms(steps[0], steps[-1]) if len(steps) > 1 else 0.0
        n_out = int(out.shape[1] - inputs["input_ids"].shape[1])
        ids = inputs["input_ids"]
        return StageTiming(
            preprocess_ms=pre_ms,
            h2d_ms=h2d,
            vision_ms=vision,
            prefill_ms=ttft_gpu - h2d - (0 if math.isnan(vision) else vision),
            ttft_gpu_ms=ttft_gpu,
            decode_ms=decode,
            tpot_ms=decode / (len(steps) - 1) if len(steps) > 1 else float("nan"),
            gpu_total_ms=self._ms(e0, e_end),
            out_tokens=n_out,
            prompt_tokens=int(ids.shape[1]),
            visual_tokens=int((ids == self.video_token_id).sum()),
            peak_mem_gb=(torch.cuda.max_memory_allocated() / 2**30) if self._cuda else 0.0,
        )

    # -- prediction (fallback grid backend; vLLM is the fast path) ----------
    def predict(self, s: Sample, fb: FrameBatch, max_new_tokens: int = 48) -> dict:
        torch = self.torch
        inputs = self.build_inputs(s, fb).to(self.model.device)
        n_prompt = int(inputs["input_ids"].shape[1])
        n_vis = int((inputs["input_ids"] == self.video_token_id).sum())
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=1 if s.is_mc else max_new_tokens, do_sample=False,
                output_logits=True, return_dict_in_generate=True,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        if self._cuda:
            torch.cuda.synchronize()
        latency = (time.perf_counter() - t0) * 1e3
        gen_ids = out.sequences[0, n_prompt:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        # latency_ms covers generate() only (GPU work + Python), not the processor.
        row = {"text": text, "prompt_tokens": n_prompt, "visual_tokens": n_vis, "latency_ms": latency}
        if s.is_mc:
            lp = torch.log_softmax(out.logits[0][0].float(), dim=-1)
            letters = option_letters(s)
            letter_lp = {L: float(lp[self.letter_ids[L]]) for L in letters if L in self.letter_ids}
            row.update(mc_summary(softmax_over_options(letter_lp, letters)))
        else:
            lps = []
            for t, logits in enumerate(out.logits):
                if t >= len(gen_ids):
                    break
                lps.append(float(torch.log_softmax(logits[0].float(), -1)[gen_ids[t]]))
            row["pred"] = text.strip()
            row["conf"] = math.exp(sum(lps) / max(1, len(lps)))  # geometric-mean token prob
        return row
