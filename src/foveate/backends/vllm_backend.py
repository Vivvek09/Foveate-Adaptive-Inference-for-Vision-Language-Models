"""vLLM offline engine: the fast path for running every question through a config.

Frames are sampled and resized by Foveate (see data/frames.py) and handed to
vLLM as a pre-sampled video plus its metadata, so vLLM neither re-samples nor
re-sizes them. Multiple-choice answers are scored from next-token log-probs
over the option letters (one decode step per question).
"""

from __future__ import annotations

import logging
import math
import time

from ..data import Sample
from ..data.frames import FrameBatch
from ..prompts import (
    chat_messages,
    clean_token,
    mc_summary,
    option_letters,
    parse_letter,
    softmax_over_options,
)

log = logging.getLogger(__name__)


class VLLMBackend:
    def __init__(
        self,
        model_id: str,
        *,
        quantization: str | None = None,
        dtype: str = "auto",
        max_res: int = 672,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.90,
        enforce_eager: bool = False,
        tensor_parallel_size: int = 1,
        max_num_seqs: int | None = None,
        seed: int = 0,
    ):
        from transformers import AutoProcessor
        from vllm import LLM

        from ..utils import default_dtype

        self.model_id = model_id
        dtype = default_dtype() if dtype == "auto" else dtype
        if quantization == "awq" and dtype == "bfloat16":
            dtype = "float16"  # AWQ kernels run in fp16
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tok = self.processor.tokenizer
        kwargs = dict(
            model=model_id,
            dtype=dtype,
            quantization=quantization,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"video": 1, "image": 0},
            # Frames arrive already resized; this range only has to contain them.
            # Keeping max_pixels near the real maximum also keeps vLLM's start-up
            # memory profiling (which uses a worst-case dummy video) realistic.
            mm_processor_kwargs={"min_pixels": 28 * 28, "max_pixels": int(max_res * max_res)},
            seed=seed,
        )
        if max_num_seqs:
            kwargs["max_num_seqs"] = max_num_seqs
        log.info("starting vLLM: %s", {k: v for k, v in kwargs.items() if k != "model"})
        self.llm = LLM(**kwargs)
        self.dtype = dtype
        self.video_metadata = True  # flips off automatically on older vLLM
        self.letter_ids = {}
        for L in "ABCDEFGH":
            ids = self.tok.encode(L, add_special_tokens=False)
            if len(ids) == 1:
                self.letter_ids[L] = ids[0]
        self.video_token_id = self.tok.convert_tokens_to_ids("<|video_pad|>")
        import inspect

        from vllm import SamplingParams

        self._has_token_id_logprobs = (
            "logprob_token_ids" in inspect.signature(SamplingParams.__init__).parameters
            or "logprob_token_ids" in getattr(SamplingParams, "__struct_fields__", ())
        )

    def _prompt(self, s: Sample, fb: FrameBatch) -> dict:
        text = self.processor.apply_chat_template(
            chat_messages(s), tokenize=False, add_generation_prompt=True
        )
        if not self.video_metadata:
            # Older vLLM: plain frame array (temporal positions assume the default fps).
            return {"prompt": text, "multi_modal_data": {"video": fb.as_array()}}
        meta = dict(fb.meta)
        meta["do_sample_frames"] = False
        meta["height"], meta["width"] = fb.height, fb.width
        return {"prompt": text, "multi_modal_data": {"video": (fb.as_array(), meta)}}

    def _params(self, s: Sample, max_new_tokens: int):
        from vllm import SamplingParams

        if s.is_mc:
            ids = [self.letter_ids[L] for L in option_letters(s) if L in self.letter_ids]
            if self._has_token_id_logprobs and len(ids) == len(option_letters(s)):
                # Exact log-probs for exactly the option letters (newer vLLM).
                return SamplingParams(temperature=0.0, max_tokens=1, logprob_token_ids=ids)
            return SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        return SamplingParams(temperature=0.0, max_tokens=max_new_tokens, logprobs=1)

    def _parse(self, s: Sample, out) -> dict:
        co = out.outputs[0]
        row = {
            "text": co.text,
            "prompt_tokens": len(out.prompt_token_ids or []),
            "visual_tokens": sum(1 for t in (out.prompt_token_ids or []) if t == self.video_token_id),
        }
        if s.is_mc:
            letters = option_letters(s)
            first = co.logprobs[0] if co.logprobs else {}
            id2letter = {v: k for k, v in self.letter_ids.items()}
            letter_lp: dict[str, float] = {}
            for tid, lp in first.items():
                L = id2letter.get(tid) or clean_token(lp.decoded_token or "")
                if L in letters:
                    letter_lp[L] = max(letter_lp.get(L, -math.inf), lp.logprob)
            if letter_lp:
                row.update(mc_summary(softmax_over_options(letter_lp, letters)))
            else:  # nothing usable in top-k: fall back to the generated text
                row.update({"pred": parse_letter(co.text, letters), "conf": 0.0,
                            "margin": 0.0, "entropy": math.log(len(letters))})
        else:
            n = max(1, len(co.token_ids))
            row["pred"] = co.text.strip()
            cum = co.cumulative_logprob
            row["conf"] = math.exp(cum / n) if cum is not None else float("nan")
        return row

    def predict_batch(
        self, samples: list[Sample], batches: list[FrameBatch], max_new_tokens: int = 48
    ) -> tuple[list[dict], float]:
        params = [self._params(s, max_new_tokens) for s in samples]
        prompts = [self._prompt(s, fb) for s, fb in zip(samples, batches)]
        t0 = time.perf_counter()
        try:
            outs = self.llm.generate(prompts, params, use_tqdm=False)
        except (TypeError, ValueError, AssertionError) as e:
            if not self.video_metadata:
                raise
            log.warning("vLLM rejected (frames, metadata) video input (%s); "
                        "retrying with plain frame arrays", str(e).splitlines()[0][:120])
            self.video_metadata = False
            prompts = [self._prompt(s, fb) for s, fb in zip(samples, batches)]
            t0 = time.perf_counter()
            outs = self.llm.generate(prompts, params, use_tqdm=False)
        wall = time.perf_counter() - t0
        return [self._parse(s, o) for s, o in zip(samples, outs)], wall

    def latency_probe(
        self, samples: list[Sample], batches: list[FrameBatch], max_new_tokens: int = 48
    ) -> list[float]:
        """Batch-1 end-to-end latency (ms) inside vLLM: processor + encoder + prefill + decode."""
        lat = []
        for s, fb in zip(samples, batches):
            p, sp = self._prompt(s, fb), self._params(s, max_new_tokens)
            t0 = time.perf_counter()
            self.llm.generate([p], sp, use_tqdm=False)
            lat.append((time.perf_counter() - t0) * 1e3)
        return lat
