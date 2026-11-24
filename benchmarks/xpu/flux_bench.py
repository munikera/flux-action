"""FLUX 3 Action (DROID policy) latency vs batch size on one GPU, CUDA or Intel XPU.

Synthetic DROID observations: three (360, 640) RGB cameras in [0, 1], an 8-value joint state and one
task caption, the inputs the released policy expects. Each batch size is timed as one
predict_action_chunk call (observation encode, the 4-step DiT sampler, action decode), after warm-up.
The text encoder is moved to the CPU once the caption is cached: repeated captions never use it.

  python flux_bench.py --device xpu --subfolder variants/gd --batches 1,2,4,8 --compile
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from flux_action.policy import FluxActionPolicy

p = argparse.ArgumentParser()
p.add_argument("--device", default="xpu")
p.add_argument("--checkpoint", default="black-forest-labs/flux-3-action-droid")
p.add_argument("--subfolder", default="variants/gd", help="gd: guidance-distilled, 4 steps, BF16")
p.add_argument("--batches", default="1,2,4,8")
p.add_argument("--compile", action="store_true", help="prepare_inference(compile=True)")
p.add_argument("--iters", type=int, default=10)
p.add_argument("--json")
a = p.parse_args()

dev = torch.device(a.device)
mod = torch.get_device_module(dev)
t0 = time.perf_counter()
# The text encoder (8.9 GB) only runs for captions it has not cached, so keep it on the CPU (FLUX Action's
# own offload option). BF16 packages load on the CPU first so the encoder never reaches the GPU (needed on
# 24 GB cards); native FP8r packages load straight onto the GPU, as FLUX Action requires.
fp8r = "fp8r" in a.subfolder
policy = FluxActionPolicy.from_pretrained(a.checkpoint, subfolder=a.subfolder, device=a.device if fp8r else "cpu")
policy.set_text_encoder_offload(True)
if not fp8r:
    policy = policy.to(a.device)
load_s = time.perf_counter() - t0
mod.empty_cache()
prepared = policy.prepare_inference(compile=a.compile)
import gc  # noqa: E402  the replaced unpacked DiT can sit in a reference cycle until collected
gc.collect()
mod.empty_cache()
print(f"after prepare: {mod.memory_allocated() / 1e9:.2f} GB allocated", flush=True)
task = "put the marker in the cup"
rng = np.random.default_rng(0)


def observation(bs):
    cam = lambda: torch.from_numpy(rng.random((bs, 3, 360, 640), dtype=np.float32)).to(dev)
    return {"images.wrist": cam(), "images.left": cam(), "images.right": cam(),
            "state": torch.from_numpy(rng.random((bs, 8), dtype=np.float32)).to(dev), "task": [task] * bs}


res = {"device": mod.get_device_name(0), "torch": torch.__version__, "subfolder": a.subfolder,
       "compile": a.compile, "prepared_backend": prepared, "load_s": round(load_s, 1), "rows": []}
with torch.inference_mode():
    policy.predict_action_chunk(observation(1))  # caches the caption's text context (encoder visits the GPU once)
    mod.empty_cache()
    for bs in [int(x) for x in a.batches.split(",")]:
        try:
            obs = observation(bs)
            t = time.perf_counter()
            out = policy.predict_action_chunk(obs)
            mod.synchronize()
            first_s = time.perf_counter() - t
            for _ in range(2):
                policy.predict_action_chunk(obs)
            mod.synchronize()
            mod.reset_peak_memory_stats()
            ts = []
            for _ in range(a.iters):
                obs = observation(bs)
                t = time.perf_counter()
                policy.predict_action_chunk(obs).cpu()
                ts.append((time.perf_counter() - t) * 1000)
        except torch.OutOfMemoryError:
            print(json.dumps({"batch": bs, "oom": True}), flush=True)
            break
        ms = float(np.mean(ts))
        row = {"batch": bs, "latency_ms": round(ms, 1), "p99_ms": round(float(np.percentile(ts, 99)), 1),
               "queries_per_s": round(bs * 1000 / ms, 2), "first_call_s": round(first_s, 1),
               "peak_mem_gb": round(mod.max_memory_allocated() / 1e9, 2), "actions_shape": list(out.shape)}
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
print(json.dumps(res, indent=2))
if a.json:
    with open(a.json, "a") as f:
        f.write(json.dumps(res) + "\n")
