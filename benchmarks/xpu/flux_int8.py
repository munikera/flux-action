"""FLUX 3 Action: BF16 vs W8A8 INT8 block GEMMs. Same seeded observations each run; actions saved per run,
compared with a reference run, and timed.

  python flux_int8.py --tag bf16            # reference (saves actions_bf16.npy)
  python flux_int8.py --tag int8 --int8     # compares with actions_bf16.npy
"""
import argparse, gc, os, time
import numpy as np
import torch
from flux_action.policy import FluxActionPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="xpu")
ap.add_argument("--subfolder", default="variants/sd")
ap.add_argument("--int8", action="store_true")
ap.add_argument("--eager", action="store_true")
ap.add_argument("--tag", required=True)
ap.add_argument("--ref", default="bf16")
ap.add_argument("--iters", type=int, default=10)
a = ap.parse_args()
dev, mod = a.device, torch.get_device_module(a.device)
fp8r = "fp8r" in a.subfolder  # native FP8r packages load straight onto the GPU
p = FluxActionPolicy.from_pretrained("black-forest-labs/flux-3-action-droid", subfolder=a.subfolder, device=dev if fp8r else "cpu")
p.set_text_encoder_offload(True)
if not fp8r:
    p = p.to(dev)
mod.empty_cache()
p.prepare_inference(compile=not a.eager); gc.collect(); mod.empty_cache()
if a.int8:
    from flux_action.models.int8_w8a8 import quantize_prepared_dit
    n = quantize_prepared_dit(p.dit); gc.collect(); mod.empty_cache()
    print(f"int8: {n} GEMMs converted; {mod.memory_allocated()/1e9:.2f} GB allocated", flush=True)
g = torch.Generator().manual_seed(0)
def obs(i):
    gg = torch.Generator().manual_seed(1000 + i)
    cam = lambda: torch.rand(1, 3, 360, 640, generator=gg).to(dev)
    return {"images.wrist": cam(), "images.left": cam(), "images.right": cam(),
            "state": (torch.rand(1, 8, generator=gg) * 2 - 1).to(dev), "task": ["put the marker in the cup"]}
with torch.inference_mode():
    for i in range(3): p.predict_action_chunk(obs(i))  # warm-up and compile
    acts = [p.predict_action_chunk(obs(i)).float().cpu().numpy() for i in range(8)]
    ts = []
    for i in range(a.iters):
        o = obs(100 + i); mod.synchronize(); t = time.perf_counter(); p.predict_action_chunk(o).cpu(); ts.append((time.perf_counter() - t) * 1000)
acts = np.concatenate(acts)
np.save(f"actions_{a.tag}.npy", acts)
print(f"{a.tag}: {np.mean(ts):.1f} ms per query (p99 {np.percentile(ts, 99):.1f}), peak {mod.max_memory_allocated()/1e9:.2f} GB", flush=True)
ref = f"actions_{a.ref}.npy"
if a.tag != a.ref and os.path.exists(ref):
    r = np.load(ref)
    d = np.abs(acts - r)
    joints, grip = d[..., :7], d[..., 7]
    print(f"{a.tag} vs {a.ref}: joints max {joints.max():.4f} rad, mean {joints.mean():.4f} rad; gripper max {grip.max():.4f}; "
          f"action range {r[..., :7].min():.2f}..{r[..., :7].max():.2f}", flush=True)
