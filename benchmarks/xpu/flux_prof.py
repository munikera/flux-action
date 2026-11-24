"""Kernel-level view of one compiled FLUX DiT sample on XPU: GPU busy vs wall time and top kernels."""
import gc, time, torch, numpy as np
from torch.profiler import profile, ProfilerActivity
from flux_action.policy import FluxActionPolicy
p = FluxActionPolicy.from_pretrained("black-forest-labs/flux-3-action-droid", subfolder="variants/gd", device="xpu")
p.set_text_encoder_offload(True); torch.xpu.empty_cache(); p.prepare_inference(compile=True); gc.collect(); torch.xpu.empty_cache()
obs = lambda: {k: torch.rand(1, 3, 360, 640, device="xpu") for k in ("images.wrist", "images.left", "images.right")} | {"state": torch.rand(1, 8, device="xpu"), "task": ["put the marker in the cup"]}
with torch.no_grad():
    for _ in range(3): p.predict_action_chunk(obs())
    torch.xpu.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        t = time.perf_counter(); p.predict_action_chunk(obs()); torch.xpu.synchronize(); wall = (time.perf_counter() - t) * 1000
ev = [e for e in prof.key_averages() if e.device_time_total > 0]
gpu = sum(e.self_device_time_total for e in ev) / 1000
n = sum(e.count for e in prof.key_averages() if e.self_device_time_total > 0)
print(f"wall {wall:.0f} ms, GPU kernel time {gpu:.0f} ms, kernels {n}")
for e in sorted(ev, key=lambda e: -e.self_device_time_total)[:14]:
    print(f"  {e.self_device_time_total/1000:8.1f} ms  x{e.count:<5} {e.key[:110]}")
