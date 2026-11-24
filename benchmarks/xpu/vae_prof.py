"""Compiled FLUX video-VAE encode (3 cameras) on XPU: top GPU kernels."""
import time, torch
from torch.profiler import profile, ProfilerActivity
from flux_action.models.video_vae import load_video_vae
spec = "black-forest-labs/flux-3-action-base:video_vae.safetensors@62878e2925e59b7a89ec14463ce89932624c490d"
vae = load_video_vae(spec, device="xpu", compile_model=False); vae.module.compile_encoder()
dt = next(vae.module.parameters()).dtype
x = (torch.rand(3, 3, 360, 640, device="xpu") * 2 - 1).to(dt)
with torch.inference_mode():
    for _ in range(4): vae.encode_frame(x)
    torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(5): vae.encode_frame(x)
    torch.xpu.synchronize(); print(f"encode 3 cams: {(time.perf_counter()-t)/5*1000:.1f} ms, dtype {dt}")
    with profile(activities=[ProfilerActivity.XPU]) as prof:
        vae.encode_frame(x); torch.xpu.synchronize()
ev = sorted([e for e in prof.key_averages() if e.self_device_time_total > 0], key=lambda e: -e.self_device_time_total)
tot = sum(e.self_device_time_total for e in ev) / 1000
print(f"GPU total {tot:.1f} ms (kernel rows only)")
for e in ev[:12]: print(f"  {e.self_device_time_total/1000:7.2f} ms x{e.count:<4} {e.key[:100]}")
