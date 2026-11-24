"""FLUX Action video VAE single-frame encode on XPU: NATTEN backend, peak memory and time."""
import logging, os, time, torch
logging.basicConfig(level=logging.INFO)
os.environ.setdefault("F3_NATTEN_BACKEND", "flex-fna")
from flux_action.models.video_vae import load_video_vae
import flux_action.models.video_vae as vv
spec = "black-forest-labs/flux-3-action-base:video_vae.safetensors@62878e2925e59b7a89ec14463ce89932624c490d"
vae = load_video_vae(spec, device="xpu", compile_model=False)
orig = vv._natten_attention_kwargs
seen = set()
def spy(q, k, v, **kw):
    out = orig(q, k, v, **kw)
    key = (tuple(q.shape), tuple(kw["kernel_size"]), out["backend"], out.get("torch_compile"))
    if key not in seen: seen.add(key); print("NA call", key, q.dtype, flush=True)
    return out
vv._natten_attention_kwargs = spy
for cams in (1, 3):
    dt = next(vae.module.parameters()).dtype
    x = (torch.rand(cams, 3, 360, 640, device="xpu") * 2 - 1).to(dt)
    torch.xpu.reset_peak_memory_stats()
    if True:
        for i in range(3):
            t = time.perf_counter(); z = vae.encode_frame(x); torch.xpu.synchronize()
            print(f"cams {cams} run {i}: {(time.perf_counter()-t)*1000:.0f} ms, latent {tuple(z.shape)}, peak {torch.xpu.max_memory_allocated()/1e9:.2f} GB", flush=True)
