"""Where a FLUX 3 Action query spends its time on XPU: observation encode (video VAE) vs the DiT sampler."""
import gc, os, sys, time, torch, numpy as np
from flux_action.policy import FluxActionPolicy
compile_ = "--compile" in sys.argv
DEV = os.environ.get("DEV", "xpu"); SUB = os.environ.get("SUB", "variants/gd"); M = torch.get_device_module(DEV)
p = FluxActionPolicy.from_pretrained("black-forest-labs/flux-3-action-droid", subfolder=SUB, device=DEV if "fp8r" in SUB else "cpu")
p.set_text_encoder_offload(True)
if "fp8r" not in SUB: p = p.to(DEV)
M.empty_cache(); p.prepare_inference(compile=compile_); gc.collect(); M.empty_cache()
times = {}
def timed(name, fn):
    def w(*a, **k):
        M.synchronize(); t = time.perf_counter(); r = fn(*a, **k); M.synchronize()
        times.setdefault(name, []).append((time.perf_counter() - t) * 1000); return r
    return w
p._encode_observation = timed("observation encode (3 cameras, video VAE)", p._encode_observation)
p._sample = timed("DiT sampler", p._sample)
vae = p.video_vae
vae.encode_frame = timed("  of which encode_frame", vae.encode_frame)
rng = np.random.default_rng(0)
obs = lambda: {"images.wrist": torch.rand(1, 3, 360, 640, device=DEV), "images.left": torch.rand(1, 3, 360, 640, device=DEV),
               "images.right": torch.rand(1, 3, 360, 640, device=DEV), "state": torch.rand(1, 8, device=DEV), "task": ["put the marker in the cup"]}
with torch.inference_mode():
    for i in range(6):
        if i == 2: times.clear()
        M.synchronize(); t = time.perf_counter(); p.predict_action_chunk(obs()); M.synchronize()
        times.setdefault("total predict_action_chunk", []).append((time.perf_counter() - t) * 1000)
print(("compile" if compile_ else "eager"), SUB, M.get_device_name(0))
for k, v in times.items(): print(f"  {k}: {np.mean(v):.0f} ms")
