"""Peak GPU memory of FLUX Action prepare_inference, text encoder kept on the CPU (fits a 24 GB card?)."""
import gc, sys, torch
from flux_action.policy import FluxActionPolicy
dev = sys.argv[1]; mod = torch.get_device_module(dev)
p = FluxActionPolicy.from_pretrained("black-forest-labs/flux-3-action-droid", subfolder="variants/sd", device="cpu")
p.set_text_encoder_offload(True)
p = p.to(dev); mod.synchronize()
print(f"loaded: {mod.memory_allocated()/1e9:.2f} GB", flush=True)
mod.reset_peak_memory_stats(); p.prepare_inference(compile=False); gc.collect(); mod.empty_cache()
print(f"prepare peak {mod.max_memory_allocated()/1e9:.2f} GB, after {mod.memory_allocated()/1e9:.2f} GB", flush=True)
