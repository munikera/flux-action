import torch
from flux_action.policy import FluxActionPolicy
p = FluxActionPolicy.from_pretrained("black-forest-labs/flux-3-action-droid", subfolder="variants/gd", device="xpu")
gb = lambda: f"alloc {torch.xpu.memory_allocated()/1e9:.2f} GB, reserved {torch.xpu.memory_reserved()/1e9:.2f} GB"
print("after load:", gb())
dev = lambda m: {str(t.device) for t in m.parameters()} if m is not None else None
print("dit", dev(p.dit), sum(t.numel() for t in p.dit.parameters())/1e9, "B params")
print("text encoder", dev(p.text_encoder), "vae", dev(p.video_vae.module) if hasattr(p.video_vae, "module") else dev(p.video_vae))
free, total = torch.xpu.mem_get_info(); print(f"device free {free/1e9:.2f} of {total/1e9:.2f} GB")
