import sys, time, subprocess, torch
from flux_action.policy import FluxActionPolicy
mode = sys.argv[1]
if mode == "direct":
    p = FluxActionPolicy.from_pretrained("black-forest-labs/flux-3-action-droid", subfolder="variants/gd", device="xpu")
else:
    p = FluxActionPolicy.from_pretrained("black-forest-labs/flux-3-action-droid", subfolder="variants/gd", device="cpu")
    p = p.to("xpu")
torch.xpu.synchronize()
free, total = torch.xpu.mem_get_info()
smi = subprocess.run(["xpu-smi", "stats", "-d", sys.argv[2]], capture_output=True, text=True).stdout
used = [l for l in smi.splitlines() if "GPU Memory Used" in l]
print(f"{mode}: alloc {torch.xpu.memory_allocated()/1e9:.2f} GB, mem_get_info free {free/1e9:.2f} GB, xpu-smi: {used[0].split('current:')[-1].strip() if used else '?'}")
