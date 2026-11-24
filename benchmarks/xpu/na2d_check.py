"""Gathered na2d vs NATTEN flex-fna vs brute force, including border windows and the VAE's shapes."""
import os, torch
import flux_action.models.video_vae as vv
from natten.functional import na2d
torch.manual_seed(0)
def brute(q, k, v, ks):
    B, H, W, h, d = q.shape; out = torch.empty_like(q, dtype=torch.float32)
    for i in range(H):
        for j in range(W):
            i0 = min(max(i - ks // 2, 0), H - ks); j0 = min(max(j - ks // 2, 0), W - ks)
            kk = k[:, i0:i0+ks, j0:j0+ks].reshape(B, -1, h, d).float(); vv_ = v[:, i0:i0+ks, j0:j0+ks].reshape(B, -1, h, d).float()
            a = torch.einsum("bhd,bnhd->bhn", q[:, i, j].float(), kk) / d**0.5
            out[:, i, j] = torch.einsum("bhn,bnhd->bhd", a.softmax(-1), vv_)
    return out
for dev in ("cpu", "xpu"):
    q, k, v = (torch.randn(2, 11, 14, 3, 64, device=dev, dtype=torch.bfloat16) for _ in range(3))
    g = vv._na2d_gather(q, k, v, (5, 5)).float()
    print(dev, "gather vs brute force:", (g - brute(q, k, v, 5).to(dev)).abs().max().item())
    if dev == "xpu":
        n = na2d(q, k, v, kernel_size=(5, 5), backend="flex-fna").float()
        print(dev, "gather vs NATTEN flex-fna:", (g - n).abs().max().item(), " NATTEN vs brute:", (n - brute(q, k, v, 5).to(dev)).abs().max().item())
        q, k, v = (torch.randn(1, 90, 160, 4, 64, device=dev, dtype=torch.bfloat16) for _ in range(3))
        torch.xpu.reset_peak_memory_stats(); g = vv._na2d_gather(q, k, v, (5, 5)); torch.xpu.synchronize()
        print("90x160 peak", round(torch.xpu.max_memory_allocated() / 1e9, 3), "GB")
