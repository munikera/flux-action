"""NATTEN flex-fna on XPU: neighborhood attention vs a dense masked reference, 2D and causal 3D."""
import time, torch
from natten.functional import na2d, na3d
import natten.backends.configs.checks as _nc
_rocm = _nc.is_rocm
_nc.is_rocm = lambda d: d.type == "xpu" or _rocm(d)  # Flex Attention runs on XPU through Triton
torch.manual_seed(0); dev = "xpu"
def ref_na2d(q, k, v, ks):
    B, H, W, h, d = q.shape; r = ks // 2
    out = torch.empty_like(q)
    for i in range(H):
        for j in range(W):
            i0 = min(max(i - r, 0), H - ks); j0 = min(max(j - r, 0), W - ks)
            kk = k[:, i0:i0+ks, j0:j0+ks].reshape(B, ks*ks, h, d); vv = v[:, i0:i0+ks, j0:j0+ks].reshape(B, ks*ks, h, d)
            a = torch.einsum("bhd,bnhd->bhn", q[:, i, j].float(), kk.float()) / d**0.5
            out[:, i, j] = torch.einsum("bhn,bnhd->bhd", a.softmax(-1), vv.float()).to(q.dtype)
    return out
q, k, v = (torch.randn(1, 12, 12, 2, 64, device=dev, dtype=torch.bfloat16) for _ in range(3))
o = na2d(q, k, v, kernel_size=(5, 5), backend="flex-fna")
print("na2d flex-fna max err vs reference:", (o.float() - ref_na2d(q, k, v, 5).float()).abs().max().item())
q3, k3, v3 = (torch.randn(1, 4, 16, 16, 4, 64, device=dev, dtype=torch.bfloat16) for _ in range(3))
o3 = na3d(q3, k3, v3, kernel_size=(4, 7, 7), is_causal=(True, False, False), backend="flex-fna"); torch.xpu.synchronize()
t = time.perf_counter()
for _ in range(10): na3d(q3, k3, v3, kernel_size=(4, 7, 7), is_causal=(True, False, False), backend="flex-fna")
torch.xpu.synchronize(); print(f"na3d causal flex-fna ok, {(time.perf_counter()-t)/10*1000:.2f} ms, out {tuple(o3.shape)}")
