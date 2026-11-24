import time, torch
import flux_action.models.video_vae as vv
torch.manual_seed(0)
def bench(f, it=20):
    for _ in range(3): f()
    torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.xpu.synchronize(); return (time.perf_counter() - t) / it * 1000
for (H, W, heads) in [(90, 160, 4), (45, 80, 8), (23, 40, 16), (12, 20, 32), (13, 17, 2)]:
    q, k, v = (torch.randn(3, H, W, heads, 64, device="xpu", dtype=torch.bfloat16) for _ in range(3))
    g = vv._na2d_gather(q, k, v, (5, 5)); t = vv._na2d_tiled(q, k, v, (5, 5))
    err = (g.float() - t.float()).abs().max().item()
    print(f"{H}x{W} h{heads}: max |tiled - gather| = {err:.4f}; gather {bench(lambda: vv._na2d_gather(q, k, v, (5, 5))):.2f} ms, tiled {bench(lambda: vv._na2d_tiled(q, k, v, (5, 5))):.2f} ms", flush=True)
