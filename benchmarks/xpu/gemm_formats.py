"""FLUX DiT GEMM shapes on one GPU: bf16 vs FP8 e4m3 rowwise (torch._scaled_mm) vs INT8 (torch._int_mm)."""
import sys, time, torch
dev = sys.argv[1]; mod = torch.get_device_module(dev); M = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
shapes = [("joint in_proj", 3072, 27648), ("mlp_out", 9216, 3072), ("attn_out", 3072, 3072)]
def bench(fn, it=30):
    for _ in range(5): fn()
    mod.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    mod.synchronize(); return (time.perf_counter() - t) / it * 1000
print(mod.get_device_name(0), "M =", M)
for name, K, N in shapes:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16); w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    flop = 2 * M * K * N / 1e12
    t = bench(lambda: a @ w.t()); row = f"{name:14s} K={K:5d} N={N:5d}  bf16 {t:6.3f} ms {flop/t*1e3:6.1f} TF/s"
    try:
        a8 = a.to(torch.float8_e4m3fn); w8 = w.to(torch.float8_e4m3fn)
        sa = torch.ones(M, 1, device=dev, dtype=torch.float32); sb = torch.ones(1, N, device=dev, dtype=torch.float32)
        f = lambda: torch._scaled_mm(a8, w8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        f(); t = bench(f); row += f" | fp8 rowwise {t:6.3f} ms {flop/t*1e3:6.1f} TF/s"
    except Exception as e:
        row += f" | fp8: {type(e).__name__}: {str(e)[:70]}"
    try:
        ai = torch.randint(-127, 127, (M, K), device=dev, dtype=torch.int8); wi = torch.randint(-127, 127, (K, N), device=dev, dtype=torch.int8)
        g = lambda: torch._int_mm(ai, wi)
        g(); t = bench(g); row += f" | int8 {t:6.3f} ms {flop/t*1e3:6.1f} TOP/s"
    except Exception as e:
        row += f" | int8: {type(e).__name__}: {str(e)[:70]}"
    print(row, flush=True)
