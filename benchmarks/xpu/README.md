# FLUX Action on Intel Arc Pro B70 (XPU)

Released `black-forest-labs/flux-3-action-droid`, synthetic DROID observation (3 cameras 360x640), one
B70 (32 GB), torch 2.14.1+xpu, batch 1, one `predict_action_chunk` (32 actions) after warm-up.

| Path | Latency | Peak | Source |
|---|---|---|---|
| Eager, `variants/gd` | 1738 ms | 18.4 GB | `flux_b70.jsonl` |
| Prepared BF16 + compile (reduce-overhead), `variants/sd` | 342 ms | 17.1 GB | `flux_b70.jsonl` |
| + tiled VAE neighborhood attention | 313 ms (VAE 22 ms, DiT 290 ms) | 25.3 GB* | `int8_tiled.log` |
| + INT8 W8A8 (159 GEMMs) | 243 ms | 18.8 GB | `int8_tiled.log` |

\* `flux_int8.py` measures peak differently from `flux_bench.py`.

Action drift vs BF16 (same observation): tiled mean 0.0025 rad, max 0.023; INT8 mean 0.021 rad, max 0.14.

    python flux_bench.py --device xpu --subfolder variants/sd --batches 1,2,4 --compile --json flux_b70.jsonl
    ./int8_ab.sh; ./int8_tiled.sh
