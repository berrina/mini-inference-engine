# Project notes — mini inference engine

Lab journal. Raw observations, benchmark numbers, and bugs as they happen.
Polished writeups go in the README.

---

## 2026-07-15 — Day 1: environment + naive matmul

**Setup:** Google Colab, T4 GPU, Triton (via pip)

**Done:**
- Naive tiled matmul kernel in Triton (BLOCK_M/N/K = 64/64/32)
- Verified against torch.matmul: max abs error 0.062, allclose passes at atol=1e-1

**Why the 0.06 error is fine:** inputs are fp16 (~3 decimal digits of
precision), each output element sums 1024 products, and Triton/cuBLAS sum
in different orders. Float addition isn't associative, so small divergence
is expected — this is precision fuzz, not a bug.

**Why acc is float32 when inputs are fp16:** accumulating 1024 additions
in fp16 compounds rounding error, and fp16 overflows at 65,504 so long
sums can hit inf. Standard pattern: load fp16 (cheap memory traffic),
accumulate fp32 (safe math). Tensor cores are built around this.

**Bug hit:** NameError on kernel launch — kernel cell hadn't run in the
session. Lesson: notebook state is per-session, run cells top to bottom.
Also fixed a real bug: B pointer advance must use stride_bk, not stride_bn.

**Benchmarks:** (pending)
- Triton: 1.814 ms → 1.2 TFLOPS
- torch (cuBLAS): 0.146 ms → 14.7 TFLOPS
- Gap to close: 8%

**experiment on block size sweep** (07/15) 
---

**Question:** which tile config (BM×BN×BK) runs fastest on the T4, and why?

**Method:** same naive kernel, swept all combos of BM/BN ∈ {32,64,128},
BK ∈ {32,64} — 18 configs. Timed each with triton.testing.do_bench at
1024³, fp16 inputs, fp32 accumulate.

**Results:**

| BM  | BN  | BK | ms     | TFLOPS |
|-----|-----|----|--------|--------|
| 32  | 64  | 32 | 0.755  | 2.8    |
| 32  | 64  | 64 | 0.787  | 2.7    |
| 64  | 32  | 64 | 0.946  | 2.3    |
| 64  | 32  | 32 | 0.967  | 2.2    |
| 64  | 64  | 32 | 1.477  | 1.5    |
| 32  | 128 | 32 | 1.408  | 1.5    |
| 128 | 32  | 32 | 1.728  | 1.2    |
| 32  | 32  | 64 | 1.871  | 1.1    |
| 32  | 32  | 32 | 2.067  | 1.0    |
| 32  | 128 | 64 | 11.257 | 0.2    |
| 64  | 64  | 64 | 10.670 | 0.2    |
| 128 | 32  | 64 | 10.304 | 0.2    |
| 64  | 128 | 64 | 20.833 | 0.1    |
| 64  | 128 | 32 | 21.802 | 0.1    |
| 128 | 64  | 32 | 22.054 | 0.1    |
| 128 | 64  | 64 | 20.286 | 0.1    |
| 128 | 128 | 32 | 35.347 | 0.1    |
| 128 | 128 | 64 | 34.990 | 0.1    |

**Finding 1 — free 2.3x:** winner is 32×64×32 at 2.8 TFLOPS (~19% of
cuBLAS, up from 8%). Our original hand-picked 64/64/32 ranked 5th.
Lesson: measure, don't guess — this is why autotuning exists.

**Finding 2 — the silent cliff:** big configs didn't error, they
collapsed (up to 28x slower than the winner) while still producing
correct output. Likely register/shared-memory spilling: when a config
needs more on-chip storage than exists, the compiler silently spills to
slow global memory. Correct-but-catastrophically-slow is a real GPU
failure mode; try/except never fired.

**Finding 3 — wide beats tall:** mirror pairs favor larger BN
(32×64 → 2.8 vs 64×32 → 2.2; 32×128 → 1.5 vs 128×32 → 1.2).
Hypothesis: coalesced access — consecutive elements along rows of B/C
are contiguous in memory, so wider BN = longer contiguous loads.
UNCONFIRMED — verify with Nsight profiling.

**Chunk math (winning config, 1024³):** each worker does K/BK = 32 laps.
Per lap: one 32×32 chunk of A (1024 elems) + one 32×64 chunk of B
(2048 elems). Total ~98K elements loaded to produce a 32×64 = 2K patch:
~48 loads per output element. Shrinking that ratio is what tiling and
reuse are for.

**Artifacts:** block_sweep_t4.png (bar chart vs cuBLAS line)

**New baseline: 2.8 TFLOPS.** Correctness re-verified at 32/64/32.

**Next:** pipelining (overlap loads with compute) + shared-memory reuse.
Target: meaningful step toward cuBLAS's 14.7.
