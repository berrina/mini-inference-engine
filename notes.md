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

**Next:** block size sweep (16/32/64/128), then Nsight profiling.
