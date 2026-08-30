# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""
Experimental Triton INT4-dequant GEMM for the ComfyUI-GGUF ``Q4_CR`` (int4_cr)
on-disk layout.

This module is a *research prototype*, deliberately NOT wired into the loader /
ops runtime path. It exists as a reproducible, correctly-validated artifact for
the question "can a Triton kernel beat comfy_kitchen's INT8 (Q8_CR) at diffusion
M".

= Why it is not production =

On this RTX 3080 Laptop (sm_86, CUDA 13.3), the kernel is numerically correct
(the fused int4-dequant math matches an fp64 reference within bf16 rounding
noise, mean rel err ~1.9%) but **much slower** than the existing paths:

    attn_in_qkv (K=4096, N=8192, M=14400):   Triton ~96ms   INT8(Q8) ~13ms   bf16 ~28ms
    mlp_fc1      (K=6144, N=12288,M=14400):   Triton ~223ms  INT8(Q8) ~34ms   bf16 ~67ms
    mlp_fc2      (K=12288,N=6144, M=14400):   Triton ~222ms  INT8(Q8) ~31ms   bf16 ~67ms

The kernel dequantizes the int4 weight to bf16 *in-kernel* and then runs a
bf16 ``tl.dot``. That is a dequant GEMM, not an int4 MMA: it pays the dequant
cost AND gets no tensor-core INT4 benefit, so it is ~3-7x slower than INT8.

Achieving true INT4 throughput needs an int4 x int4 tensor-core MMA
(``m16n8k64`` s4) with offline Hadamard rotation, which Triton 3.1.0 cannot
emit cleanly on Ampere (no native int4 tl.dot). That is a substantial CUDA/CUTLASS
effort, out of scope for this repo.

= Usage (offline / test only) =

The module is importable and its correctness is covered by
``tests/test_targeted_quantization.py``. It is gated on CUDA + Triton being
importable, so it never runs in a CPU-only environment.
"""

import os

import torch


def _triton_available() -> bool:
    """Return True when Triton + CUDA can be used for the experimental kernel."""
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
        return True
    except Exception:
        return False


_HAS_TRITON = _triton_available()
_HAS_TRITON_CUDA = False


if _HAS_TRITON:
    # Importing triton on Windows may require CUDA_PATH / a CC fallback to be set
    # by the caller (see the working environment notes). We still attempt it.
    import triton
    import triton.language as tl
    _HAS_TRITON_CUDA = True


def _q4cr_mm_kernel_factory():
    """Build the Triton JIT kernel (imported lazily to avoid compile cost)."""
    import triton
    import triton.language as tl

    @triton.jit
    def q4cr_mm_kernel(
        x_ptr, q_ptr, s_ptr, z_ptr, out_ptr,
        M, N, K, stride_xm, stride_xk,
        stride_qn, stride_qk, stride_sg, stride_sn,
        stride_zg, stride_zn, group_size,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        offs_mm = offs_m < M
        offs_nn = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + offs_k
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk
            x = tl.load(x_ptrs, mask=(offs_mm[:, None]) & (kk[None, :] < K), other=0.0)
            byte_col = k0 // 2 + tl.arange(0, BLOCK_K // 2)
            q_ptrs = q_ptr + offs_n[:, None] * stride_qn + byte_col[None, :] * stride_qk
            q_bytes = tl.load(
                q_ptrs, mask=(offs_nn[:, None]) & (byte_col[None, :] < K // 2), other=0
            ).to(tl.uint8)
            q_lo = q_bytes & 0xF
            q_hi = (q_bytes >> 4) & 0xF
            q = tl.interleave(q_lo, q_hi).to(tl.bfloat16)
            g_idx = (k0 + offs_k) // group_size
            s_ptrs = s_ptr + g_idx[None, :] * stride_sg + offs_n[:, None] * stride_sn
            scale = tl.load(
                s_ptrs, mask=(offs_nn[:, None]) & (g_idx[None, :] < K // group_size), other=1.0
            ).to(tl.bfloat16)
            z_ptrs = z_ptr + g_idx[None, :] * stride_zg + offs_n[:, None] * stride_zn
            zero = tl.load(
                z_ptrs, mask=(offs_nn[:, None]) & (g_idx[None, :] < K // group_size), other=0.0
            ).to(tl.bfloat16)
            w = ((q - 8.0) * scale + zero).to(tl.bfloat16)
            acc += tl.dot(x, tl.trans(w))
        out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=(offs_mm[:, None]) & (offs_nn[None, :]))

    return q4cr_mm_kernel


def triton_q4cr_mm(x, qweight, wscales, wzeros, group_size=64,
                   BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, warps=4, stages=3):
    """
    Experimental fused INT4-dequant GEMM for the Q4_CR (int4_cr) layout.

    y = x @ W.T where W[n,k] = (uint4(n,k) - 8) * scale[k//G, n] + zero[k//G, n].

    Requires Triton + CUDA. Raises RuntimeError otherwise. For correctness
    reference, see ``dequant.py`` (the INT4 reference dequant path).
    """
    if not _HAS_TRITON_CUDA:
        raise RuntimeError("Triton/CUDA not available; experimental kernel disabled.")
    kernel = _q4cr_mm_kernel_factory()
    M, K = x.shape
    N, _Kh = qweight.shape
    x = x.contiguous()
    qweight = qweight.contiguous()
    wscales = wscales.contiguous()
    wzeros = wzeros.contiguous()
    out = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    kernel[grid](
        x, qweight, wscales, wzeros, out,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        wscales.stride(0), wscales.stride(1),
        wzeros.stride(0), wzeros.stride(1),
        group_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=warps, num_stages=stages,
    )
    return out
