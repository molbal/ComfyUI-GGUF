"""Triton fused matmul backend for standard GGML K quants."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


SUPPORTED = {"Q4_K", "Q5_K", "Q6_K"}
_BLOCK_SIZE = 256
_TYPE_SIZES = {"Q4_K": 144, "Q5_K": 176, "Q6_K": 210}


def _supports(*, qtype, device, input_dtype, weight_shape, weight_dtype):
    return (
        triton is not None
        and device.type == "cuda"
        and qtype in SUPPORTED
        and input_dtype in {torch.float16, torch.bfloat16, torch.float32}
        and len(weight_shape) == 2
        and weight_shape[1] % _BLOCK_SIZE == 0
        and weight_dtype == torch.uint8
    )


if triton is not None:

    @triton.jit
    def _kquant_linear_kernel(
        x_ptr,
        q_ptr,
        bias_ptr,
        out_ptr,
        m_size,
        n_size,
        k_size,
        q_row_bytes,
        stride_xm,
        stride_xk,
        stride_qn,
        stride_om,
        stride_on,
        QTYPE: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        OUTPUT_DTYPE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offsets = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        valid_n = n < n_size

        for block in range(NUM_BLOCKS):
            block_k = block * 256
            qbase = q_ptr + n * stride_qn + block * q_row_bytes
            if QTYPE == 2:
                d = tl.load((qbase + 208).to(tl.pointer_type(tl.float16)), mask=valid_n, other=0.0).to(tl.float32)
                dmin = tl.zeros_like(d)
            else:
                d = tl.load(qbase.to(tl.pointer_type(tl.float16)), mask=valid_n, other=0.0).to(tl.float32)
                dmin = tl.load((qbase + 2).to(tl.pointer_type(tl.float16)), mask=valid_n, other=0.0).to(tl.float32)
            scales = tl.load(qbase[:, None] + 4 + tl.arange(0, 16)[None, :], mask=valid_n[:, None], other=0)
            for chunk in range(0, 256, BLOCK_K):
                local = chunk + tl.arange(0, BLOCK_K)
                x = tl.load(
                    x_ptr + m[:, None] * stride_xm + (block_k + local[None, :]) * stride_xk,
                    mask=(m[:, None] < m_size) & (local[None, :] < 256),
                    other=0.0,
                )
    
                if QTYPE == 0 or QTYPE == 1:
                    group = local // 32
                    within = local % 32
                    qoffset = 16 if QTYPE == 0 else 48
                    qbyte = tl.load(
                        qbase[:, None] + qoffset + (group[None, :] // 2) * 32 + within[None, :],
                        mask=valid_n[:, None],
                        other=0,
                    )
                    q = (qbyte >> ((group[None, :] % 2) * 4)) & 0xF
                    if QTYPE == 1:
                        high = tl.load(qbase[:, None] + 16 + within[None, :], mask=valid_n[:, None], other=0)
                        q = q | (((high >> group[None, :]) & 1) << 4)
    
                    scale_index = group % 4
                    scale_low = tl.load(qbase[:, None] + 4 + scale_index[None, :], mask=valid_n[:, None], other=0)
                    scale_high = tl.load(qbase[:, None] + 12 + scale_index[None, :], mask=valid_n[:, None], other=0)
                    min_low = tl.load(qbase[:, None] + 8 + scale_index[None, :], mask=valid_n[:, None], other=0)
                    scale = tl.where(
                        group[None, :] < 4,
                        scale_low & 0x3F,
                        (scale_high & 0x0F) | ((scale_low >> 2) & 0x30),
                    )
                    minimum = tl.where(
                        group[None, :] < 4,
                        min_low & 0x3F,
                        (scale_high >> 4) | ((min_low >> 2) & 0x30),
                    )
                    w = d[:, None] * scale.to(tl.float32) * q.to(tl.float32) - dmin[:, None] * minimum.to(tl.float32)
                else:
                    group16 = local // 16
                    group32 = local // 32
                    within = local % 32
                    ql_group = group32 % 4
                    ql_byte = (local // 128) * 64 + (ql_group % 2) * 32 + within
                    ql_shift = (ql_group // 2) * 4
                    ql = tl.load(qbase[:, None] + ql_byte[None, :], mask=valid_n[:, None], other=0)
                    qh_byte = (local // 128) * 32 + within
                    qh_shift = ql_group * 2
                    qh = tl.load(qbase[:, None] + 128 + qh_byte[None, :], mask=valid_n[:, None], other=0)
                    q = ((ql >> ql_shift[None, :]) & 0xF) | (((qh >> qh_shift[None, :]) & 0x3) << 4)
                    scale = tl.load(
                        qbase[:, None] + 192 + group16[None, :],
                        mask=valid_n[:, None],
                        other=0,
                    ).to(tl.int8).to(tl.float32)
                    w = d[:, None] * scale * (q.to(tl.float32) - 32.0)
    
                acc += tl.dot(x, tl.trans(w).to(x.dtype), input_precision="ieee")

        if HAS_BIAS:
            acc += tl.load(bias_ptr + n, mask=valid_n, other=0.0)[None, :]
        tl.store(
            out_ptr + m[:, None] * stride_om + n[None, :] * stride_on,
            acc.to(OUTPUT_DTYPE),
            mask=(m[:, None] < m_size) & valid_n[None, :],
        )

def linear(*, input, weight, qtype, weight_shape, bias=None):
    if not _supports(
        qtype=qtype,
        device=input.device,
        input_dtype=input.dtype,
        weight_shape=weight_shape,
        weight_dtype=weight.dtype,
    ):
        raise RuntimeError("Triton K-quant backend does not support this route")
    if input.ndim < 2:
        raise ValueError("K-quant Linear requires at least a 2-D activation")

    original_shape = input.shape
    x = input.reshape(-1, original_shape[-1]).contiguous()
    n_size, k_size = weight_shape
    output = torch.empty((x.shape[0], n_size), device=x.device, dtype=x.dtype)
    qtype_code = {"Q4_K": 0, "Q5_K": 1, "Q6_K": 2}[qtype]
    block_n = 32
    block_m = 16
    block_k = 32
    grid = (triton.cdiv(x.shape[0], block_m), triton.cdiv(n_size, block_n))
    _kquant_linear_kernel[grid](
        x,
        weight,
        bias,
        output,
        x.shape[0],
        n_size,
        k_size,
        _TYPE_SIZES[qtype],
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        output.stride(0),
        output.stride(1),
        QTYPE=qtype_code,
        NUM_BLOCKS=k_size // _BLOCK_SIZE,
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        OUTPUT_DTYPE=(
            triton.language.float16
            if input.dtype == torch.float16
            else triton.language.bfloat16
            if input.dtype == torch.bfloat16
            else triton.language.float32
        ),
        num_warps=4,
    )
    return output.reshape(*original_shape[:-1], n_size)


class _Backend:
    supports = staticmethod(_supports)
    linear = staticmethod(linear)


backend = _Backend()




