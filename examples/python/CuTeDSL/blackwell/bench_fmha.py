"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import argparse
import enum
import math
import time
from typing import Type, Tuple

import torch
import torch.nn.functional as F
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils as utils
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

from fmha import *
from triton.testing import do_bench


def bench_fmha(
    q_shape: Tuple[int, int, int, int],
    k_shape: Tuple[int, int, int, int],
    in_dtype: Type[cutlass.Numeric],
    out_dtype: Type[cutlass.Numeric],
    qk_acc_dtype: Type[cutlass.Numeric],
    pv_acc_dtype: Type[cutlass.Numeric],
    mma_tiler_mn: Tuple[int, int],
    is_persistent: bool,
    has_casual_mask: bool,
    scale_q: float,
    scale_k: float,
    scale_v: float,
    inv_scale_o: float,
    scale_softmax: float,
):
    """Execute Fused Multi-Head Attention (FMHA) on Blackwell architecture and validate results.

    This function creates random input tensors for query, key, and value, then performs the
    complete FMHA computation pipeline. It supports configurable data types, tiling parameters,
    and various attention masking options. Results can be validated against a PyTorch reference
    implementation or run multiple times for performance measurement.

    The implementation leverages specialized tensor memory operations and efficient math
    operations optimized for Blackwell architecture, including pipelined computation stages
    for maximum throughput.

    :param q_shape: Query tensor shape (B, S_q, H, D) where B=batch size, S_q=query sequence length,
                    H=number of heads, D=head dimension
    :type q_shape: Tuple[int, int, int, int]
    :param k_shape: Key tensor shape (B, S_k, H_k, D) where B=batch size, S_k=key sequence length,
                    H_k=number of key heads (H must be divisible by H_k), D=head dimension
    :type k_shape: Tuple[int, int, int, int]
    :param in_dtype: Input data type for query, key and value tensors
    :type in_dtype: Type[cutlass.Numeric]
    :param out_dtype: Output data type for attention output
    :type out_dtype: Type[cutlass.Numeric]
    :param qk_acc_dtype: Accumulator data type for query-key matrix multiplication
    :type qk_acc_dtype: Type[cutlass.Numeric]
    :param pv_acc_dtype: Accumulator data type for probability-value matrix multiplication
    :type pv_acc_dtype: Type[cutlass.Numeric]
    :param mma_tiler_mn: Matrix multiply accumulate tile shape (M, N)
    :type mma_tiler_mn: Tuple[int, int]
    :param is_persistent: Whether to use persistent kernel optimization
    :type is_persistent: bool
    :param has_casual_mask: Whether to apply causal masking
    :type has_casual_mask: bool
    :param scale_q: Scaling factor for query tensor
    :type scale_q: float
    :param scale_k: Scaling factor for key tensor
    :type scale_k: float
    :param scale_v: Scaling factor for value tensor
    :type scale_v: float
    :param inv_scale_o: Inverse scaling factor for output tensor
    :type inv_scale_o: float
    :param scale_softmax: Attention score scaling factor (defaults to 1/sqrt(D) if set to 0)
    :type scale_softmax: float
    :param tolerance: Maximum acceptable error for validation
    :type tolerance: float
    :param warmup_iterations: Number of warmup iterations
    :type warmup_iterations: int
    :param iterations: Number of iterations to run for performance testing
    :type iterations: int
    :param skip_ref_check: Skip validation against reference implementation
    :type skip_ref_check: bool

    :raises ValueError: If input shapes are incompatible or head dimension is unsupported
    :raises RuntimeError: If GPU is unavailable for computation
    """

    # print(f"Running Blackwell SM100 FMHA test with:")
    # print(f"  q_shape: {q_shape}")
    # print(f"  k_shape: {k_shape}")
    # print(f"  in_dtype: {in_dtype}")
    # print(f"  out_dtype: {out_dtype}")
    # print(f"  qk_acc_dtype: {qk_acc_dtype}")
    # print(f"  pv_acc_dtype: {pv_acc_dtype}")
    # print(f"  mma_tiler_mn: {mma_tiler_mn}")
    # print(f"  is_persistent: {is_persistent}")
    # print(f"  has_casual_mask: {has_casual_mask}")
    # print(f"  scale_q: {scale_q}")
    # print(f"  scale_k: {scale_k}")
    # print(f"  scale_v: {scale_v}")
    # print(f"  inv_scale_o: {inv_scale_o}")
    # print(f"  scale_softmax: {scale_softmax}")
    # print(f"  tolerance: {tolerance}")

    # Unpack parameters
    b, s_q, h, d = q_shape
    b_, s_k, h_k, d_ = k_shape

    if b != b_:
        raise ValueError("q & k must have the same batch size")

    if d != d_:
        raise ValueError("q & k must have the same head dimension")

    if d not in {32, 64, 128}:
        raise ValueError("head dimension must be 32, 64, or 128")

    if h % h_k != 0:
        raise ValueError("h must be divisible by h_k")

    if in_dtype not in {cutlass.Float8E4M3FN, cutlass.Float16}:
        raise ValueError("in_dtype must be Float8E4M3FN or Float16")

    if out_dtype not in {cutlass.Float8E4M3FN, cutlass.Float16}:
        raise ValueError("out_dtype must be Float8E4M3FN or Float16")

    if qk_acc_dtype not in {cutlass.Float32}:
        raise ValueError("qk_acc_dtype must be Float32")

    if pv_acc_dtype not in {cutlass.Float32}:
        raise ValueError("pv_acc_dtype must be Float32")

    # if iterations < 1:
    #     raise ValueError("iterations must be at least 1")

    h_r = h // h_k

    # Prepare pytorch tensors: Q, K, V (random from 0 to 2) and O (all zero)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    def create_and_permute_tensor(b, s, h_r, h_k, d, dtype, is_dynamic_layout=True):
        # (b, s, h_r, h_k, d) -> (s, d, h_r, h_k, b)
        shape = (b, s, h_r, h_k, d)
        permute_order = (1, 4, 2, 3, 0)
        is_fp8 = dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}

        # torch does not support fp8 type
        torch_dtype = cutlass.torch.dtype(dtype) if not is_fp8 else torch.uint8

        # Create dtype torch tensor (cpu)
        torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            shape,
            torch_dtype,
            permute_order=permute_order,
            init_type=cutlass.torch.TensorInitType.RANDOM,
            init_config=cutlass.torch.RandomInitConfig(
                min_val=0 if is_fp8 else -2, max_val=2
            ),
        )
        # Create dtype torch tensor (gpu)
        torch_tensor_gpu = torch_tensor_cpu.cuda()

        # Create f32 torch tensor (cpu)
        f32_torch_tensor = torch_tensor_cpu.to(dtype=torch.float32)

        # Create dtype cute tensor (gpu)
        cute_tensor = from_dlpack(torch_tensor_gpu, assumed_align=16)
        cute_tensor.element_type = dtype
        if is_dynamic_layout:
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=1)
        cute_tensor = cutlass_torch.convert_cute_tensor(
            f32_torch_tensor,
            cute_tensor,
            dtype,
            is_dynamic_layout=is_dynamic_layout,
        )

        return f32_torch_tensor, cute_tensor, torch_tensor_gpu

    q_ref, q_tensor, q_torch = create_and_permute_tensor(
        b, s_q, h_r, h_k, d, in_dtype, is_dynamic_layout=True
    )
    k_ref, k_tensor, k_torch = create_and_permute_tensor(
        b, s_k, 1, h_k, d, in_dtype, is_dynamic_layout=True
    )
    v_ref, v_tensor, v_torch = create_and_permute_tensor(
        b, s_k, 1, h_k, d, in_dtype, is_dynamic_layout=True
    )
    o_ref, o_tensor, o_torch = create_and_permute_tensor(
        b, s_q, h_r, h_k, d, out_dtype, is_dynamic_layout=True
    )

    mma_tiler = (*mma_tiler_mn, d)

    mask_type = MaskType.NO_MASK
    if has_casual_mask:
        mask_type = MaskType.CAUSAL_MASK
    else:
        if s_k % mma_tiler_mn[1] != 0:
            mask_type = MaskType.RESIDUAL_MASK

    fmha = BlackwellFusedMultiHeadAttentionForward(
        qk_acc_dtype,
        pv_acc_dtype,
        mma_tiler,
        is_persistent,
        mask_type,
    )

    # Get current CUDA stream from PyTorch
    torch_stream = torch.cuda.current_stream()
    # Get the raw stream pointer as a CUstream
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    if scale_softmax == 0.0:  # default to 1/sqrt(d)
        scale_softmax = 1.0 / math.sqrt(q_shape[1])
    log2_e = math.log2(
        math.exp(1.0)
    )  # gpu uses exp2 for perf concerns, we need an extra factor 'log2_e' here

    scale_softmax = scale_q * scale_k * scale_softmax
    scale_softmax_log2 = scale_softmax * log2_e
    scale_output = scale_v * inv_scale_o

    print("Compiling kernel with cute.compile ...")
    start_time = time.time()
    # compile fmha kernel
    compiled_fmha = cute.compile(
        fmha,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        scale_softmax_log2,
        scale_output,
        current_stream,
    )
    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time:.4f} seconds")

    # sync before benchmark
    torch.cuda.synchronize()

    # Warmup
    # for _ in range(warmup_iterations):
    #     compiled_fmha(
    #         q_tensor,
    #         k_tensor,
    #         v_tensor,
    #         o_tensor,
    #         scale_softmax_log2,
    #         scale_output,
    #         current_stream,
    #     )

    # # Execute kernel
    # for _ in range(iterations):
    #     compiled_fmha(
    #         q_tensor,
    #         k_tensor,
    #         v_tensor,
    #         o_tensor,
    #         scale_softmax_log2,
    #         scale_output,
    #         current_stream,
    #     )

    # benchmark
    fn = lambda: compiled_fmha(
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        scale_softmax_log2,
        scale_output,
        current_stream,
    )

    ms = do_bench(
        fn,
        warmup=100,
        rep=1000,
    )

    def flops(ms):
        # The number of flops is 2 * b * s_q * s_k * h * d.
        # This is for the two GEMMs (QK^T and P*V).
        total_ops = 2 * b * h * s_q * s_k * d
        if has_casual_mask:
            total_ops /= 2
        return total_ops / ms / 1e9

    print(
        f"bench_fmha_blackwell (batch_size={b}, q_len={s_q}, kv_len={s_k}, num_heads={h}, head_dim={d}, causal={has_casual_mask}),"
        f" perf: {flops(ms):.3f} TFLOPs/s, time: {ms:.3f} ms"
    )

if __name__ == "__main__":
    # add benchmark cases here
    benchmark_cases = [
        # b, s, h, d, dtype, causal
        (4, 1024, 8, 128, cutlass.Float16, False),
        (4, 1024, 8, 128, cutlass.Float16, True),
        (4, 4096, 8, 128, cutlass.Float16, False),
        (4, 4096, 8, 128, cutlass.Float16, True),
        (4, 1024, 8, 64, cutlass.Float16, False),
        (4, 1024, 8, 64, cutlass.Float16, True),
    ]

    for b, s, h, d, dtype, causal in benchmark_cases:
        q_shape = (b, s, h, d)
        k_shape = (b, s, h, d)

        bench_fmha(
            q_shape=q_shape,
            k_shape=k_shape,
            in_dtype=dtype,
            out_dtype=dtype,
            qk_acc_dtype=cutlass.Float32,
            pv_acc_dtype=cutlass.Float32,
            mma_tiler_mn=(128, 128),
            is_persistent=True,
            has_casual_mask=causal,
            scale_q=1.0,
            scale_k=1.0,
            scale_v=1.0,
            inv_scale_o=1.0,
            scale_softmax=0.0,
        )
    
