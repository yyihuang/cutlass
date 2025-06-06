# NVIDIA CUTLASS Changelog


## [3.9.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.9.0) (2025-04-24)

* Support for Blackwell SM120 kernels for GeForce GPUs in CUTLASS 3.x API:
  - Collective mainloops that target for:
    * [Blockscaled datatypes with support for dense GEMM](./include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp)
    * [Blockscaled datatypes with support for sparse GEMM](./include/cutlass/gemm/collective/sm120_blockscaled_sparse_mma_tma.hpp)
  - New [GEMM](./include/cutlass/gemm/dispatch_policy.hpp) and [epilogue](./include/cutlass/epilogue/dispatch_policy.hpp) dispatch policies for collectives, kernel layers, and builders.
  - [Blackwell SM120 epilogue](./include/cutlass/epilogue/fusion/sm120_visitor_store_tma_warpspecialized.hpp) and [full set of EVT fusions](./include/cutlass/epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp).
* Set of examples that demonstrate the usage of the 3.x API for targeting Blackwell SM120 architecture:
  - [Blockscaled GEMM with NVFP4 input datatype and BF16 output tensor](./examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu).
  - [Blockscaled GEMM with NVFP4 input datatype and NVFP4 output tensor with scale factor generation](./examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu).
  - [Blockscaled GEMM with mixed input datatype (MXFP8 and MXFP6) and BF16 output tensor](./examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu).
  - [Grouped GEMM with nvfp4 datatype](./examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu).
  - [Sparse Blockscaled GEMM with mxfp8 input datatype and BF16 output tensor](./examples/80_blackwell_geforce_sparse_gemm/80a_blackwell_geforce_mxfp8_bf16_sparse_gemm.cu).
  - [Sparse Blockscaled GEMM with NVFP4 input datatype and NVFP4 output tensor](./examples/80_blackwell_geforce_sparse_gemm/80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm.cu).
* Set of unit tests that demonstrate the usage of both [sparse](./test/unit/gemm/device/sm120_blockscaled_sparse_tensorop_gemm/) and [dense](./test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/) Blackwell SM120 blockscaled GEMM.
* Support for Blackwell SM100 Sparse kernels:
  - Collective mainloop that target for
    * [SM100 Sparse GEMM](./include/cutlass/gemm/collective/sm100_sparse_mma_warpspecialized.hpp)
* Set of example that demonstrate the usage of the 3.x API for targeting Blackwell SM100 Sparse GEMM:
  - [Sparse GEMM](./examples/83_blackwell_sparse_gemm/83_blackwell_sparse_gemm.cu)
  - [Blockscaled Sparse GEMM with NVFP4 input data type](./examples/84_blackwell_narrow_precision_sparse_gemm/84a_blackwell_nvfp4_bf16_sparse_gemm.cu)
  - [Blockscaled Sparse GEMM with mixed input data type (MXFP8 and MXFP4)](./examples/84_blackwell_narrow_precision_sparse_gemm/84b_blackwell_mixed_mxfp8_bf16_sparse_gemm.cu)
* Set of unit tests that demonstrate the usage of [sparse](./test/unit/gemm/device/sm100_sparse_tensorop_gemm) and [blockscaled sparse](./test/unit/gemm/device/sm100_blockscaled_sparse_tensorop_gemm) Blackwell SM100 GEMM.
* A new Multi-head Latent Attention (MLA) for SM100 Blackwell architecture in CUTLASS [example](./examples/77_blackwell_fmha/) covers the flashMLA-like weight-absorbed decoding use-case.
* A new FMHA Backward kernel for SM100 Blackwell architecture extends CUTLASS [example](./examples/77_blackwell_fmha/) to show how the five backward pass MMAs can be fused into a single kernel to achieve high performance.
* A new [distributed GEMM example](./examples/82_blackwell_distributed_gemm/82_blackwell_distributed_gemm.cu) for SM100 Blackwell architecture.
* Enhancement and new support of block-wise and group-wise GEMM for Hopper and Blackwell architectures:
  - Enhancement of [blockwise GEMM](./examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu) for Hopper architecture.
  - Enhancement of [groupwise GEMM](./examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling.cu) for Hopper architecture.
  - Support for [grouped GEMM with blockwise and groupwise scaling](./examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling/) for Hopper architecture.
  - Support for [grouped-wise GEMM](./tools/profiler/src/blockwise_gemm_operation_profiler.cu) in CUTLASS profiler.
  - Support for [blockwise GEMM](./examples/81_blackwell_gemm_blockwise/81_blackwell_gemm_blockwise.cu) for Blackwell architecture.
  - Support for [groupwise GEMM](./examples/81_blackwell_gemm_blockwise/81_blackwell_gemm_groupwise.cu) for Blackwell architecture.
  - Support for [grouped GEMM with blockwise](./examples/81_blackwell_gemm_blockwise/81_blackwell_grouped_gemm_blockwise.cu) and [groupwise scaling](./examples/81_blackwell_gemm_blockwise/81_blackwell_grouped_gemm_groupwise.cu) for Blackwell architecture.
* Added support for enhanced kernel performance search (auto-tuning) in CUTLASS profiler:
  - Sorting performance results by GFLOPs/second: Users can now sort the final performance report based on GFLOPs/second, making it easier to identify the most efficient kernels.
  - Exhaustive search for best kernel performance in GFLOPs/second: The profiler now searches for the best-performing kernel across a range of problem sizes, swizzle sizes, rasterization orders, and dynamic cluster configurations to maximize performance.
  - Performance search under a fixed GEMM shape: Enables exhaustive tuning within a fixed GEMM shape, exploring various kernel parameters to find the best configuration.
  - More detailed introductions and examples to leverage this feature can be found in [profiler.md](./media/docs/cpp/profiler.md#exhaustive-search-mode-and-top-k-output-ranking-according-to-performance-in-gflopss).
* Support `void` as the D element in sm100 kernel epilogues.
* Various improvements and fixes from the community and CUTLASS team. Thanks to everyone who submitted PRs!
* Optimal code generation with CUDA toolkit versions 12.8U1.

## [3.8.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.8.0) (2025-01-25)

* Support for new CuTe building blocks specifically for Blackwell SM100 architecture:
  - [5th generation Blackwell Tensor Core instructions (TCGen05)](./include/cute/atom/mma_traits_sm100.hpp) via CuTe MMA atoms.
  - Extensions to [Tensor Memory Accelerator](./include/cute/atom/copy_traits_sm100_tma.hpp) via CuTe Copy atoms.
  - Exposure of Blackwell's new tensor memory (note: distinct from TMA) as [`tmem`](./include/cute/pointer.hpp) across CuTe as a first class data locale.
  - Exposure of [`tmem->rmem`, `rmem->tmem` and `smem->tmem data movement instructions`](./include/cute/atom/copy_traits_sm100.hpp) as copy atoms in CuTe.
  - [`make_tmem_copy()`](./include/cute/atom/copy_traits_sm100.hpp) utility method to ease creation of tiled copies for tmem copy atoms.
  - Support for [new variants of LDSM on Blackwell](./include/cute/atom/copy_traits_sm100.hpp) via CuTe Copy atoms.
* Support for new CUTLASS building blocks specifically for Blackwell SM100 architecture:
  - Various narrow precision [FP4, FP6, and FP8](./include/cutlass/exmy_base.h) formats as well as their [block-scaled variants NVFP4, MXFP4, MXFP6, and MXFP8](./include/cutlass/float_subbyte.h)
  - [Pipelines that implement Blackwell specific synchronization](./include/cutlass/pipeline/sm100_pipeline.hpp).
  - [Cluster launch control API supporting preferred and fallback cluster shapes](./include/cutlass/cluster_launch.hpp).
  - Data types including NVFP4, MXFP4, MXFP6, and MXFP8 and all their supported element and scale factor types.
  - Tile schedulers using [Blackwell's Cluster Launch Control (CLC) feature](./media/docs/cpp/blackwell_cluster_launch_control.md) to implement dynamic persistence scheduling for [GEMMs](./include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp), and [stream-K](./include/cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp).
  - Extensions to testbeds and reference check code for unit tests and CUTLASS profiler.
* Full support for Blackwell SM100 kernels in CUTLASS 3.x API:
  - [Blackwell specific kernel layers](./include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp) that
    + Implement a new warp-specialization recipe tuned specifically for Blackwell SM100 architecture.
    + Leverage all the new features such as CLC based tile scheduling, preferred cluster, and TMEM based double buffering of accumulators.
    + Support stream-K load balancing for all kernel types everywhere via composable scheduler support.
  - Blackwell collective mainloops that target the TCGen05 MMA instructions (both SS and TS) for
    * [Non-block scaled data types without support for pointer array and grouped GEMM with TMA](./include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp)
    * [Non-block scaled data types with support for pointer array and grouped GEMM with TMA](./include/cutlass/gemm/collective/sm100_mma_array_warpspecialized.hpp)
    * [Block scaled data types without support for pointer array and grouped GEMM with TMA](./include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp)
    * [Block scaled data types with support for pointer array and grouped GEMM with TMA](./include/cutlass/gemm/collective/sm100_blockscaled_mma_array_warpspecialized.hpp)
  - Blackwell [collective mainloop for convolution kernels](./include/cutlass/conv/collective/sm100_implicit_gemm_umma_warpspecialized.hpp) supporting non-block scaled data types for fprop, dgrad, and wgrad.
  - New [GEMM](./include/cutlass/gemm/dispatch_policy.hpp), [convolution](./include/cutlass/conv/dispatch_policy.hpp), and [epilogue](./include/cutlass/epilogue/dispatch_policy.hpp) dispatch policies for collectives, kernel layers, and builders.
  - [Blackwell epilogue that supports loading accumulators from `tmem`](./include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp) and [full set of EVT fusions]().
* CUTLASS library and profiler integration for block scaled data types for kernel emission, profiling, and verification.
  - Support for preferred and fallback cluster shapes via profiler command line arguments parsing to set dynamic cluster shapes.
  - Support for dynamic datatypes by parsing profiler via profiler command line arguments parsing to set dynamic datatype setting in TCGen05 MMA instruction descriptors.
  - Support for mixed input GEMM kernels on Hopper in the profiler.
* New CUTLASS profiler flag `use-cuda-graphs` to reduce overheads when benchmarking launch-bound kernels.
* A new 3.x version of grouped GEMM to the CUTLASS library and generates kernels for Hopper and Blackwell. Now grouped GEMM support is enabled in the CUTLASS profiler (`./cutlass_profiler --operation=GroupedGemm --help` for details).
* Set of examples that demonstrate the usage of the 3.x API for targeting Blackwell SM100 architecture:
  - [Basic FP16 and FP8 GEMMs with minimal changes from Hopper examples](./examples/70_blackwell_gemm/), demonstrating ease of migration for off the shelf kernels using the 3.x collective builder API.
  - GEMM with [opt-in collective builder schedules showcasing available recipes](./examples/71_blackwell_gemm_with_collective_builder/71_blackwell_gemm_with_collective_builder.cu) for Blackwell.
  - Block scaled data type GEMMs targeting Blackwell's native block scaled Tensor Cores:
    + [NVFP4 inputs with BF16 output](./examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu)
    + [NVFP4 inputs with NVFP4 output](./examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu)
    + [Mixed MXFP8 and MXFP6 inputs with BF16 output](./examples/72_blackwell_narrow_precision_gemm/72c_blackwell_mixed_mxfp8_bf16_gemm.cu)
  - GEMM example demonstrating [Blackwell's new preferred cluster support via dynamic cluster shapes](./examples/73_blackwell_gemm_preferred_cluster/blackwell_gemm_preferred_cluster.cu) for increased occupancy.
  - [GEMM with CLC based StreamK scheduler for load balancing](./examples/74_blackwell_gemm_streamk/blackwell_gemm_streamk.cu).
  - Grouped GEMM for [vanilla FP8 data inputs](./examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm.cu) and [NVFP4 block scaled inputs](./examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm_block_scaled.cu).
  - Convolution kernels for [fprop](./examples/76_blackwell_conv/76_blackwell_conv_fprop.cu), [dgrad](./examples/76_blackwell_conv/76_blackwell_conv_dgrad.cu), and [wgrad](./examples/76_blackwell_conv/76_blackwell_conv_wgrad.cu).
  - [Fused multi-head attention fprop kernel](./examples/77_blackwell_fmha/77_blackwell_fmha.cu) supporting fp16/bf16/fp8 data types across head dims of 32,64, and 128.
  - A new BF16x9 GEMM [kernel](./examples/78_blackwell_emulated_bf16x9_gemm/78_blackwell_emulated_bf16x9_gemm.cu) that emulates FP32 GEMM (SGEMM) using BF16 operations.
* Set of examples that demonstrate the usage of the 3.x API for targeting Hopper architecture:
  - A set of new [Hopper grouped GEMM kernels](./examples/69_hopper_mixed_dtype_grouped_gemm/) that support mixed A and B datatypes.
  - A new [Hopper FP8 GEMM with groupwise scaling](./examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling.cu).
* Documentation updates:
  - [Quickstart - instantiating a Blackwell block-scaled GEMM](./media/docs/cpp/quickstart.md#instantiating-a-blackwell-gemm-kernel).
  - Detailed [Blackwell block-scaled GEMM functionality documentation](./media/docs/cpp/blackwell_functionality.md)
  - A new [functionality documentation](./media/docs/cpp/functionality.md) specifically for 3.x API comprehensively documenting all supported kernel types, data types, kernel features, minimum CUDA tookit support etc for 3.x supported architectures.
  - Updates to [compatibility](./README.md#compatibility) section regarding supported compilers, operating systems, CUDA Toolkits, Hardware Architectures, and [Target Architecture](./README.md#Target-Architecture).
  - Updates to [profiler documentation](./media/docs/cpp/profiler.md) for testing mixed input GEMM kernels on Hopper.

## [3.7.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.7.0) (2025-01-11)
- [Hopper blockwise scaling FP8 GEMM](./examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu) uses 2D scaling tensor, assigning one value per threadblock.  This allows a finer-grained scaling to be applied for each output tile per gemm-k iteration. The operands and scaling tensors are loaded from global memory to shared memory using TMA and cp_async, respectively. The scaling is applied inside the mainloop.  Details with figures are [here](https://github.com/NVIDIA/cutlass/pull/1932#issue-2645398439).
- [Distributed GEMM](./examples/65_distributed_gemm/65_distributed_gemm.cu) is a new (experimental) API which can turn existing CUTLASS GEMM kernels into pipelined Tensor Parallel GEMMs that run efficiently on NVLink-based network of GPUs. Its pipelining schedules can hide most of the communication behind computation, and relies on point-to-point communication, which can simply use CUDA runtime's peer device access feature. It also utilizes remote TMA loads and memcopies with CUDA graphs to handle communication primarily through the Copy Engine, leaving all SMs free for Hopper's persistent kernels.  For more details you can refer to the [DistGEMM blog post](https://blog.shi-labs.com/distributed-gemm-88be6a481e2b).
- Improved persistent grid launch for Hopper kernels with large cluster sizes (>= size of 4) using the new `make_kernel_hardware_info` API as shown in [example 48](./examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu).
- Enabled high precision accumulation for Hopper FP8 Sparse GEMM.
- Potential API breaking changes:
  + Fix `cute::UniversalCopy` for type safety.
  + No longer implicitly select `cute::SM80_CP_ASYNC_*` based on input tensors. This avoids implicit downstream synchronization requirements. To use `SM80_CP_ASYNC`, users must explicitly select the appropriate CopyAtom.
  + Fix `cute::SM80_CP_ASYNC_CACHEALWAYS`, `cute::SM80_CP_ASYNC_CACHEGLOBAL`, `cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL`, `cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL` to avoid implicitly selecting `ZFILL` behavior on predication.
  + Remove `cute::copy_vec<T>` in favor of `cute::copy_aligned` and `cute::copy(AutoVectorizingCopyWithAssumedAlignment<NumBits>,...)`.
  + A refactor of default epilogue struct `DefaultEpilogue` [API](./include/cutlass/epilogue/collective/default_epilogue.hpp) to avoid reading non-void `ElementC` value for `ElementC = void` kernel.
- New CUTLASS profiler flags: `profiling-duration`, `min-iterations`, and `kernels-file` documented in [profiler.md](./media/docs/cpp/profiler.md#cutlass-profiler).
- Various improvements and fixes from the community and CUTLASS team. Thanks to everyone who submitted PRs!
- Optimal code generation with CUDA toolkit versions 12.6.

## [3.6.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.6.0) (2024-10-03)

- [Hopper structured sparse GEMM](./examples/62_hopper_sparse_gemm/62_hopper_sparse_gemm.cu).
  + [FP16](./test/unit/gemm/device/sm90_sparse_gemm_f16_f16_f32_tensor_op_f32.cu)
  + [FP8](./test/unit/gemm/device/sm90_sparse_gemm_f8_f8_f32_tensor_op_f32.cu)
  + [INT8](./test/unit/gemm/device/sm90_sparse_gemm_s8_s8_s32_tensor_op_s32.cu)
  + [TF32](./test/unit/gemm/device/sm90_sparse_gemm_tf32_tf32_f32_tensor_op_f32.cu)
- A refactor to the CUTLASS 3.x convolution `kernel::ConvUniversal` [API](./include/cutlass/conv/kernel/sm90_implicit_gemm_tma_warpspecialized.hpp) to bring it in line with `gemm::GemmUniversal`. Now the 3.x convolution API is no longer considered as a beta API.
- [An improved mixed input GEMM](./examples/55_hopper_mixed_dtype_gemm/README.md) and a [lookup table implementation](./examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_fp8_gemm.cu) for `INT4`x`FP8` scale-only mode.
- [EVT nodes for Top-K selection and softmax](./include/cutlass/epilogue/fusion/sm90_visitor_topk_softmax.hpp) and [GEMM example using those](./examples/61_hopper_gemm_with_topk_and_softmax/61_hopper_gemm_with_topk_and_softmax.cu).
- [Programmatic Dependent Launch](./include/cutlass/arch/grid_dependency_control.h) (PDL) that leverages a new Hopper feature to speedup two back-to-back kernels, and its corresponding [documentations](./media/docs/cpp/dependent_kernel_launch.md).
- [A new debugging tool, synclog](./include/cutlass/arch/synclog.hpp), for dumping out all synchronization events from within a kernel to a file. Please see [synclog documentation](./media/docs/cpp/utilities.md#debugging-asynchronous-kernels-with-cutlasss-built-in-synclog-tool) for details.
- A new TMA-enabled [epilogue](./include/cutlass/epilogue/collective/sm90_epilogue_array_tma_warpspecialized.hpp) for grouped GEMM that brings significant performance improvement, as well as its EVT support.
- A SIMT-enabled pointer-array [epilogue](./include/cutlass/epilogue/collective/sm70_epilogue_vectorized_array.hpp).
- A new [Ping-Pong kernel schedule for Grouped GEMM](./include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_pingpong.hpp) and some other optimizations.
- [A new instantiation strategy for CUTLASS profiler kernels](./python/cutlass_library/sm90_shapes.py) along with [improved documentation for instantiation level in CUTLASS profiler](./media/docs/cpp/profiler.md#instantiating-more-kernels-with-hopper).
- A new hardware support for comparisons and computations of [`cutlass::bfloat16_t`](./include/cutlass/bfloat16.h)
- Fixed use of isnan on Windows for [`half_t`](./test/unit/core/functional.cu).
- Various improvements and fixes from the community and CUTLASS team. Thanks to everyone who submitted PRs!
- Optimal code generation with CUDA toolkit versions 12.6.

## [3.5.1](https://github.com/NVIDIA/cutlass/releases/tag/v3.5.1) (2024-07-25)

- [Minimal SM90 WGMMA + TMA GEMM example in 100 lines of code](./examples/cute/tutorial/wgmma_sm90.cu)
- [Exposure of L2 `cache_hint`s in TMA copy atoms](./include/cute/arch/copy_sm90_tma.hpp#L48)
- Exposure of raster order and tile swizzle extent in [CUTLASS library profiler](./media/docs/cpp/profiler.md#GEMM), and
[example 48](./examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu).
- [TMA store based and EVT supported epilogues](./include/cutlass/epilogue/collective/sm90_epilogue_array_tma_warpspecialized.hpp) for [Hopper pointer array batched kernels](./test/unit/gemm/device/sm90_gemm_f16_f16_f16_tensor_op_f32_ptr_array.cu).
- A new [`GemmSparseUniversal` API for CUTLASS 2.x Ampere kernels](./include/cutlass/gemm/device/gemm_sparse_universal.h) to enable serial and parallel split-k for sparse tensor cores and new tiny tile sizes to better support LLM inferrence:
  + [FP16 TN](./test/unit/gemm/device/gemm_f16t_f16n_f32t_tensor_op_f32_sparse_sm80.cu#L269-L393) and [NT](./test/unit/gemm/device/gemm_f16n_f16t_f32t_tensor_op_f32_sparse_sm80.cu#L269-L411).
  + [int8 TN](./test/unit/gemm/device/gemm_s8t_s8n_s32t_tensor_op_s32_sparse_sm80.cu#L264-L452).
  + [int4 TN](./test/unit/gemm/device/gemm_s4t_s4n_s32t_tensor_op_s32_sparse_sm80.cu#L264-L452).
  + [FP32 TN](./test/unit/gemm/device/gemm_f32t_f32n_f32t_tensor_op_f32_sparse_sm80.cu#L427-L642) and [NT](./test/unit/gemm/device/gemm_f32n_f32t_f32t_tensor_op_f32_sparse_sm80.cu#L427-L456).
- [CUDA host adapter](./include/cutlass/cuda_host_adapter.hpp) extensions to support TMA descriptor construction driver APIs.
- Inclusion of more [Hopper fprop, dgrad, and wgrad convolution kernels in CUTLASS library and profiler](./python/cutlass_library/generator.py).
- Support for residual add (beta != 0) in convolution kernels.
- A new convolution [epilogue](./examples/16_ampere_tensorop_conv2dfprop/ampere_tensorop_conv2dfprop.cu#L269) for CUTLASS 2.x to support non-packed NHWC output.
- A refactor of [include files throughout CUTLASS core directories](./include/cutlass/gemm/collective/collective_mma_decl.hpp) to reduce circular dependencies and [tests to guard against them](./test/self_contained_includes/CMakeLists.txt).
- [A guide for setting up VSCode to work well with CUTLASS](./media/docs/cpp/ide_setup.md) and [expanded code style guide](./media/docs/cpp/programming_guidelines.md).
- Better support for MSVC as a host compiler.
- Many performance optimizations, improvements, and bug fixes including fixes for FlashAttention-2.
- Optimal code generation with CUDA toolkit versions 12.4 and 12.5u1.

## [3.5.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.5.0) (2024-04-09)

- Implicit GEMM Convolutions targeting Hopper SM90A via WGMMA + [TMA im2col](./include/cute/atom/copy_traits_sm90_im2col.hpp)
  + Native implementation in CUTLASS 3.x using CuTe, mirroring the [same design hierarchy as that of GEMMs](./media/docs/cpp/gemm_api_3x.md).
  + Support for 1D, 2D, and 3D convolutions in a [rank-agnostic fashion](./include/cutlass/conv/convnd_problem_shape.hpp).
  + Support for [Fprop](./test/unit/conv/device_3x/fprop/sm90_conv3d_fprop_implicit_gemm_s8_s8_s32_tensorop_s32.cu), [Dgrad](./test/unit/conv/device_3x/dgrad/sm90_conv2d_dgrad_implicit_gemm_f16_f16_f32_tensorop_f16.cu), and [Wgrad](./test/unit/conv/device_3x/wgrad/sm90_conv1d_wgrad_implicit_gemm_f16_f16_f32_tensorop_f16.cu) algorithms
  + [CUTLASS profiler support](./python/cutlass_library/conv3x_emitter.py) for 2D and 3D convolutions implemented via the 3.x API.
  + NOTE: this is a beta release. Further updates to CUTLASS will include major performance improvements, feature enablement, and possible breaking changes to the API until 3.7 release. Your feedback is welcome on the design!
- Support for [Ada (SM89) FP8 tensor cores via the 2.x API](./examples/58_ada_fp8_gemm/ada_fp8_gemm.cu). Requires CUDA 12.4 or newer.
- [Ampere gather/scatter convolution example](./examples/59_ampere_gather_scatter_conv/README.md) in CuTe and CUTLASS 3.x
  + Showcasing how custom kernels can be written and optimized using CUTLASS 3.x and CuTe and the general strategy for implementing convolutions as specializations of GETTs.
  + Implementation of a coarse grained sparse gather/scatter kernel achieving peak performance on Ampere class tensor cores.
- 32x and 16x tile sizes are added to CUTLASS 2.x to improve the performance of narrow-tall and wide-short matrices.
  + [Ampere FP16 TN](./test/unit/gemm/device/gemm_f16t_f16n_f16t_tensor_op_f32_sm80.cu) and [NT](./test/unit/gemm/device/gemm_f16n_f16t_f16t_tensor_op_f32_sm80.cu#L227-L301), [Ampere INT8 TN](./test/unit/gemm/device/gemm_s8t_s8n_s8t_tensor_op_s32_sm80.cu#L392-L1342), [Ampere INT4 TN](./test/unit/gemm/device/gemm_s4t_s4n_s4t_tensor_op_s32_sm80.cu#L372-L934).
  + [Turing FP16 TN](./test/unit/gemm/device/gemm_f16t_f16n_f16t_tensor_op_f32_sm75.cu#L55-L394), [Turing INT8 TN](./test/unit/gemm/device/gemm_s8t_s8n_s8t_tensor_op_s32_sm75.cu#L166-L537), [Turing INT4 TN](./test/unit/gemm/device/gemm_s4t_s4n_s4t_tensor_op_s32_sm75.cu#L310-L564).
- Updates to CuTe documentation for [`cute::Tensor<>`](./media/docs/cpp/cute/03_tensor.md), [MMA atoms](./media/docs/cpp/cute/0t_mma_atom.md), and an overhauled [CuTe GEMM tutorial series](./examples/cute/tutorial).
- Extensions to CuTe to support [L2 prefetching](./include/cute/algorithm/prefetch.hpp) and [TMA store+reductions](./include/cute/arch/copy_sm90_tma.hpp#L1337).
- Remove C++11 requirement on a few CUTLASS 2.x API header files. All CUTLASS files now require C++17.
- Fixes to greatly reduce build warnings.
- Updates and bugfixes from the community (thanks!)

## [3.4.1](https://github.com/NVIDIA/cutlass/releases/tag/v3.4.1) (2024-02-14)

- Statically available [CUTLASS Version macros](./include/cutlass/version.h) that allow for handling API changes between CUTLASS releases on the users' side.
- Improvements for Hopper [Group-GEMMs](./examples/57_hopper_grouped_gemm) and [Pointer-Array Batched GEMMs](./examples/56_hopper_ptr_array_batched_gemm).
- Updates and bugfixes from the community (thanks!).

## [3.4.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.4.0) (2024-01-12)
* Expanded [Mixed-input Hopper GEMMs](./examples/55_hopper_mixed_dtype_gemm) support covering {16-bit, 8-bit} x {8-bit, 4-bit} input types with fast numerical converters and group scaling factors.
* Performance improvements to [Mixed-input Hopper GEMMs](./examples/55_hopper_mixed_dtype_gemm)
* Beta release of [Pointer-Array Batched GEMMs](./examples/56_hopper_ptr_array_batched_gemm) now available on Hopper GPUs utilizing TMA and WGMMA (requires CUDA 12.3 or above).
* Beta release of [Group-GEMM](./examples/57_hopper_grouped_gemm) utilizing TMA and WGMMA (requires CUDA 12.3 or above).
* [Ampere Sparse GEMM](./examples/15_ampere_sparse_tensorop_gemm/ampere_sparse_tensorop_gemm_with_visitor.cu) supports Epilogue Visitor Tree (EVT) now.
* NamedBarriers usability improvement and list of [ReservedNamedBarriers](./include/cutlass/arch/barrier.h) has been officially released.
* Improved [CuTe documentation](./media/docs/cpp/cute/) including improved clarity and depth of [Quickstart](./media/docs/cute/00_quickstart.md), [CuTe Layout](./media/docs/cpp/cute/01_layout.md), and [CuTe Layout Algebra](./media/docs/cpp/cute/02_layout_algebra.md). Associated code comments, post-conditions, and details in [CuTe Core Unit Tests](./test/unit/cute/core/) also improved.

## [3.3](https://github.com/NVIDIA/cutlass/releases/tag/v3.3.0) (2023-10-31)
* [Mixed-input Hopper GEMMs](./examples/55_hopper_mixed_dtype_gemm) support covering 16-bit x 8-bit input operand types.
* [Mixed-input Ampere GEMMs](https://github.com/NVIDIA/cutlass/pull/1084) with support for canonical layouts (TN). The implementation supports upcast on operandB {fp16, bf16} x {s8, u8}, and upcast on operandA {s8, u8} x {fp16, bf16}.
* [Copy Async based Hopper GEMMs](./test/unit/gemm/device/sm90_gemm_bf16_bf16_bf16_alignx_tensor_op_f32_warpspecialized_cooperative.cu) - which support lower than 16B aligned input tensors.
* Kernel schedules and Builder support for mixed precision and Copy Async GEMMs with < 16B aligned input tensors.
* Profiler support for lower-aligned Hopper GEMMs.
* Performance Improvements to [Scatter-Gather Hopper Example](./examples/52_hopper_gather_scatter_fusion).
* Sub-Byte type fixes and improvements.
* EVT Support for RELU with Aux bitmap tensor store (used in dRELU). See [SM90 EVT fusions](./include/cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp) for details.
* Fusion support for backprop fusions including drelu, dgelu, and dbias.
* Support for void-C kernels and SM80 mixed-input GEMMs in the CUTLASS Python interface

## [3.2.2](https://github.com/NVIDIA/cutlass/releases/tag/v3.2.2) (2023-10-25)
* Minor patch for issue/1138

## [3.2.1](https://github.com/NVIDIA/cutlass/releases/tag/v3.2.1) (2023-09-22)
* Python support SM90 Epilogue Visitor Tree (EVT) on top of the C++ support released in 3.2.0.
* SM80 EVT support in C++ and Python.
* Other SM90 epilogue improvements.
* Splitting CUTLASS library into smaller units based on operation, arch and datatypes. See [1105](https://github.com/NVIDIA/cutlass/discussions/1105) for details.
* Making `tools/library/scripts` packageable - `tools/library/scripts` is now moving to `python/cutlass_library`. See the Python [README](./python/README.md) for details.
* SM90 TF32 kernel improvements for all layouts.
* SM90 rasterization direction support in the CUTLASS profiler.
* Improvement for CUTLASS profiler build times.
* Remove Python-C++ bindings.

## [3.2.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.2.0) (2023-08-03)

* New warp-specialized persistent FP8 GEMM kernel [kernel schedules](./include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp) and [mainloops](./include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8.hpp)  targeting Hopper architecture that achieve great performance with TMA, WGMMA, and threadblock clusters. An example showcasing [Hopper warp-specialized FP8 GEMMs](./examples/54_hopper_fp8_warp_specialized_gemm). FP8 GEMMs come with a fast accumulation mode. When enabled, problem execution might be faster but at the cost of lower accuracy because intermediate results will not periodically be promoted to a higher precision.
* New [Epilogue Visitor Tree (EVT)](./examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu) support for Hopper TMA epilogues. EVTs allows for user-defined customized epilogue fusion patterns without having to write a new epilogue.
* [Stream-K](./include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp) feature for Hopper. Note that this is only a functional implementation of stream-K, and should not be used for performance comparison. Optimizations are expected in a future release.
* Improved CTA rasterization and support for CTA swizzling for Hopper kernels using the [Tile Scheduler](./include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp).
* Improved performance for [warp-specialized TensorFloat-32 (TF32) GEMM kernels](test/unit/gemm/device/sm90_gemm_tf32_tf32_f32_tensor_op_f32_gmma_rs_cluster_warpspecialized.cu) targeting Hopper TMA.
* [Hopper GEMM+Permute](./examples/53_hopper_gemm_permute/53_hopper_gemm_permute.cu), an example of fusing tensor reordering (permutation) with GEMM mainloop or epilogue.
* New CUTLASS 2D Convolution Python interface. New [example](./examples/python/03_basic_conv2d.ipynb) here.
* Support for Windows (MSVC) builds. Tested with Visual Studio 2019 v16.11.27 on Windows 10.0.
* Optimal performance using [**CUDA 12.2u1**](https://developer.nvidia.com/cuda-downloads)
* Updates and bugfixes from the community (thanks!)

## [3.1.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.1.0) (2023-04-14)
* New CUTLASS Python interface that aims to provide an ease-of-use interface for instantiating, emitting, compiling, and running CUTLASS kernels via Python. More details [here](./python/README.md) and new [examples](./examples/python).
* New [efficient epilogues](test/unit/gemm/device/sm90_gemm_f16_f16_f16_tensor_op_f32_cluster_warpspecialized_cooperative.cu#L783) using TMA for Hopper.
* Support for [fused epilogues](test/unit/gemm/device/sm90_gemm_f16_f16_f16_tensor_op_f32_cluster_warpspecialized_cooperative_bias_elementwise.cu), such Bias, ReLU and GELU, using the new efficient epilogues.
* New [warp-specialized TensorFloat-32 (TF32) GEMM kernels](test/unit/gemm/device/sm90_gemm_tf32_tf32_f32_tensor_op_f32_gmma_rs_cluster_warpspecialized.cu) targeting Hopper TMA.
* New [*warp-specialized persistent cooperative*](./include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp) kernel design that allows for larger tile sizes and improves performance on Hopper.
* An [example](./examples/51_hopper_gett) showcasing GEMM-Like Tensor-Tensor Contraction (GETT) capability on Hopper.
* Epilogue builders. Similar to mainloop builders (see [example 49](./examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu)), epilogue builders aim to generate the best-possible epilogue while exposing incremental opt-ins for greater customization.
* Profiler support for overriding kernel and epilogue builder auto schedules for 3.x API kernels, allowing specific policies to be run in the CUTLASS profiler.
* Performance optimizations for the [*warp-specialized persistent ping-pong*](./include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp) kernel.
* Changes to the [GEMM API 3.x](./media/docs/cpp/gemm_api_3x.md), involving the host-facing arguments and the underlying `Params` structs.
* [FMHA Backward Pass](./examples/41_fused_multi_head_attention/fused_multi_head_attention_backward.cu) from Meta xFormers.
* [Streamk GEMM with Broadcast](./examples/47_ampere_gemm_universal_streamk/ampere_gemm_universal_streamk_broadcast.cu) enables epilogue broadcast with StreamK GEMM.
* [Batched B2B GEMM](./examples/13_two_tensor_op_fusion) now can run multiple Back-to-Back GEMM with the same problem size in parallel.
* [Batched Strided GEMV](test/unit/gemm/device/gemv.cu) support both row major and column major input matrix.
* [Permute + GEMM fusion](./examples/39_gemm_permute) can fuse Permute with following GEMM now.  Before, we only support fusing GEMM with Permute in the epilogue.
* [Row Broadcast](./include/cutlass/epilogue/threadblock/predicated_tile_iterator_row_broadcast.h) can be fused in the epilogue.
* The GitHub branch is renamed from `master` to `main` in this release.
* Optimal performance using [**CUDA 12.1**](https://developer.nvidia.com/cuda-downloads)
* Updates and bugfixes from the community (thanks!)

## [3.0.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.0.0) (2023-01-23)
* [CuTe](./media/docs/cpp/cute/00_quickstart.md), a [new core library and backend](./include/cute) for CUTLASS 3.0 that defines a single Layout vocabulary type and an associated algebra of layouts for a much more expressive and composable abstraction for tensors, sets of parallel agents, and operations by said agents on tensors.
* [A new conceptual operation hierarchy](./media/docs/cpp/cutlass_3x_design.md) that replaces the architecture-centric hierarchy of CUTLASS 2.x and [documentation for CUTLASS 3.0's GEMM API changes](./media/docs/cpp/gemm_api_3x.md).
* Strict API backwards compatibility that exposes both 2.x and 3.x API kernels through the same [`device::GemmUniversalAdapter`](./include/cutlass/gemm/device/gemm_universal_adapter.h) and [`kernel::GemmUniversal`](./include/cutlass/gemm/kernel/gemm_universal.hpp) types, allowing users to include both APIs in the same translation units. More information can be found in the [3.x backwards compatibility section](./media/docs/cpp/cutlass_3x_backwards_compatibility.md).
* Updates to [Functionality](./media/docs/cpp/functionality.md) which directs users on which kernels are supported via CUTLASS-2 and CUTLASS-3.
* Updates to [Compatibility](./README.md#compatibility) Section regarding supported compilers, operating systems, CUDA Toolkits, Hardware Architectures and [Target Architecture](./README.md#Target-Architecture).
* New warp-specialized GEMM [kernel schedules](./include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp) and [mainloops](./include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp) targeting Hopper architecture that achieve great performance with TMA, WGMMA, and threadblock clusters.
* Extensions to CUTLASS profiler to support threadblock cluster shapes in library and profiler tile configurations.
* [CUTLASS library integration](./tools/library/src/gemm_operation_3x.hpp) for 3.x API kernels built through the new `CollectiveBuilder` API, enabling CUTLASS profiler.
* Support for [Hopper GEMMs](./examples/48_hopper_warp_specialized_gemm) through the new 3.0 API with CuTe-based exposure of the Hopper [Tensor Memory Accelerator](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor) and [WGMMA Tensor Core](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions) features.
* Set of examples that demonstrate the usage of the new 3.0 API to easily build GEMM kernels targeting Hopper: examples [48](./examples/48_hopper_warp_specialized_gemm), [49](./examples/49_hopper_gemm_schedules_with_collective_builder), and [50](./examples/50_hopper_gemm_with_epilogue_swizzle).

## [2.11.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.11.0) (2022-11-19)
* [Stream-K](./examples/47_ampere_gemm_universal_streamk), which is a new general way to do split-K.  It can not only improve performance, but can also significantly reduce the number of tile sizes that need to be profiled to find the best one.
* [Fused multi-head attention Kernel](./examples/41_fused_multi_head_attention).  It has two variants: one uses batched GEMM for the fixed sequence length, and the other one uses group GEMM for the variable sequence length.  Both versions just need one kernel.
* [Dual GEMM](./examples/45_dual_gemm), which can fuse A x B and A x C into one kernel. Two GEMMs has no producer-consumer dependency.
* Hopper improves [double precision matrix multiplication](./test/unit/gemm/device/gemm_f64n_f64t_f64t_tensor_op_f64_sm90.cu) by 2x compared to Ampere at iso-clocks. It is supported since CUDA 11.8.
* [BLAS3](./test/unit/gemm/device/hemm_cf64_cf64_cf64_tensor_op_f64_sm90.cu) functions with Hoppers new double precision matrix multiplication instructions.
* [ELL Block Sparse GEMM](./examples/43_ell_block_sparse_gemm), which uses an [ELL matrix](https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/) to describe the sparsity of A matrix.  B and output matrices are still dense. The block size can be arbitary.
* Optimized [Group Conv](./examples/42_ampere_tensorop_group_conv) for SingleGroup mode, which requires that the output channel per group is a multiple of Threadblock tile N.
* [Optimized DepthWise Conv](./examples/46_depthwise_simt_conv2dfprop/depthwise_simt_conv2dfprop.cu).  Two new modes are added
  * [kOptimized](./test/unit/conv/device/depthwise_conv2d_fprop_direct_conv_f16nhwc_f16nhwc_f16nhwc_simt_f16_sm60.cu) - use direct conv to compute instead of implicit GEMM.
    *  The restrictions are: 1) input ,output channel and group number should be multiple of (128 / sizeof(input element)). 2) The input filter size should be the same as the template parameter configuration.
  * [kFixedStrideDilation](./test/unit/conv/device/depthwise_conv2d_fprop_direct_conv_fixed_stride_dilation_f16nhwc_f16nhwc_f16nhwc_simt_f16_sm60.cu) - which puts stride and dilation into templates to further improve the performance. In this mode, kernel persistents some inputs into register to squeeze more performance, so large filter/stride/dilation is not recommanded.
    * The restrictions are: 1) input, output channel and group number should be multiple of (128 / sizeof(input element)). 2) input filter size, stride, dilation should same as the template parameter configuration.
* [Scripts](./examples/44_multi_gemm_ir_and_codegen) to fuse multiple back-to-back GEMM.  Its implementation was discussed in a GTC'22 Spring [talk](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41606/).
* [FP8 data type definition](./include/cutlass/float8.h) and [conversion routines](./include/cutlass/numeric_conversion.h#L1274-2115).
* Updates and bugfixes from the community (thanks!).  Big shout out to Meta's [xFormers](https://github.com/facebookresearch/xformers).

* **Deprecation announcement:** CUTLASS plans to deprecate the following:
  * Maxwell and Pascal GPU architectures
  * Ubuntu 16.04
  * CUDA 10.2

## [2.10.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.10.0) (2022-08-23)
* [CUTLASS Python](./examples/40_cutlass_py) now supports GEMM, CONV, Group GEMM for different data types as well as different epilogue flavours.
* Optimizations for CUTLASS's [Grouped GEMM](./examples/24_gemm_grouped/gemm_grouped.cu) kernel.  Threadblock scheduling part is improved.  Some computation can be moved to the host side if applicable.  [Grouped Syr2k](./examples/38_syr2k_grouped/syr2k_grouped.cu) kernels are added, too.
* Optimizations for [GEMM+Softmax](./examples/35_gemm_softmax).  All the reduction computation is fused into the previous GEMM.  More template arguments are provided to fine tune the performance.
* [Grouped GEMM for Multihead Attention](./examples/41_multi_head_attention).  This general group gemm based MHA does not require the sequence length of all GEMMs to be the same which makes it most useful for natural language processing.
* [GEMM + Layer norm fusion for Ampere](./examples/37_gemm_layernorm_gemm_fusion/) splits the layernorm into two parts and both of them can be fused into the GEMMs before and after separately.  In addition to use square sum to compute variance of layernorm, [Shift-K](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data) is provided if square sum raise numerical issues.
* [GEMM Epilogue Permutation Fusion](./examples/39_gemm_permute) can apply user provided permutation layout mapping in the GEMM epilogue.
* [Grouped convolution targeting implicit GEMM](test/unit/conv/device/group_conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_sm80.cu) introduces the first group convolution implementation to CUTLASS.  It is an Analytical implementation, not an Optimized.  The restrictions are: 1) input and output channel number should be multiple of group number. 2) split-K is not supported.  The implementation has 2 modes:
  * kSingleGroup: output channel per group is multiple of Threadblock tile N.
  * kMultipleGroup: Threadblock tile N is multiple of output channel per group.
* [Depthwise separable convolution](test/unit/conv/device/depthwise_conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_simt_f16_sm60.cu) introduces the first depthwise convolution which is also Analytical for now.  The restrictions are: 1) SIMT only 2) No split-K 3) input channel equals to output channel equals to group number.
* Standalone [Layernorm](./tools/util/include/cutlass/util/device_layernorm.h) and [Pooling](./tools/util/include/cutlass/util/device_nhwc_pooling.h) kernels.
* [Back-to-back GEMM/CONV](./examples/13_two_tensor_op_fusion) relaxes the requirement that the first GEMM K dimension needs to be the multiple of Threadblock Tile K dimension.
* Optimal performance using [**CUDA 11.6u2**](https://developer.nvidia.com/cuda-downloads)
* Updates and bugfixes from the community (thanks!)

## [2.9.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.9.0) (2022-04-21)

* [First layer Convolution kernels](./test/unit/conv/device/conv2d_fprop_fixed_channels_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_sm80.cu) specialized for small channel counts and reduced alignment
  * [Few channels](./include/cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_few_channels.h) specialization for reduced alignment capabilities
  * [Fixed channels](./include/cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_fixed_channels.h) further specialized when channel count perfectly matches the access vector size
  * [Unit tests](./test/unit/conv/device/conv2d_fprop_few_channels_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_sm80.cu)
  * [Python-based instance emitter](./python/cutlass_library/generator.py) in the CUTLASS Library and support in the Profiler
* [BLAS3](https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-3-function-reference) operators accelerated by Tensor Cores
  * Supported types: f32, cf32, f64, cf64, tf32x3, complex tf32x3
  * [HERK](./test/unit/gemm/device/her2k_cf32h_cf32n_tensor_op_fast_f32_sm80.cu) with [emitter](./python/cutlass_library/rank_k_operation.py)
  * [SYRK](./test/unit/gemm/device/syrk_f32n_f32t_tensor_op_fast_f32_sm80.cu) with [emitter](./python/cutlass_library/rank_k_operation.py)
  * [SYMM](./test/unit/gemm/device/symm_f32n_f32n_tensor_op_fast_f32_ls_sm80.cu) with [emitter](./python/cutlass_library/symm_operation.py)
  * [TRMM](./test/unit/gemm/device/trmm_f32n_f32t_f32t_tensor_op_fast_f32_ls_sm80.cu) with [emitter](./python/cutlass_library/trmm_operation.py)
  * [Unit tests](./test/unit/gemm/device/testbed_rank_k_universal.h)
* [CUTLASS Python](./examples/40_cutlass_py) demonstrating JIT compilation of CUTLASS kernels and a Python-based runtime using [CUDA Python](https://developer.nvidia.com/cuda-python)
  * [Python-based runtime](./tools/library/scripts/rt.py) interoperable with existing emitters
* [GEMM + Softmax example](./examples/35_gemm_softmax)
* [Gather and Scatter Fusion with GEMM](./examples/36_gather_scatter_fusion) can gather inputs and scatters outputs based on indices vectors in the same GEMM kernel.
  * It can select random rows in a row major matrix.
  * It can select random columns in a column major matrix.
* [Back-to-back GEMM/CONV](./examples/13_two_tensor_op_fusion) fully supports buffering the first GEMM/CONV results in the shared memory for the latter one to use.  It can eliminate register spill when the tile size is big.  Additionally, bias vector add is supported in the first GEMM/CONV.
  * Supported kernels: GEMM and CONV.
  * Supported types: fp16 and int8.
  * Supported architectures: Turing and Ampere.
* [Transposed Convolution](./examples/34_transposed_conv2d) (a.k.a Deconvolution) support which reuses Dgrad implementation.
* [Utility functions](./tools/util/include/cutlass/util) that can pad NHWC and convert between NCHW and NHWC.
* [Small alignment implicit gemm](https://github.com/NVIDIA/cutlass/issues/242) support for Fprop/Dgrad/Wgrad so that padding is no longer mandated to use tensor cores in these kernels.
* Epilogue enhancement:
  * Eliminate bank conflicts in int8 tensor core kernels.
  * Half2 usage if epilogue compute type is fp16.
  * More activation functions: Silu, Hardswish, Leaky Relu.
  * New elementwise fusion pattern for [residual block](./include/cutlass/epilogue/thread/linear_combination_residual_block.h).
* [Group GEMM](./examples/24_gemm_grouped) thread block number calculation fix which helps to launch the intended number of threadblocks to fully occupy the GPUs.
* [Parallel GEMM splitk](https://github.com/NVIDIA/cutlass/pull/277) support in the CUTLASS profiler.
* Optimal performance using [**CUDA 11.6u2**](https://developer.nvidia.com/cuda-downloads)
* Updates and bugfixes from the community (thanks!)


## [2.8.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.8.0) (2021-11-19)

* **TF32x3:** emulated single-precision using Tensor Cores
  * 45+ TFLOPs on NVIDIA A100
  * [GEMM SDK example](./examples/27_ampere_3xtf32_fast_accurate_tensorop_gemm/27_ampere_3xtf32_fast_accurate_tensorop_gemm.cu) (real)
  * [COMPLEX GEMM SDK example](./examples/29_ampere_3xtf32_fast_accurate_tensorop_complex_gemm/29_3xtf32_complex_gemm.cu) (complex)
  * [Implicit GEMM Convolution SDK example](./examples/28_ampere_3xtf32_fast_accurate_tensorop_fprop/ampere_3xtf32_fast_accurate_tensorop_fprop.cu)
* **Mainloop fusion for Convolution:** convolution with fused per-channel scale-bias-relu
  * [Conv Fprop SDK example](./examples/25_ampere_fprop_mainloop_fusion/ampere_fprop_mainloop_fusion.cu)
  * [Conv WGrad SDK example](./examples/26_ampere_wgrad_mainloop_fusion/ampere_wgrad_mainloop_fusion.cu)
  * [cutlass::conv::device::ImplicitGemmConvolutionFusion](./include/cutlass/conv/device/implicit_gemm_convolution_fusion.h)
* **Grouped GEMM:** similar to batched GEMM with distinct problem size per group
  * [SDK example](./examples/24_gemm_grouped) with performance comparison with Batched Strided GEMM
  * [cutlass::gemm::device::GemmGrouped](./include/cutlass/gemm/device/gemm_grouped.h)
* [Implicit GEMM Convolution fusion](./examples/13_two_tensor_op_fusion/) supports staging 1st convolution's output accumulator in the shared memory on Turing. This allows more flexible warp tile sizes and less regsiter pressue.
* Optimal performance using [**CUDA 11.5**](https://developer.nvidia.com/cuda-downloads)
* Updates from the community (thanks!)

* **Deprecation announcement:** CUTLASS plans to deprecate the following:
  * Maxwell and Pascal GPU architectures
  * Ubuntu 16.04
  * CUDA 10.2

## [2.7.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.7.0) (2021-09-24)
  * Mainloop fusion for GEMM: [summation over A or B](./examples/23_ampere_gemm_operand_reduction_fusion/ampere_gemm_operand_reduction_fusion.cu)
  * [Strided DGRAD (optimized iterators)](./include/cutlass/conv/kernel/default_conv2d_dgrad.h)
  * [Half-precision GELU_taylor activation functions](./include/cutlass/epilogue/thread/activation.h#L196)
    * Use these when accumulation and epilogue compute types are all `cutlass::half_t`
  * Tuning and bug fixes to [fused GEMM + GEMM example](./examples/13_two_tensor_op_fusion/)
  * Support for smaller than 128b aligned Convolutions: [see examples](test/unit/conv/device/conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm80.cu#L272)
  * Caching of results to accelerate Convolution [unit tests](test/unit/conv/device/cache_testbed_output.h)
    * Can be enabled or disabled by running `cmake .. -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=OFF`
  * Corrections and bug fixes reported by the CUTLASS community
    * Thank you for filing these issues!

## [2.6.1](https://github.com/NVIDIA/cutlass/releases/tag/v2.6.1) (2021-09-03)
  * Arbitrary padding and striding for CUTLASS Strided DGRAD Convolution operator (Analytic Iterators)
  * Tuning for GEMMs fused with partial reductions
  * Corrections and bug fixes reported by the CUTLASS community
    * Thank you for filing these issues!

## [2.6.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.6.0) (2021-07-22)
  * Optimal performance when compiled with the [CUDA 11.4 Toolkit](https://developer.nvidia.com/cuda-toolkit)
    * Adopt the new L2 prefetch feature in [cp.async](./include/cutlass/arch/memory.h) and [global load](./include/cutlass/arch/memory_sm80.h)
  * Fused operators with GEMM and Convolution
    * [Fused broadcast in epilogue](test/unit/gemm/device/gemm_with_broadcast_f16n_f16n_f16n_tensorop_f32_sm75.cu)
    * [Fused partial reduction in epilogue](./test/unit/gemm/device/gemm_with_reduction_f16n_f16n_f16n_tensorop_f32_sm75.cu)
  * 64b tensor strides and leading dimensions support for GEMMs
  * Affine rank=2 matrix layouts
    * Row stride and column stride for matrices using [cutlass::layout::AffineRank2](./include/cutlass/layout/matrix.h)
    * Support [FP64 tensor core](./examples/18_ampere_fp64_tensorop_affine2_gemm/ampere_fp64_tensorop_affine2_gemm.cu) and SIMT GEMM.
  * [Batched GEMV](./test/unit/gemm/device/gemv.cu) preview implementation
  * [New strided Dgrad](test/unit/conv/device/conv2d_strided_dgrad_implicit_gemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_sm80.cu) implementation
    * Accelerates over previous implementation by cutting down redundant math by 4x
    * Support using new `Dy` and `w` analytic iterators and existing `cutlass::conv::device::ImplicitGemmConvolution` interface
  * Quaternion-valued GEMM and Convolution in single- and double-precision (targeting CUDA Cores)
    * Updates to [quaternion.h](./include/cutlass/quaternion.h) and [functional.h](./include/cutlass/functional.h)
    * SDK Example for [GEMM](./examples/21_quaternion_gemm/quaternion_gemm.cu) and [Convolution](./examples/22_quaternion_conv/quaternion_conv.cu)
    * [Unit tests for GEMM](./test/unit/gemm/device/simt_qgemm_nn_sm50.cu) and [Convolution](./test/unit/conv/device/conv2d_fprop_implicit_gemm_qf32nhwc_qf32nhwc_qf32nhwc_simt_f32_sm50.cu)
  * Many improvements to the epilogue.
    * Provide an [option](./include/cutlass/epilogue/threadblock/epilogue.h) to not fully unroll the epilogue to reduce the code size and improve the performance when using complicated elementwise operations
    * Performance improvement for FP16 tensor core kernels
    * Bug fixes
  * Enhanced Clang support and the combination of Clang 13 and CUDA 11.4 can build and run kernels from Pascal and Ampere.
  * Updated minimum CUDA Toolkit requirement to 10.2
    * [CUDA 11.4 Toolkit](https://developer.nvidia.com/cuda-toolkit) recommended
  * Corrections and bug fixes reported by the CUTLASS community
    * Thank you for filing these issues!

## [2.5.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.5.0) (2021-02-26)
  * Tensor reductions
    * _m_-to-_n_ reductions of tensors with affine layout
    * [Specializations](./test/unit/reduction/device/tensor_reduce_contiguous.cu) for reductions including contiguous dimension
    * [Specializations](./test/unit/reduction/device/tensor_reduce_strided.cu) for reductions excluding contiguous dimension
    * Custom reduction functors such as `cutlass::logical_and`
    * Large tensor support, up to 2^63 elements (however, each dimension is limited to an extent of 2^31)
  * Optimizations for 3-D convolution
    * [Optimized tile iterators](./include/cutlass/conv/threadblock/conv3d_fprop_activation_tile_access_iterator_optimized.h) using precomputed delta table for 3-D convolution
    * Full coverage of [forward](test/unit/conv/device/conv3d_fprop_implicit_gemm_f16ndhwc_f16ndhwc_f32ndhwc_tensor_op_f32_sm80.cu) and [backwards](test/unit/conv/device/conv3d_dgrad_implicit_gemm_f16ndhwc_f16ndhwc_f32ndhwc_tensor_op_f32_sm80.cu) passes for 3D convolution
  * [Fused Convolution+Convolution example](./examples/13_two_tensor_op_fusion/README.md)
  * Corrections and bug fixes reported by the CUTLASS community
    * Thank you for filing these issues!


## [2.4.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.4.0) (2020-11-19)
  * Implicit GEMM convolution kernels supporting CUDA and Tensor Cores on NVIDIA GPUs
    * Operators: forward (Fprop), backward data gradient (Dgrad), and backward weight gradient (Wgrad) convolution
    * Data type: FP32, complex<FP32>, Tensor Float 32 (TF32), BFloat16 (BF16), Float16, Int4, Int8, Int32
    * Spatial dimensions: 1-D, 2-D, and 3-D
    * Layout: NHWC, NCxHWx
  * Implicit GEMM convolution components:
    * Global memory iterators supporting Fprop, Dgrad, and Wgrad
    * `MmaMultistage` for implicit GEMM convolution for NVIDIA Ampere architecture
    * `MmaPipeline` for implicit GEMM convolution for NVIDIA Volta and Turing architectures
    * [Documentation](./media/docs/cpp/implicit_gemm_convolution.md) describing Implicit GEMM Convolution algorithm and implementation

## [2.3.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.3.0) (2020-09-23)
 * [NVIDIA Ampere Architecture features](https://devblogs.nvidia.com/nvidia-ampere-architecture-in-depth/)
   * [Sparse Tensor Core GEMM kernels](test/unit/gemm/device/gemm_f16n_f16n_f32t_tensor_op_f32_sparse_sm80.cu):
     * Direct access to Sparse Tensor Cores and maximum performance via [`mma.sp.sync`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma-and-friends)
   * Fast SGEMM targeting GeForce RTX 30-series CUDA Cores
 * Minor Features:
   * [Activation functions](./include/cutlass/epilogue/thread/activation.h) such as [GeLU](./include/cutlass/epilogue/thread/linear_combination_gelu.h) and [Sigmoid](./include/cutlass/epilogue/thread/linear_combination_sigmoid.h)
   * Small [matrix](./include/cutlass/matrix.h) and [quaternion](./include/cutlass/quaternion.h) template classes in device code
   * [Floating-point constants](./include/cutlass/constants.h)
 * NVIDIA Ampere GPU Architecture examples and documentation:
   * [Tensor Float 32](./examples/14_ampere_tf32_tensorop_gemm/ampere_tf32_tensorop_gemm.cu) and
   * [Sparse Tensor Cores](./examples/15_ampere_sparse_tensorop_gemm/ampere_sparse_tensorop_gemm.cu)
   * Documentation added on CUTLASS [efficient row-major epilogue](./media/docs/cpp/gemm_api.md#efficient-epilogue)

## [2.2.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.2.0) (2020-06-08)
 * [NVIDIA Ampere Architecture features](https://devblogs.nvidia.com/nvidia-ampere-architecture-in-depth/)
   * Fast Tensor Core operations:
    * Maximum performance via [`mma.sync`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma-and-friends)
    * Tensor Float 32, BFloat16, and double-precision data types
    * Mixed integer data types (int8, int4, bin1)
   * Asynchronous copy for deep software pipelines via [`cp.async`](https://docs.nvidia.com/cuda/parallel-thread-execution)
   * Described in [GTC 2020 Webinar (SR 21745)](https://developer.nvidia.com/gtc/2020/video/s21745) (free registration required)
 * Features:
   * SDK examples showing GEMM fused with bias+relu and fused GEMM+GEMM
   * Complex-valued GEMMs targeting NVIDIA Ampere Tensor Cores in double-precision and Tensor Float 32
   * Gaussian complex GEMMs using 3m complex multiply algorithm
   * Universal GEMM kernel supporting two batch modes and two algorithms for parallel reductions
 * Policy updates:
   * [CUDA 11 Toolkit](https://developer.nvidia.com/cuda-toolkit) needed to enable NVIDIA Ampere Architecture features
   * Disabled F16C by default for compatibility - enable on cmake command line with `-DCUTLASS_ENABLE_F16C=ON`

## [2.1.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.1.0) (2020-04-06)
 * BLAS-style host-side API added to [CUTLASS Library](./media/docs/cpp/quickstart.md#cutlass-library)
    * API to launch compiled kernel instances for GEMM and planar complex GEMM
 * Planar Complex GEMM kernels targeting Volta and Turing Tensor Cores
    * Computes complex matrix products on matrices stored as disjoint real and imaginary parts
    * [SDK Examples of Planar Complex GEMMs](./examples/10_planar_complex/planar_complex.cu)
 * Minor enhancements and bug fixes

## [2.0.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.0.0) (2019-11-19)
 * Substantially refactored for
    * Better performance, particularly for native Turing Tensor Cores
    * Robust and durable templates spanning the design space
    * Encapsulated functionality embodying modern C++11 programming techniques
    * Optimized containers and data types for efficient, generic, portable device code
  * Updates to:
    * [Quick start guide](./media/docs/cpp/quickstart.md)
    * [Documentation](./README.md#documentation)
    * [Utilities](./media/docs/cpp/utilities.md)
    * [CUTLASS Profiler](./media/docs/cpp/profiler.md)
 * Native Turing Tensor Cores
    * Efficient GEMM kernels targeting Turing Tensor Cores
    * Mixed-precision floating point, 8-bit integer, 4-bit integer, and binarized operands
 * Coverage of existing CUTLASS functionality
    * GEMM kernels targeting CUDA and Tensor Cores in NVIDIA GPUs
    * Volta Tensor Cores through native mma.sync and through WMMA API
    * Optimizations such as parallel reductions, threadblock rasterization, and intra-threadblock reductions
    * Batched GEMM operations
    * Complex-valued GEMMs
 * **Note: a host compiler supporting C++11 or greater is required.**

# CUTLASS 1.x

## [1.3.2](https://github.com/NVIDIA/cutlass/releases/tag/v1.3.2) (2019-07-09)
 * Performance improvement for Volta Tensor Cores TN and TT layouts.

## [1.3.1](https://github.com/NVIDIA/cutlass/releases/tag/v1.3.1) (2019-04-09)
 * Corrected NVRTC unit tests.

## [1.3.0](https://github.com/NVIDIA/cutlass/releases/tag/v1.3.0) (2019-03-20)
 * Efficient GEMM kernel targeting Volta Tensor Cores via `mma.sync` instruction added in CUDA 10.1.

## [1.2.0](https://github.com/NVIDIA/cutlass/releases/tag/v1.2.0) (2018-10-26)
 * Parallelized reductions across threadblocks ("Split-K")
   * Improved IGEMM performance
 * Batched strided WMMA GEMMs

## [1.1.0](https://github.com/NVIDIA/cutlass/releases/tag/v1.1.0) (2018-09-19)
  * Turing Features
    * WMMA GEMM targeting TensorCores - INT8, INT4, 1-bit
  * Batched Strided GEMM
  * Threadblock rasterization strategies
    * Improved performance for adverse problem sizes and data layouts
  * Extended CUTLASS Core comonents
    * Tensor views support arbitrary matrix and tensor layouts
    * Zip iterators for structuring multiple data streams
  * Enhanced CUTLASS utilities
    * Reference code for tensor operations in host and device code
    * Added HostMatrix<> for simplified matrix creation
  * Examples
    * Basic GEMM, tensor views, CUTLASS utilities, batched GEMM, WMMA GEMM

## [1.0.1](https://github.com/NVIDIA/cutlass/releases/tag/v1.0.1) (2018-06-11)

  * Intra-threadblock reduction added for small threadblock tile sizes
    * sgemm_64x128x16, sgemm_128x128x16, sgemm_128x64x16, sgemm_128x32x16, sgemm_64x64x16, sgemm_64x32x16
    * igemm_32x32x128
  * GEMM _K_ residue handled during prologue prior to mainloop
  * Replaced Google Test copy with submodule. Use `git submodule init --recursive --update`

## [1.0.0](https://github.com/NVIDIA/cutlass/commit/2028ebe120aab22bfd0b2baf8902d4c9627eb33f) (2018-05-16)

  * Substantial rewrite to accommodate new architecture
  * Kernels: SGEMM, DGEMM, IGEMM, HGEMM, WMMA GEMM
  * Unit and performance tests

## [0.0.1](https://github.com/NVIDIA/cutlass/commit/d08ba8ac46e2fa3f745e070c390182edb56b2e91) (2017-12-04)

  * Initial release


## Copyright

Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
