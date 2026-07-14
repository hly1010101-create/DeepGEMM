#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <uint32_t kNumHeads, uint32_t kHeadDim, // q的head数目， dim维度
          bool kIsCompressedLogits, // 是否只保留有效logits
          uint32_t BLOCK_Q, uint32_t BLOCK_KV, // 一次处理多少个 query token, 一次载入、计算多少个 KV token
          uint32_t kNumQStages, uint32_t kNumKVStages, // q,kv tile的 shared-memory 环形缓冲区数量
          uint32_t kNumSMs, // 运行该kernel的sm的数量
          uint32_t kNumSpecializedThreads, // 专职线程数，负责 TMA 搬运 Q/KV/scale、维护 barrier、调度任务等。
          uint32_t kNumMathThreads,  // 计算线程数，负责 UMMA/MMA、从 TMEM 读累加结果、乘 weight、ReLU 和写 logits。
          typename logits_dtype_t, // logits的数据类型
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>  //  Number of 128-thread warpgroups used for MMA and logits computation.
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads /*线程总数*/, 1 /*最少block数量*/)
void sm100_fp8_mqa_logits(const uint32_t seq_len /*seq*/, const uint32_t seq_len_kv /*seqkv*/,
                          const uint32_t max_seqlen_k /*最大有效k长度*/, const uint32_t stride_logits /*物理行步长，？？（后续深度思考）*/,
                          uint32_t* cu_seq_len_k_start,  // Shape: [seq_len]
                          uint32_t* cu_seq_len_k_end,  // For Q token q_idx, its valid KV range is [cu_seq_len_k_start[q_idx], cu_seq_len_k_end[q_idx]).
                          logits_dtype_t* logits, // shape(seq, stride_logits)
                          const __grid_constant__ cute::TmaDescriptor tensor_map_q, // 这里传入的是 TMA 描述符本身，不是数据指针。
                          const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales, // kv的反量化参数
                          const __grid_constant__ cute::TmaDescriptor tensor_map_weights) { // shape(seq_len, num_heads)
/*
cute::TmaDescriptor 是 CUDA Tensor Memory Accelerator 使用的硬件描述符，内部记录了：
Q/KV/scale/weight 的全局内存基址
张量维度与 stride
数据类型、布局及边界信息
__grid_constant__ 表示该 kernel grid 内所有 block 共享同一份只读参数，编译器可将其作为 grid-constant kernel parameter 处理。
*/
    // TODO: consider TMA multicast：TMA multicast 是 Hopper 及更新架构中，一次从全局内存发起的 TMA 搬运，同时把同一份数据写入同一个 thread-block cluster 内多个 block 的 shared memory 的能力。
    // Normally, `h (kNumHeads) == 32` and `d (kHeadDim) == 64`
    // For one block, we process `[q_start:q_end, h, d] @ [kv_start:kv_end, d] -> [q_start:q_end, kv_start:kv_end]`；这里是简化写法
    // Q should be load only at once for a block：在数据复用策略上和 FlashAttention 是一致的
    const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q); // 计算q_blocks数量防止越界

    // Types
    // __syncthreads() 只能保证：线程同步，
    //  ClusterTransactionBarrier 是硬件 mbarrier 的封装，除了“参与线程到齐”之外，还维护一个 transaction byte count：
    // 预计要搬：N bytes
    // 已完成搬运：M bytes
    // 只有 M == N，barrier 才 ready
    using Barrier = cutlass::arch::ClusterTransactionBarrier; 

    // Utils
    const auto sm_idx = blockIdx.x;  // sm数量 = blocks数量，拉取任务的逻辑还需要复盘
    const auto warp_idx = cutlass::canonical_warp_idx_sync(); //所有 lane 都通过同一次 warp shuffle 获得同一个结果，并要求调用时 warp 已经汇合（CUTLASS 源码也明确注明这一前提）。
    const auto warpgroup_idx = warp_idx / 4; // 4个warp为一组
    const auto lane_idx = ptx::get_lane_idx(); // 和threadidx%32等同
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;  // kNumMathWarpGroups是模板参数，计算搬运warpgroup的起始warp

    // Prefetch TMA descriptors
    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads"); // check，special是一个warpgroup， math是多个warpgroup
    if (warp_idx == kSpecWarpStart) { // tma descriptor是地址、shape、stride、swizzle 等元数据，不是 Q/KV 的实际数据，这里搬运的是描述符，不是真实数据
        /*把 descriptor 预热到 TMA 可访问的缓存/路径中。随后专职 warp 0、warp 1 发起各自的 TMA copy 时，通常直接命中，不需要重复承担冷启动访问延迟。*/
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }

    // Shared memory configs
    // NOTES: weight may be unaligned
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3); // 计算每个stage需要的size大小
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE = math::constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, 512u); // shared-memory 的整个布局要求按 512 B 边界组织，以满足该 kernel 的 TMA/swizzle 布局约束

    // Align to 512 bytes for swizzle-64B
    extern __shared__ __align__(512) uint8_t smem_buffer[]; 
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling"); // check 是否符合TMA swizzling
    DG_STATIC_ASSERT(SMEM_WEIGHT_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");

    // TMA configs
    // WG 0：KV 的前 128 行 × 同一份 N=128 列
    // WG 1：KV 的后 128 行 × 同一份 N=128 列
    constexpr uint32_t kNumTmemCols = BLOCK_Q * kNumHeads * kNumMathWarpGroups;  // 计算要申请的tmem列数，MQA kernel 把 Q tile 展开成 UMMA 的 N 维：每个 math warpgroup并行处理不同的 KV 子块，KV 子块对应 M 维
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory"); // 超过拒绝

    // 在已分配的一整块动态 shared memory 中，计算并标记各个子区域的起始地址。
    // Data on shared memory
    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * i);
    });
    /* 等同于这个操作
    __nv_fp8_e4m3* get_smem_q(uint32_t i) {
    auto byte_address = smem_buffer + i * SMEM_Q_SIZE_PER_STAGE;
    return reinterpret_cast<__nv_fp8_e4m3*>(byte_address);
    }
    */
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i));
    });
    auto smem_kv_scales = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages +
            SMEM_KV_SIZE_PER_STAGE * kNumKVStages + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });

    // TMA barriers
    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_q_barriers     = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; }); // 生产完成信号
    auto empty_q_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); }); // 消费完成信号，kNumQStages流水线深度（对应stages个信号）
    auto full_kv_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });
    auto full_umma_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + i); });
    auto empty_umma_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + kNumMathWarpGroups + i); });

    // Tensor memory allocation
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + kNumMathWarpGroups * 2);
    /*
    extern __shared__ uint8_t smem_buffer[];
    [Q stages]
    [weight stages]
    [KV stages]
    [KV-scale stages]
    [full_q barriers]
    [empty_q barriers]
    [full_kv barriers]
    [empty_kv barriers]
    [full_umma barriers]
    [empty_umma barriers]
    [TMEM pointer]
    */

    // Initialize barriers
    DG_STATIC_ASSERT(kNumSpecializedThreads % 128 == 0 and kNumSpecializedThreads >= 64, "Invalid threads");  // check
    if (warp_idx == kSpecWarpStart and cute::elect_one_sync()) { // lane_idx == 0 假设 lane 0 一定活跃；elect_one_sync() 会从当前 active lanes 中选一个可用线程。
        // Q：跨多个 KV block 复用，释放条件更严格； KV：每个 block 独立消费，完成一轮即可释放
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1); // 设定这个硬件 barrier 的到达计数目标。这里一个专职 producer 线程负责发起该 stage 的 TMA，因此目标是 1。
            empty_q_barriers[i]->init(kNumMathThreads + 32); // + 32 对应 发起 UMMA 的那个专职 warp。kNumMathThreads 个 math 线程 + 32 个 UMMA issuer warp 的线程
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(kNumMathThreads); // math threads 的 arrive 已通过 full_umma 间接覆盖 UMMA 依赖
        }
        cutlass::arch::fence_barrier_init(); // 所有 barrier 的 init 写入， 先对整个 CTA/cluster 可见，再允许其他线程使用该 barrier
    }
    if (warp_idx == kSpecWarpStart + 1) {
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                full_umma_barriers[i]->init(1); // 已写入 TMEM、可以读取的信号。
                empty_umma_barriers[i]->init(128); // 该 warpgroup 已经读完这块 TMEM accumulator、可以被下一轮 UMMA 覆盖的信号。
            }
            cutlass::arch::fence_barrier_init(); 
        }
        // Allocate tensor memory
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem); // 申请tensor memory
    }
    __syncthreads(); // 初始化结束

    // Register reconfigurations
    /*
    寄存器重分配
    specialized warps：TMA、barrier、调度为主，寄存器需求较低 -> 每线程保留 40 个寄存器
    math warps：需要保存 weights、TMEM 读出的 accum、临时归约变量等 -> 每线程保留 232 个寄存器
    */
    constexpr uint32_t kNumSpecializedRegisters = 40;
    constexpr uint32_t kNumMathRegisters = 232;

    // Block scheduler， block调度
    uint32_t block_q_idx = sm_idx, q_iter_idx = 0; // q_iter_idx（计算迭代次数）
    const auto get_next_block_q_idx = [&]() -> cute::tuple<uint32_t, uint32_t> {
        return {block_q_idx + kNumSMs, q_iter_idx + 1}; // 下一个task，以及迭代次数+1
    };
    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q]; // 存储kv有效range
    const auto load_schedule = [&](const uint32_t& q_iter_offset = 0) -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max(); // 2^32-1
        uint32_t end = cute::numeric_limits<uint32_t>::min(); // 0

        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1); // 计算q_idx，防止越界
            seq_k_start[i] = cu_seq_len_k_start[q_idx]; // 计算kv的start和end
            seq_k_end[i] = cu_seq_len_k_end[q_idx]; 
            start = min(start, min(seq_k_start[i], seq_len_kv)); // 防止越界
            end = max(end, min(seq_k_end[i], seq_len_kv)); 
        }
        // TMA alignment requirements for SF KV
        start = start / 4 * 4; // 对齐到4的倍数，直接从第13个搬，可能不满足该TMA layout的对齐约束。
        return {(q_iter_idx + q_iter_offset) % kNumQStages,       // Q pipeline stage    // q_iter_offset是相当当前stage的迭代次数，提前得到下一轮要使用的 stage/phase，并把下一 Q block 的 Q 数据发起 TMA 预取。
                ((q_iter_idx + q_iter_offset) / kNumQStages) & 1, // Q pipeline phase
                start, math::ceil_div(end - start, BLOCK_KV)};          // Task info，起始点及几个kv分块（kv分块已经内含end的information）
    };

    // KV pipeline
    /*
    Q 用 q_iter_offset=1 是因为“下一 Q”属于下一轮外层循环；KV 的“下一KV”仍在当前这层 for 循环里，不需要额外 offset 参数。
    */
    uint32_t num_total_kv_blocks = 0;
    const auto get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {
            (num_total_kv_blocks + kv_block_idx) % kNumKVStages,         // KV pipeline stage
            ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1    // KV pipeline phase, 防止把旧 KV block 的 barrier 状态误认为新 KV block 的状态。
        };
        /*
        kNumKVStages = 3
        KV global block 0 -> stage 0, phase 0
        KV global block 1 -> stage 1, phase 0
        KV global block 2 -> stage 2, phase 0
        KV global block 3 -> stage 0, phase 1
        */
    };

    // UMMA settings
    // Construct instruction with layout D
    constexpr uint32_t UMMA_M = 128;
    constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);  // UMMA_K= 32/1 = 32；
    constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads; // attention.hpp中定义block_qh = 128； block_q = block_qh / kNumHeads

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();  // 之前的操作不依赖数据，可以提前做；这个操作是等待前序算子完成。前序kernel会发送cudaTriggerProgrammaticLaunchCompletion()信号。

    /*
    kSpecWarpStart + 0：TMA producer
    kSpecWarpStart + 1：UMMA issuer
    kSpecWarpStart + 2：不承担计算
    kSpecWarpStart + 3：不承担计算
    */

    if (warp_idx == kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>(); // 每线程的寄存器配额被降为 40

        // Prefetch
        /*
        定义一个lambda，将一个 Q block 及其对应的 weights，用 TMA 异步搬到指定的 Q pipeline stage，并设置该 stage 的 full_q 完成条件。
        Q 和 weight 都搬完
        -> full_q barrier ready
        -> consumer 才能同时安全读取 Q 和 weight
        */
        const auto issue_tma_q = [&](const uint32_t& stage_idx, const auto& block_idx) {
            tma::copy<kHeadDim, BLOCK_Q * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, block_idx * BLOCK_Q * kNumHeads);
            tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx], 0, block_idx * BLOCK_Q);
            full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
        };
        // 一个线程负责发起 TMA Q，其他线程等待
        if (cute::elect_one_sync() and block_q_idx < num_q_blocks)
            issue_tma_q(0, block_q_idx);

        // Only the first lane persistently schedules over blocks ，仅第一个线程束（lane）持续跨块进行调度。
        if (cute::elect_one_sync()) {
            while (block_q_idx < num_q_blocks) {
                /*
                auto result = load_schedule(1);
                uint32_t q_stage_idx    = result[0];
                uint32_t q_phase        = result[1];
                uint32_t kv_start       = result[2];
                uint32_t num_kv_blocks  = result[3];
                */
                CUTE_TIE_DECL(load_schedule(1), q_stage_idx, q_phase, kv_start, num_kv_blocks);

/*
关于mbarrier的说明，使用sm的smem作为储存后端，硬件提供专用cache负责真正的值
mbarrier对应的所有操作都是作用于cache，在初始化的时候smem上的值和cache值统一；销毁的时候cache值写入smem，cache释放。正常状态更新主要由硬件缓存维护，不保证cache实时写回 smem
mbarrier数据表示上是一个64bit的整型数据：
Phase(1bit) -- 用来记录轮次
Arrive Count(20bit) -- 负责记录尚未达到的线程数(use负数)
Lock(1bit) -- 当arrive的线程数大于期待线程数时，则此mbarrier出错，Lock设置为1进入错误锁定状态
Transaction(21bit) -- 负责记录尚未达到的数据量（use负数），当其中的值=0时，数据抵达
Expected Arrive Count(20bit) --期待要到达的线程数。当下一轮时，Arrive Count == Expected Arrive Count，节省时间。
Reserved(1bit) -- 保留位，没有其他含义

mbarrier 使用方法：？？这里需要补充。
mbarrier wait(x) 需要当mbarrier内部的Phase != x时，才放行。
*/

                // Wait Q consumer release
                // 计算出本轮 q_phase，通过q_phase^1得到上一轮的phase值
                // wait(q_phase^1) ==> while(phase == q_phase^1) {},PASS;
                /*
                进一步翻译：
                if 上一轮数据没消费完，Phase没翻转，等于q_phase^1，阻塞。
                esle 上一轮数据消费完，Phase翻转，不等于q_phase^1, pass。
                */
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);    

                // Issue TMA Q
                /*
                循环过程一直在加载Q数据，然后观察依赖，前三次都是直接循环加载，后续的存在依赖情况
                stage 0 初始加载      → 不需要 empty wait
                stage 1 首次加载      → 初始为空，wait 直接通过
                stage 2 首次加载      → 初始为空，wait 直接通过
                stage 0 第一次复用    → 必须等待旧 Q 消费完成
                */
                if (const auto& next_block_q_idx = cute::get<0>(get_next_block_q_idx()); next_block_q_idx < num_q_blocks)
                    issue_tma_q(q_stage_idx, next_block_q_idx);  // 发起下一轮 Q block 的 TMA 预取。

                // Issue TMA KV
                #pragma unroll  // for循环上界是一个运行时变量，不一定能展开
                for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                    // Wait consumer release
                    CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);  // 获取下一轮 KV block 的 stage 和 phase。
                    empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);  // 等待前一个阶段的 consumer 释放。

                    // Issue TMA KV
                    // 取当前kv block的kv和kv_scales，用TMA异步搬到指定的KV pipeline stage，并设置该 stage 的 full_kv 完成条件。
                    tma::copy<kHeadDim, BLOCK_KV, kHeadDim>(&tensor_map_kv, full_kv_barriers[kv_stage_idx],
                                                            smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
                    tma::copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales, full_kv_barriers[kv_stage_idx],
                                              smem_kv_scales[kv_stage_idx], kv_start + kv_block_idx * BLOCK_KV, 0);
                    full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE); // 两组 TMA 已经发射, 生产者线程执行 arrive, barrier 登记预计到达的总字节数, 数据可能仍在传输，尚未全部抵达
                }
                num_total_kv_blocks += num_kv_blocks; // 更新累计的 KV block 总数。

                // Jump to the next block
                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx); // 跳转到下一个 Q block。更新block_q_idx和q_iter_idx。
            }
        }
    } else if (warp_idx == kSpecWarpStart + 1) {  // 等待 full_q / full_kv barrier，构造并发射 UMMA FMA，将结果写入 TMEM， 通过 full_umma_barriers 通知 math warp-groups
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>(); // 每线程的寄存器配额被降为 40

        // Require full allocation
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0); // 要求 TMEM 分配从列 0 开始；后续 UMMA/TMEM 地址均以 0 为基址。
        // Make UMMA desc
        /*
        auto instr_desc = cute::UMMA::make_instr_desc<
            cutlass::float_e4m3_t,  // A 元素类型
            cutlass::float_e4m3_t,  // B 元素类型
            float,                  // 累加器/输出类型
            UMMA_M,                 // 输出 tile 的 M
            UMMA_N,                 // 输出 tile 的 N
            cute::UMMA::Major::K,   // A 的主维布局
            cute::UMMA::Major::K    // B 的主维布局
        >();
        */
        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>(); // 构造 UMMA 描述符。
        auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc); // 转化为运行时指令编码

        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);  // 重新计算当前 Q block 要使用的 Q stage、Q phase、KV 起点及 KV tile 数， 疑问？ 为什么这里要重新计算

            // Wait TMA Q arrival
            full_q_barriers[q_stage_idx]->wait(q_phase);  // 等待 Q block 的 TMA 预取完成。

            // Compute over KV blocks
            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                // Compute `[BLOCK_Q * kNumHeads, kHeadDim] @ [BLOCK_KV, kHeadDim] -> [BLOCK_Q, BLOCK_KV]`
                // Wait TMA KV arrival
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase); // 计算当前第 kv_block_idx 个 KV tile 应落在哪个 KV pipeline stage
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);  // 等待 KV block 的 TMA 预取完成。

                // Issue UMMA
                DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads, "Invalid block size");  // 隐含保证kNumMathWarpGroups * UMMA_M == BLOCK_KV； BLOCK_KV = 256，kNumMathThreads = 256，kNumMathWarpGroups = 256 / 128 = 2，UMMA_M = 128;
                DG_STATIC_ASSERT(kHeadDim % UMMA_K == 0, "Invalid head dim");  // 隐含保证kHeadDim是UMMA_K的倍数
                #pragma unroll  // UMMA发射两次，两个UMMA tile合起来正好覆盖一个kv block
                for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                    empty_umma_barriers[i]->wait(((num_total_kv_blocks + kv_block_idx) & 1) ^ 1); // 上一轮 TMEM 累加结果已经读完，可以覆盖。疑问？？
                    ptx::tcgen05_after_thread_sync(); // 疑问？
                    #pragma unroll // 沿 kHeadDim 分块，连续发射多条 UMMA 指令，把一个完整的点积累加到同一块 TMEM 中
                    for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {  
                        auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                            smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K); // 为当前 UMMA 的 A 操作数构造一份 shared-memory 描述符
                        auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                            smem_q[q_stage_idx], 0, k * UMMA_K); 
                        cute::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, i * UMMA_N, k, runtime_instr_desc);  // 发射mma指令
                    }
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(full_umma_barriers[i]));  // 提交此前发射的 UMMA + 注册“完成后更新 full_umma_barrier”，前面的 UMMA 可能仍在执行，不能认为已经完成。
                }
            }

            // 推进逻辑 KV tile 累计序号，使下一个 Q block 的 KV pipeline
            // stage/phase 和 UMMA/TMEM barrier parity 延续轮转，而不是从 0 重置。
            num_total_kv_blocks += num_kv_blocks;  

            // UMMA warp must also arrive on empty_q to prevent running ahead
            // of math warps in the Q pipeline
            empty_q_barriers[q_stage_idx]->arrive();  // UMMA warp 已经不会再发射新的、引用当前 Q stage 的指令。区别于不再访问Q，后续还有一个umma的barrier。全部为空后才能覆盖旧的stage

            // Jump to the next block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx); // 获取下一轮的block_q_idx，和q_iter_idx
        }
    } else if (warp_idx == kSpecWarpStart + 2 or warp_idx == kSpecWarpStart + 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();  // 寄存器重配置要求以完整 warp-group 为单位参与。所以这个必须指定寄存器重分配
    } else if (warp_idx < kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();  // math warp-group

        // Offsets
        const auto tmem_start = warpgroup_idx * UMMA_N; // 计算每个 math warp-group 对应的 TMEM 起始列地址
        const auto math_thread_idx = warp_idx * 32 + lane_idx; // 计算math_thread_idx

        // Helper lambda for loading tensor memory
        // 定义了一个 lambda 函数
        // 从 tmem_addr 指定的 TMEM 地址加载 N 个 FP32 累加结果，写入线程本地寄存器数组 accum
        auto tmem_load = [](auto num_elems_c, const uint32_t& tmem_addr, float* accum) {
            constexpr int N = decltype(num_elems_c)::value; // 从 num_elems_c 的类型中取出编译期整数值，可以提前确定指令。有一定疑问？
            DG_STATIC_ASSERT(N == 32 or N == 64, "Unsupported TMEM load size"); // 编译期限制每线程的 TMEM 加载规模；当前仅支持一次加载 32 或 64 个 FP32 累加值。
            using Loader = cute::conditional_t<N == 32,
                cute::SM100_TMEM_LOAD_32dp32b32x,
                cute::SM100_TMEM_LOAD_32dp32b64x>;
            [&]<size_t... Is>(cute::index_sequence<Is...>) {
                Loader::copy(tmem_addr, reinterpret_cast<uint32_t*>(accum)[Is]...);
            }(cute::make_index_sequence<N>{});  // load 累加结果
            cutlass::arch::fence_view_async_tmem_load(); // 等待此前发射的异步 TMEM load 完成。
        };

        // Local register buffers
        float weights[BLOCK_Q][kNumHeads];  // 分配：放在每个math thread的寄存器中

        // math操作
        while (block_q_idx < num_q_blocks) { 

/*
对于一个 KV block：warpgroup 0: [128 KV, BLOCK_Q × kNumHeads]，warpgroup 1: [128 KV, BLOCK_Q × kNumHeads]，写入对应位置即可。
warpgroup 的 TMEM tile：[128, BLOCK_Q * kNumHeads]，每个 warpgroup 处理固定的 UMMA_M = 128 个 KV token。
拆分为 BLOCK_Q 次：每次 warpgroup 处理 [128, kNumHeads]
单线程每次的处理的fragment：[kNumHeads]
使用float2一次处理2个数据：做rule，× weight，reduce，两个数据reduce，× scale_kv，最后一个线程持有一个数据
store该数据
*/
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks); // 计算当前的q_stage_idx, q_phase, kv_start, num_kv_blocks

            // Wait TMA Q arrival
            full_q_barriers[q_stage_idx]->wait(q_phase);  // Q的TMA已经搬运完成

            // Read weights
            // 当前 Q stage 中的 weights 从 shared memory 读取到每个 math 线程自己的寄存器数组
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads; ++ j)
                    weights[i][j] = ptx::ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + j);
            }

            // Compute over KV blocks，对当前 Q block 所覆盖的所有 KV tile 逐块进行计算。
            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                // Compute `[BLOCK_Q * kNumHeads, kHeadDim] @ [BLOCK_KV, kHeadDim] -> [BLOCK_Q, BLOCK_KV]`
                // Wait TMA KV arrival
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase); // 计算kv_stage_idx，kv_phase 
                full_kv_barriers[kv_stage_idx]->wait(kv_phase); // kv tile和其scales搬运完成，可以读取

                // Read per-KV scales
                float scale_kv = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + math_thread_idx);  // 一个kv对应一个scales

                // Wait UMMA arrival
                full_umma_barriers[warpgroup_idx]->wait((num_total_kv_blocks + kv_block_idx) & 1);
                ptx::tcgen05_after_thread_sync();  //umma抵达，TMEM 结果可以读取。

                // Release KV empty
                empty_kv_barriers[kv_stage_idx]->arrive();  // 在 TMEM 归约和 logits 写回之前释放 KV stage，因为后续计算只依赖 TMEM 结果

                // Reduce over the head dim and store
                const auto kv_offset = kv_start + kv_block_idx * BLOCK_KV + math_thread_idx; // 计算当前 math 线程负责的全局 KV token 索引。
                DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head"); // check

                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    // Load accumulator from TMEM
                    float accum[kNumHeads];  // 对于单线程来说，每个线程读取kNumHeads个数据
                    tmem_load(cute::Int<kNumHeads>{}, tmem_start + i * kNumHeads, accum); // load结果

                    // Release TMEM empty
                    if (i == BLOCK_Q - 1) {
                        ptx::tcgen05_before_thread_sync();
                        empty_umma_barriers[warpgroup_idx]->arrive(); // 当最后一个 Q 行的 accumulator 已从 TMEM 加载到寄存器后，释放当前 math warp-group 对应的 TMEM 区域。
                    }

                    // Accumulate weighted ReLU in parallel
                    // 对当前 Q token 与当前 KV token 的所有 head 点积结果执行
                    // sum_h(weights[i][h] * ReLU(accum[h])；每次用 float2 处理两个 head，
                    // 并使用两条独立累加链提高指令级并行度。
                    auto sum_0 = make_float2(0, 0);
                    auto sum_1 = make_float2(0, 0);

                    const auto transform = [&](const uint32_t& j, const float2& sum) {
                        auto a = make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0));  // relu操作，消除负数
                        auto b = make_float2(weights[i][j], weights[i][j + 1]); // 准备weight
                        return __ffma2_rn(a, b, sum); // a * b + sum
                    };

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads; j += 4) {
                        sum_0 = transform(j, sum_0);
                        sum_1 = transform(j + 2, sum_1);  // 一次性处理4个数据
                    }

                    // 把两条独立的 float2 累加链按分量合并
                    auto sum = __fadd2_rn(sum_0, sum_1); // reduce
                    auto result = static_cast<logits_dtype_t>(scale_kv * (sum.x + sum.y)); // 两个维度reduce，× scale_kv

                    // Store into the global memory
                    const auto q_offset = (block_q_idx * BLOCK_Q + i) * static_cast<uint64_t>(stride_logits); // 计算q写入地址
                    if constexpr (kIsCompressedLogits) {
                        if (seq_k_start[i] <= kv_offset and kv_offset < seq_k_end[i])
                            logits[q_offset + kv_offset - seq_k_start[i]] = result; // 写入结果
                    } else {
                        logits[q_offset + kv_offset] = result; // 全部写入，不管有效值
                    }
                    __syncwarp(); // warp级别线程同步
                }
            }
            num_total_kv_blocks += num_kv_blocks; // 更新num_total_kv_blocks

            // Release Q empty
            empty_q_barriers[q_stage_idx]->arrive(); // 释放Q，可以写入

            // Jump to the next block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx); // 跳转下一个Qtile
        }

        // Free tensor memory
        /*
        所有 math warp-groups：完成计算并参与同步
        仅 math warp 0：执行 TMEM free
        所有线程：随后退出 kernel
        */
        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (warp_idx == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }
}

} // namespace deep_gemm
