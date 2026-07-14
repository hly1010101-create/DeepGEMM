# SM100 FP8 MQA Notes

## Q Pipeline Barrier Phase: To Revisit

The Q pipeline has a fixed number of shared-memory stages (`kNumQStages`),
but a CTA can process more Q blocks than that. A stage is therefore reused:

```text
iteration 0 -> stage 0 -> Q block A
iteration 1 -> stage 1 -> Q block B
iteration 2 -> stage 2 -> Q block C
iteration 3 -> stage 0 -> Q block D
```

Questions to resolve while rereading the code:

- Why is it unsafe to overwrite stage 0 with Q block D before every consumer of Q block A has released it?
- What are the distinct meanings of: "Q block A has been copied", "UMMA has issued work using A", and "A is fully reusable"?
- How do `full_q_barriers[stage]` and `empty_q_barriers[stage]` distinguish the old use of stage 0 (A) from the later use (D)?
- How does the mbarrier phase bit toggle across reuse, and why do the waits use `q_phase` for `full_q` but `q_phase ^ 1` for `empty_q`?

Relevant scheduling code:

```cpp
q_stage_idx = q_iter_idx % kNumQStages;
q_phase = (q_iter_idx / kNumQStages) & 1;

full_q_barriers[q_stage_idx]->wait(q_phase);
empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);
```

Working model: a stage may be overwritten only after its `empty_q` barrier
confirms that every consumer has finished. The phase bit labels alternating
reuse generations of the same barrier, rather than representing a permanent
boolean `full` or `empty` state.

## Q Stage Selection: To Revisit

The Q-stage selection in the scheduler is:

```cpp
(q_iter_idx + q_iter_offset) % kNumQStages
```

Questions to resolve while rereading the producer/consumer loops:

- Why does `q_iter_offset = 0` select the stage for the Q block currently
  being computed, while `q_iter_offset = 1` selects the stage for the next Q
  block to prefetch?
- How does the prefetch in iteration `q_iter_idx = 0` make stage 1 contain the
  current Q block when the computation advances to `q_iter_idx = 1`?

Example with `kNumQStages = 3`:

```text
iteration 0: compute Q0 in stage 0; prefetch Q1 into stage 1
iteration 1: compute Q1 in stage 1; prefetch Q2 into stage 2
iteration 2: compute Q2 in stage 2; prefetch Q3 into stage 0
```

## TCGEN05 Fence After Empty-Barrier Wait: To Revisit

Question: what ordering does the following instruction provide after the UMMA
issuer waits for the math warp-group to release its TMEM region?

```cpp
empty_umma_barriers[i]->wait(previous_phase);
ptx::tcgen05_after_thread_sync();
```

Current conclusion: `tcgen05_after_thread_sync()` lowers to
`tcgen05.fence::after_thread_sync`. It is not a thread synchronization point
and does not wait for UMMA or TMEM operations to complete. It orders the
preceding thread synchronization (the `empty_umma` barrier wait) before later
TCGEN05 operations that reuse or overwrite the same TMEM region.

The complete consumer-to-producer handoff is:

```text
math warp-group:
TMEM load
  -> tcgen05_before_thread_sync()
  -> empty_umma_barrier.arrive()

UMMA issuer warp:
empty_umma_barrier.wait()
  -> tcgen05_after_thread_sync()
  -> tcgen05.mma reuses/overwrites TMEM
```

Point to verify against the PTX ISA: the `before_thread_sync` and
`after_thread_sync` fences bridge TCGEN05 operations with ordinary thread
synchronization; neither fence performs the synchronization itself.

## TMEM Load Parameter-Pack Expansion: To Revisit

Question: how does the templated lambda below turn a TMEM load into 32 or 64
register output arguments, and what does each `...` mean?

```cpp
using Loader = cute::conditional_t<N == 32,
    cute::SM100_TMEM_LOAD_32dp32b32x,
    cute::SM100_TMEM_LOAD_32dp32b64x>;

[&]<size_t... Is>(cute::index_sequence<Is...>) {
    Loader::copy(tmem_addr, reinterpret_cast<uint32_t*>(accum)[Is]...);
}(cute::make_index_sequence<N>{});

cutlass::arch::fence_view_async_tmem_load();
```

Current understanding:

- `conditional_t` selects the TMEM-load instruction wrapper at compile time.
- `size_t... Is` declares a non-type template parameter pack.
- `make_index_sequence<N>` creates the compile-time sequence `0..N-1`.
- `[Is]...` expands `accum[0]` through `accum[N-1]` into separate output
  arguments to `Loader::copy`.
- `accum` is viewed as `uint32_t*` because the instruction wrapper exposes
  32-bit register outputs; the resulting bits are later consumed as FP32.
- `fence_view_async_tmem_load()` is required before subsequent code consumes
  the values loaded asynchronously from TMEM.

Point to verify against the CUTLASS/PTX implementation: the exact completion
and ordering semantics of `fence_view_async_tmem_load()`, including the PTX
instruction to which it lowers.
