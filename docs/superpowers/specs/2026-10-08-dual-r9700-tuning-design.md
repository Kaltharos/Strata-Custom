# Dual Radeon AI PRO R9700 tuning: design

Date: 2026-10-08. Engine base: 0.1.40.4 (`6674a00`). Target: Ubuntu 26.04 LTS, 2x Radeon AI PRO
R9700 (gfx1201, 32 GiB each), 64 GB DDR5-6000 or more. Workload: IQ3_S today; Q4 and Q8/FP8 later.

## Purpose

Make Strata materially faster on a two-R9700 Linux box, and make the same machinery carry Q4 and
Q8/FP8 when those are wanted. The project is forked and hyperspecialized for this hardware, so
engine changes are in scope, as are changes that would not be acceptable upstream.

## The architecture this design is built on

Strata is **not** a VRAM-resident engine. It is tiered, and the tier a weight lives in decides the
token's speed far more than any kernel does.

| Tier | Mechanism | Where |
|---|---|---|
| VRAM | `ExpertCache` slots from the expert profile, adaptive swaps (`--adapt-every`) | `include/strata/core/expert_cache.hpp:81-108` |
| RAM | `--resident-experts` (whole complement) or `--resident-budget-gib N` (hottest N GiB), page-locked when the driver allows | `include/strata/core/expert_source.hpp:115-125`, `docs/DETAILS.md:245-256` |
| File | `--mmap-experts` over `experts.bin`; or the GGUFs read in place through `native_experts.txt` (3 reads per expert, `STRATA_FETCH_THREADS`); `RouterLookahead` warms predicted pages | `include/strata/core/expert_source.hpp:203-229`, `docs/DETAILS.md:213-221` |

The consequence, in the engine's own measurement (`docs/UNSLOTH_Q4.md:180-181`, one 3.5-token verify
round on an RTX 5070 with a 40 GiB budget):

| Stage | Time | Share |
|---|---:|---:|
| Reading experts from the SSD | 550-750 ms | ~86% |
| CPU expert kernels | ~75 ms | ~10% |
| GPU | ~13 ms | **~1.5%** |

**The GPU is ~1.5% of the time in the disk-bound tier.** That single table sets this design's
priorities: kernel work cannot reach the dominant term, and the second GPU's value is not obvious a
priori.

That the tiering works is not in question: "This is what runs Unsloth's UD-Q4_K_XL (72 GiB of experts)
on a 64 GB PC: 7-8.5 tokens/s at N = 40 on an RTX 5070" (`docs/DETAILS.md:255-256`) - on 12 GB of
VRAM and the same 64 GB of RAM. So Q4 is a throughput problem, not a fitting problem, and the target
box has 64 GB of VRAM plus 64 GB of RAM (~128 GB) against Q4_K_XL's 111.3 GB on disk
(`docs/UNSLOTH_Q4.md:49`).

### What follows from this for each goal

- **IQ3_S, experts in VRAM (now).** The file tier falls away, so the token is decided by GPU work,
  the serial 48-layer chain, and the host gap. `include/strata/core/session.hpp:207-225` measures
  that gap on an NVIDIA card: `--no-pool` is 38.73 ms/token against a 26.32 ms pure-GPU floor, so
  **~12.4 ms/token (a quarter of the token) is spent in the loop with no expert work at all - 0.258 ms
  per layer**, during which "the GPU has nothing queued and is IDLE". That text also names the fix:
  *"if it is the large half then the fix is fewer driver calls (R2.4), not a faster kernel."*
- **Q4 and Q8/FP8 (later).** These are the RAM budget plus the file tier. The levers are profile
  accuracy, RAM budget sizing, and file-tier latency - not VRAM capacity.

## Findings that shape the work

Read from the tree; each is a citation, not a measurement on the target box.

1. **The shipped expert profile is knowingly miscalibrated.** `tools/make_profile.py`'s own header:
   the shipped base ranks all 24,576 pairs, and *"a plain `--base` run cannot be moved by any trace -
   the output is the base again"*, so `--reorder` (or `--no-base`) is required. It further records
   that the coverage curve in `src/program/generate.cpp` carries *"94.9% claimed against 69.2%
   measured at 5,805 pairs held, on the IQ3_S 4-way rig"*. A profile fitted to the actual workload is
   therefore a real, available win - and it matters at every tier, because the profile decides what
   the VRAM cache holds, what the RAM budget takes, and what is left to the files.
2. **Native IQ packs never capture per-layer graphs.** `src/program/generate.cpp:5006`:
   `if (!o.no_capture && !native_pack)`. Captured graphs are the existing mitigation for per-layer
   host overhead ("~2,000 kernel launches become 48 graph launches", `session.hpp:237-239`), and the
   repo states a native pack "runs verify windows only" (`generate.cpp:5006`). So the IQ3_S path
   carries per-layer launch cost that the captured path does not - and this is where the latency work
   goes.
3. **The profiling tools split correctly for this pack.** `STRATA_DECODE_TIMING` (serve path,
   `generate.cpp:9776`) and `STRATA_VERIFY_PROFILE` (`verify.cpp:546`, `prof_on_`) run in the
   verify-window path that native packs use. `--gpu-stages` does not: it needs per-layer graphs and
   "refuses a native (IQ) pack" (`generate.cpp:5195`). Phase 1 may rely on the first two only.
4. **HIP's MMQ coverage is narrow.** `STRATA_PREFILL_MMQ` is on in setup's HIP build
   (`setup.py:2473`), but `CMakeLists.txt:76` scopes `STRATA_MMQ_KQUANTS` to "the CUDA experts of
   Unsloth's UD-Q4_K_XL, and on HIP the dense GGUF projections" - so on HIP only Q8_0 prompt-expert
   MMQ exists, and Q4_K / Q5_K / Q5_1 do not. Any AMD prefill work for Q4 starts from the kernel gap.
5. **Tensor parallelism is the wrong axis here.** Uniform TP-2 on 2x R9700 is bandwidth-neutral -
   each GPU reads half the weights at half the rate - so it adds a per-layer collective without
   removing read traffic. [llama.cpp measured zero gain from AllReduce on 2x R9700 with P2P
   working](https://github.com/ggml-org/llama.cpp/pull/27825). This matches the repo's own split
   measurements (`docs/AMD_HIP.md:293-297`). Recorded as a rejected alternative rather than left open.
6. **Setup gaps that keep any result from surviving an update.** Verified earlier in this session:
   `setup.py` never writes `split_skip_if_fits` although `serve/server.py:2053` and
   `generate.cpp:2907` wire it; `setup.py:5409-5412` gives a multi-card AMD install `layer_split:
   "auto"` and nothing else; `setup.py:5443` disables calibration for HIP, so AMD gets a core-count
   rule where NVIDIA gets a measurement; `docs/AI_SETUP.md` has no AMD performance section.

## The reference point: vLLM on the same box

The user measures **100-120 tok/s decode and up to ~18,000 tok/s prefill** on the same 2x R9700 with
vLLM (tcclaviger's Qwen3.8-Flash-Next config), against **60-70 tok/s decode and ~2,400 tok/s prefill**
on Strata with IQ3_S. That gap is the yardstick this design is measured against, and it splits into
two very different problems.

**Prefill (~7.5x): identified causes.** Each is a named code path, not an inference.

| Cause | Evidence |
|---|---|
| The QSA prompt attention defaults to the FP32 kernel; the RDNA4 WMMA kernel is 7.2-7.5x faster and is **opt-in** because it changes output bits | `docs/AMD_HIP.md:312-326`, `src/kernels/cuda/qsa_prompt_attn.cu:1129-1179`, `1915-1929` |
| MMQ prompt kernels exist for Q8_0 only on HIP; Q4_K / Q5_K / Q5_1 are CUDA-only (`STRATA_MMQ_KQUANTS` is CUDA; on HIP the same option covers only dense GGUF projections via `STRATA_DENSE_MMQ`) | `CMakeLists.txt:76`, `setup.py:4940-4943`, `docs/UNSLOTH_Q4.md:27` |
| Without a hipBLASLt table matching arch *and* library version, the prompt's dense GEMMs run plain hipBLAS | `docs/AMD_HIP.md:42-45` |
| vLLM prefill is matmul-dominated on tensor cores; Strata's prompt path also moves experts through the tiers | `docs/DETAILS.md:309-318` |

**Decode (60-70 vs 100-120): the arithmetic favours Strata, which locates the bottleneck.** The
active parameters are ~6 B, so per token:

| Engine | Bytes/token | tok/s | Effective bandwidth | Share of 640 GB/s |
|---|---:|---:|---:|---:|
| vLLM, FP8 | ~6 GB | 100-120 | ~600-720 GB/s | ~94-113% - at the HBM wall |
| Strata, IQ3_S | ~2.25 GB | 60-70 | ~135-158 GB/s | ~21-25% |

Strata is **not bandwidth-bound and not GPU-bound** on decode. It moves half the bytes per token that
vLLM does, at a quarter of the achieved bandwidth, so the ceiling is structural: the serial
per-token split (one GPU computes at a time), the uncaptured per-layer launch cost on native packs
(finding 2), and KV streaming over PCIe at long context (`--kv-resident`). This is the strongest
available evidence that the Section 2 latency work is the right target - and it also means **the
vLLM decode number is plausibly beatable**, not merely approachable, since FP8's byte count is not a
constraint Strata shares.

**PCIe generation changes the weighting.** This box runs PCIe 5.0 x8 (~32 GB/s effective), twice the
PCIe 4.0 x8 (~16 GB/s) of the 2x R9700 llama.cpp measurement. That does not revive tensor
parallelism - uniform TP-2 is bandwidth-neutral (finding 5), so halving a collective tax on a fix
that does not address the bottleneck does not make it address the bottleneck - but it does change two
things:

- expert streaming from the RAM and file tiers is twice as fast per byte;
- **`--pcie-frac` must be measured here, never inherited.** The AMD recipes' `--pcie-frac 0` gives the
  GPU no share of the misses, which measured 8.6-13.6 tok/s *on a gfx1030 card*; a *given* share also
  skips the per-stage link probes (`src/program/generate.cpp:1915-1917`, `3050-3058`). On a Gen5 link
  the share is worth more than on the hardware those recipes were tuned for.

## Non-goals

- **No tensor parallelism.** See finding 5 and the PCIe paragraph above.
- **No VRAM-fitting work.** The tiering already handles overflow; the target is throughput.
- **No change to the quant formats themselves.** Q4 and FP8 are engine paths to be made faster, not
  new packs to be invented.
- **No inherited `--pcie-frac 0` / `--adapt-every 0`.** Those recipes were measured on one card, and
  a *given* share skips the per-stage link probes (`generate.cpp:1915-1917`, `3050-3058`) that measure
  each card's own H2D bandwidth. On two cards these are measured for this box, never copied.

## Design

### Section 1 - Baseline, and the instrumentation that decides everything

Change nothing. Capture what the current install actually does, using the tools the repo already has.
**The step-by-step commands and the expected findings are in
[DUAL_R9700_PHASE0.md](../../DUAL_R9700_PHASE0.md).**

**Tier accounting.** `STRATA_DECODE_TIMING=1` plus `STRATA_VERIFY_PROFILE=1` give, per request, the
verify time split into *the wait for the GPU, the CPU expert pool (plan, activation quantization,
jobs), the stage, commit and draft* (`docs/DETAILS.md:297-304`). `GET /metrics` gives `ram_blobs`,
`file_blobs`, `file_mb`, `hit_rate` and `pcie_share` per recent request, with totals
(`docs/DETAILS.md:286-295`). Together these answer the only question that matters first: **is the
IQ3_S token GPU-bound, host-bound, or still touching RAM/files?**

**The startup log** gives the expert cache line, the layer-split line, slots per card, peak VRAM, and
the resident-RAM accounting.

**A/B harness.** A wrapper around `tools/hip/bench_prefill.py` adding what that script leaves to the
operator (`docs/AMD_HIP_PERFORMANCE.md:51-65`): restart the engine between arms (env is read once at
start, and the cache is built at start); **interleaved A/B/A/B pairs**, because 12-20% run-to-run
spread is documented on this hardware class (`docs/AMD_HIP.md:309-310`); discard the first fresh
prompt after each start (949 vs 1,606 tok/s cold vs warm, `docs/AMD_HIP.md:560-562`); capture the log
unbuffered; and record MTP `drafts_offered` / `drafts_accepted`, since any change that alters output
bits can pay for a throughput gain with a decode loss.

**Deliverable:** `bench/results/<date>-dual-r9700/` with baseline logs and the wrapper in `tools/hip/`.
**Gate:** if the baseline shows the IQ3_S token is already GPU-bound with negligible host gap, the
Section 2 ranking inverts and Section 3 becomes the work.

### Section 2 - The second GPU: measure the roles, then commit

Three candidate roles for GPU1, A/B'd with the Section 1 instrumentation rather than chosen on
intuition. The layer split is the incumbent, and it is genuinely open whether it wins:

| Role | Mechanism | Evidence to weigh |
|---|---|---|
| Pipeline peer (incumbent) | `--layer-split auto` or explicit; `--pipeline-windows 2` overlaps the next window instead of guess-and-rollback | on R9700 + 9070 XT the split read 4K at 1,384 vs 1,794 tok/s alone and decoded 42-43 vs 51-52 (`AMD_HIP.md:293-297`) - but that pair had unequal cards and the small one's cache was the constraint |
| Expert overflow tier | `--peer-device` (P2P rows, adaptive tier); **refused with a layer split** (`generate.cpp:2269-2271`) | the mechanism exists and is the natural fit for Q4 later, where overflow is the whole problem |
| Draft/speculation engine | GPU1 runs the MTP draft layer and its verify work while GPU0 runs the model | the draft layer currently lands on the last stage only, so during a window one card's draft work is serialized behind the other's layers |

Also in this section, because it is cheap and touches every tier:

- **A workload-fitted expert profile.** One-shot engine run with `--dump-routing FILE` on prompts
  typical of the real use, then `tools/make_profile.py --reorder` (the `--reorder` flag is mandatory;
  with the shipped base, a plain run reproduces the base exactly). Point `--expert-profile` at the
  result. This is the one change here that helps IQ3_S-in-VRAM and the Q4 path alike.
- **Per-stage dense trim** (`STRATA_STAGE_TRIM=1`) so each card loads only its own layers' dense
  weights (`docs/MULTI_GPU.md:101-107`), freeing VRAM for slots. It can change which experts run on
  GPU, so acceptance is measured.
- **The layer's own latency work**, if Section 1 shows the host gap dominating: native packs have no
  captured graphs (finding 2), so the per-layer launch cost is unmitigated on this path. Options
  range from graph capture for native packs to fusing the per-layer step; the choice is made from the
  measured split, not in advance.
- **Prefill: the RDNA4 WMMA prompt attention.** `STRATA_HIP_WMMA=1` is the single largest documented
  prefill lever on this card - 7.2-7.5x the portable kernel, R9700 4K 1,784 -> 2,427 and 16K 1,797 ->
  2,700 tok/s (`docs/AMD_HIP.md:312-326`) - and it is off only because its output bits differ from
  ~token 50. It is the first thing to try against the vLLM prefill gap, and it interacts with the
  prompt ring (96 slots vs the default 384), so the ring must be held fixed while it is measured.
  Accepting the output change is the user's call; the spec records it as such.

**Acceptance:** the winning role stated with an interleaved paired comparison, the profile change
measured separately from the role change (one variable at a time), and the tier accounting shown
before and after.

### Section 3 - Q4 and Q8/FP8

Driven by Section 1's tier split, and by what the engine already does. Targets, in order of expected
value:

1. **Expert-profile accuracy at the RAM/file boundary.** With a budget, the profile decides what is
   resident; a fitted profile plus the existing `RouterLookahead` is the predictor pair. Measure the
   warm-hit rate (`routing prefetch`, `DETAILS.md:286-288`).
2. **RAM budget sizing.** `--resident-budget-gib` is documented as "the setting that matters most; a
   bigger budget was faster in every measurement (24 / 32 / 40 GiB)" (`docs/UNSLOTH_Q4.md:142-147`).
   On 64 GB of RAM the right N for this box is measured, not assumed, and the cgroup/`MemAvailable`
   guard (`AMD_HIP_PERFORMANCE.md:36-48`) is part of the acceptance.
3. **File-tier read latency.** `STRATA_IO_PREFETCH=1` with `STRATA_IO_PF_STAGE=1` measured +25.7% on a
   32 GB box and -32.6% on a 16 GB one (`DETAILS.md:233-237`), so its sign depends on the memory
   regime: measured here, not inherited. The Linux-specific I/O path and its `file tier I/O` log line
   (`DETAILS.md:223-230`) give the read accounting.
4. **Q8/FP8 specifically:** confirm which formats the engine has kernels for on gfx1201 before
   designing anything. The repo supports Q4_K / Q5_K / Q5_1 / Q8_0 / Q4_0 / Q4_1 expert kernels and
   states other quantizations "use formats this engine may not have kernels for"
   (`UNSLOTH_Q4.md:289-291`). FP8 as a weight format is not established by anything read so far, so
   this is a gate, not an assumption.

**Acceptance:** per-change, an interleaved paired comparison; the tier split before and after; and any
format claim backed by a named kernel that exists in the tree.

### Section 4 - Make it durable

1. **Write `split_skip_if_fits` from setup only where measurably justified** - the rule being that the
   first card's cache would hold every profiled pair for the model being installed.
2. **Extend calibration to HIP**, or add a clearly-named AMD equivalent. `setup_calibration(cfg, hip)`
   already takes the backend flag and `tools/calibrate.py` already understands `layer_split` and
   `split_skip_if_fits`, so the gap is the caller (`setup.py:5443`, `5458`), not the machinery.
3. **Add the AMD performance section to `docs/AI_SETUP.md`**, so an AI-assisted install reaches what a
   reader of `AMD_HIP.md` would.
4. **Document the dual-R9700 configuration** in the repo's style: plain words, measured numbers, and
   the machine each was measured on. Every number from this box, labelled as such.

**Acceptance:** `python tools/test_setup_*.py` passes (they need no GPU or download); new setup
behaviour has a test in the same style.

## Verification strategy

- **Here, statically:** the setup tests run without a GPU or a model, so Section 4 is verifiable before
  it reaches the target box. Kernel or engine changes cannot be compiled or run here at all.
- **On the target box, per the user:** engine start, `strata-device --selftest`, `ctest` for any engine
  change, then the A/B harness. Logs return to this session.
- **No claim without the log that shows it.** Where a result sits inside the measured spread, the spec
  says so rather than reporting the favourable half.

## Open questions for Phase 0

1. What do `STRATA_DECODE_TIMING` and `/metrics` show for IQ3_S today: GPU wait, CPU pool, stage,
   commit, draft - and the RAM/file blob counts?
2. Which model form is installed: a native IQ pack built by `tools/iq_pack.py`, and does its folder
   have an `experts.bin`? This decides which tier code path runs.
3. Actual PCIe link width/generation per card, what the per-stage `probe_pcie_h2d_gbps` reports, and
   the measured effect of `--pcie-frac` on **this** Gen5 link (never inherited from the AMD recipes).
4. Which card drives the display, if any (`--vram-reserve-later-mib`).
5. Context length and whether `--kv-resident` is on - both change the VRAM available for slots.
6. Which hipBLASLt version the installed ROCm reports, and whether `STRATA_HIPBLASLT_TUNING` is set.

## Risks

- **Measurement noise dominates small effects.** 12-20% run-to-run spread is documented here; only
  interleaved pairs with reported ranges are admissible evidence.
- **Native-pack graph capture is a large change.** If Section 1 shows the host gap is small on this
  box, the expensive item in Section 2 is dropped rather than assumed worthwhile.
- **Options that change output bits** (`STRATA_STAGE_TRIM`, adaptive swaps, `STRATA_HIP_WMMA`) can flip
  near-ties. Each needs the user's consent and a note in the docs; the repo is explicit that
  `STRATA_HIP_WMMA`'s output "differs from the default kernel's in its last bits".
- **The acceptance criteria can legitimately fail.** If the baseline is already optimal on the measured
  axes, the spec says so instead of manufacturing a result.
