# Dual R9700: Phase 0 measurement runbook

This is the measurement step of
[the tuning design](superpowers/specs/2026-10-08-dual-r9700-tuning-design.md). It changes nothing. Its
job is to record what the current install does and to locate the bottleneck, so the optimisation work
targets a measured limit instead of a guess.

Run it on the Ubuntu 26.04 box. Nothing here needs internet, root, or a rebuild.

## What this is for

Three questions decide everything that follows:

1. **Where does a decode window's time go** - waiting for the GPU, the CPU expert pool, the host-side
   stage, commit, or draft? This decides whether the per-layer host gap is worth attacking
   (`--pipeline-windows`, graph capture for native packs) or whether the GPU is genuinely busy.
2. **Is any expert still coming from RAM or the file tier** on IQ3_S? If yes, the 64 GB of VRAM is not
   being used and the cache sizing is the first target. If no, the token is compute/host-bound.
3. **What is the PCIe reality** - link generation and width per card, and what the engine's own
   per-stage `probe_pcie_h2d_gbps` reports? That sets what `--pcie-frac` is worth on this box.

## Before you start

Restart the engine once cleanly and leave the box otherwise idle. Background load - a browser, a
compile, another model - changes these numbers more than most of the changes we will test.

Record the GPU power caps. The repo's own measurements were all taken at fixed caps (85 W on the MI50
pair, 272 W on the 7900 XTX); the harness captures them into `hardware.txt` automatically, but they
must be the same for every arm:

```sh
rocm-smi --showpower --showtemp
```

## Run it

The harness restarts the engine per arm, because the engine reads its environment once at start and
builds the expert cache at start.

```sh
cd /path/to/Strata

# 1. the baseline: hardware, startup lines, decode timing, prefill and decode
python3 tools/hip/dual_gpu_bench.py --config strata-iq3s.json --label base

# 2. the per-GPU-stage split (slower decode by design - it is a locator, not a speed run)
python3 tools/hip/dual_gpu_bench.py --config strata-iq3s.json --label base-profiled --profile

# 3. the biggest documented prefill lever, interleaved against the control
python3 tools/hip/dual_gpu_bench.py --config strata-iq3s.json --arms wmma,base --rounds 2
```

Use your real config file name in place of `strata-iq3s.json`. The default port and model name come
from that config.

**What each step costs:** every arm restarts the engine and reloads the model, and step 3 is four
arms, so expect this to take a while. That is the price of valid numbers - `docs/AMD_HIP.md` records
12-20% run-to-run spread on this hardware class, which is wider than several of the effects we are
chasing, so a single run of each arm cannot separate signal from drift.

**Step 3 is a real decision, not a warm-up.** `STRATA_HIP_WMMA=1` measured 7.2-7.5x the portable
kernel on the R9700 (4K 1,784 -> 2,427 tok/s, 16K 1,797 -> 2,700) but its output differs from the
default's in the last bits, from roughly token 50 on (`docs/AMD_HIP.md:312-326`). If you are not
willing to accept that, say so and I will drop it from the plan rather than keep proposing it.

## What to send back

Under `bench/results/<date>-dual-r9700/`:

| File | Why it matters |
|---|---|
| `SUMMARY.txt` | every arm side by side: slots, prefill, decode, the decode-timing split, the tier counts |
| `hardware.txt` | cards, VRAM, PCIe links, power caps, ROCm and hipBLASLt versions, the engine's device list |
| `base/parsed.json` | the parsed startup lines for the control arm |
| `base-profiled/parsed.json` | the per-GPU-stage split |
| `server-<label>.out` | only if an arm failed to come up |

`SUMMARY.txt` plus the two `parsed.json` files is normally enough. The raw `engine.log` per arm is
there if something looks wrong.

If you would rather do it by hand, the equivalent is: start the server with
`STRATA_DECODE_TIMING=1` in the environment, send one request, then read the
`strata decode timing:` line out of the engine log (`docs/DETAILS.md:297-304`).

## How to read the result

- **Decode timing separates `ms_host` from GPU work.** If the "wait for the GPU" term dominates, the
  GPU is the floor and only kernel/format work helps. If the stage/commit/host terms dominate, the fix
  is fewer driver calls and better overlap - the direction chosen for this fork.
- **`expert tiers` with 0 file blobs and 0 RAM blobs** means IQ3_S really is fully VRAM-resident, and
  the Q4 work is a separate problem from the IQ3_S latency work.
- **`expert cache N slots` per card** against 24,576 profiled pairs total tells us whether the two
  cards' caches together cover the model or whether the layer split is costing coverage.
- **The `layer split auto:` line** reports the placement and the coverage it predicts. The engine
  records that its own coverage curve is optimistic (94.9% claimed against 69.2% measured at one
  point, `tools/make_profile.py`), so the measured tier counts matter more than the prediction.

## Expected findings, stated in advance

So the result can contradict them honestly:

- **IQ3_S decode is not bandwidth-bound.** 6 B active parameters at ~3 bits is roughly 2.25 GB per
  token; at 60-70 tok/s that is ~135-158 GB/s against ~640 GB/s of HBM. Compare the vLLM FP8 config on
  the same box: ~6 GB per token at 100-120 tok/s is ~600-720 GB/s, i.e. at the wall. If decode timing
  shows the GPU busy, this reasoning is wrong and the GPU really is the limit.
- **Prefill has a large, identified headroom** if the QSA prompt attention is on the FP32 path.
- **The host gap may be small on this box.** The 0.258 ms/layer figure is from an NVIDIA card
  (`include/strata/core/session.hpp:207-225`). If this box shows almost no host term, the expensive
  part of the plan is dropped rather than pursued.
