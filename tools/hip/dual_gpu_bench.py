#!/usr/bin/env python3
"""tools/hip/dual_gpu_bench.py - restart the server per arm and capture what decides the next move.

Why this exists rather than calling bench_prefill.py directly: the engine reads its environment once
at start and builds the expert cache at start, so every arm needs its own process.  It also has to
gather the diagnostics bench_prefill.py deliberately leaves to the operator (docs/AMD_HIP_PERFORMANCE.md
"Reproduce the configuration"), and it must interleave arms, because 12-20% run-to-run spread is
documented on R9700-class hardware (docs/AMD_HIP.md, "Speed switches").

Standard library only: nothing to install on the box.

    # one arm, hardware + startup + decode timing + prefill/decode
    python3 tools/hip/dual_gpu_bench.py --config strata-iq3s.json --label base

    # interleaved A/B/A/B over the same arms (what a real comparison needs)
    python3 tools/hip/dual_gpu_bench.py --config strata-iq3s.json --arms wmma,base --rounds 2

    # every arm this file knows about
    python3 tools/hip/dual_gpu_bench.py --config strata-iq3s.json --arms all --rounds 2

Outputs, under --out (default bench/results/<date>-dual-r9700/):
    hardware.txt            cards, VRAM, PCIe links, power caps, ROCm, and the engine's own device list
    <label>/engine.log      the engine's stderr for that arm
    <label>/bench.json      bench_prefill.py's per-request numbers
    <label>/parsed.json     this script's parse of the startup lines and the decode timing
    server-<label>.out      the server's own stdout/stderr (why an arm failed to come up)
    SUMMARY.txt             every arm side by side

The engine log path is taken from the config, so nothing here guesses where the engine writes.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------------------------------------- arms
# Each arm is env overrides plus extra engine args.  `base` changes nothing, so it is the control to
# compare every other arm against.  The interleaving in run() is what makes the comparison valid.
ARMS = {
    "base": {},
    "wmma": {"env": {"STRATA_HIP_WMMA": "1"}},
    # The AMD recipes all pass --pcie-frac 0.  Those were measured on one card and a Gen4 link; this
    # box is Gen5 x8, where giving the GPU a share of the misses is worth re-measuring.  `0` means
    # "no share given" and also skips the per-stage link probes (src/program/generate.cpp:1915-1917).
    "pcie0": {"args": ["--pcie-frac", "0"]},
    "reserve300": {"args": ["--vram-reserve-mib", "300"]},
    # Informational: on gfx12 this is already the default (off), so it is a no-op check that the log
    # agrees, not an optimisation.  Kept so the log states it rather than assuming.
    "shstream_on": {"env": {"STRATA_SH_STREAM": "1"}},
    "select_wmma": {"env": {"STRATA_SELECT_WMMA": "1"}},
}

# Diagnostic flags.  STRATA_DECODE_TIMING is cheap (it reads counters the decode loop already keeps) and
# is always on because the per-window split is the point of this harness.  STRATA_VERIFY_PROFILE is NOT:
# it times every GPU stage with events and docs/DETAILS.md warns "it slows the decode a little: use it to
# compare, not to measure speed" - so it is opt-in via --profile and never on during a throughput arm.
DIAG_ENV = {"STRATA_DECODE_TIMING": "1"}
PROFILE_ENV = {"STRATA_VERIFY_PROFILE": "1"}
# Cheap and useful: says whether a hipBLASLt table loaded and whether any GEMM fell back to plain hipBLAS.
VERBOSE_ENV = {"STRATA_HIPBLASLT_VERBOSE": "1"}


def run(cmd, **kw):
    """A command that is allowed to fail: returns (rc, stdout+stderr)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=kw.pop("timeout", 60), **kw)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return 1, f"<{type(e).__name__}: {e}>"


def hardware(out_dir: Path, cfg: dict):
    """Everything about the box that a later reader needs to judge the numbers."""
    lines = [f"# dual_gpu_bench hardware capture", f"date: {date.today().isoformat()}",
             f"kernel: {run(['uname', '-a'])[1].strip()}", ""]

    lines.append("## ROCm / HIP")
    for cmd in (["hipcc", "--version"], ["rocminfo"], ["amd-smi", "version"]):
        rc, txt = run(cmd, timeout=90)
        head = "\n".join(txt.strip().splitlines()[:6])
        lines.append(f"$ {' '.join(cmd)}   (rc={rc})\n{head}\n")

    lines.append("## hipBLASLt version header (decides which tuning table is valid)")
    for p in ("/opt/rocm/include/hipblaslt/hipblaslt-version.h",
              os.path.expandvars("$ROCM_PATH/include/hipblaslt/hipblaslt-version.h")):
        if Path(p).is_file():
            lines.append(f"--- {p}\n{Path(p).read_text(errors='replace')}")

    lines.append("## PCIe links (the generation and width decide what --pcie-frac is worth)")
    rc, txt = run(["bash", "-lc",
                   "for d in /sys/class/drm/card*/device; do "
                   "  echo \"$(basename $(dirname $d)): $(cat $d/current_link_speed 2>/dev/null)"
                   " x$(cat $d/current_link_width 2>/dev/null) (max $(cat $d/max_link_speed 2>/dev/null)"
                   " x$(cat $d/max_link_width 2>/dev/null)) $(cat $d/uevent 2>/dev/null | grep -i pci_id)\"; "
                   "done"])
    lines.append(txt.strip() or "<no /sys/class/drm/card*/device entries>")
    rc, txt = run(["bash", "-lc", "lspci -vv 2>/dev/null | grep -A3 -iE 'vga|display' | head -60"])
    lines.append("\n$ lspci (display devices)\n" + (txt.strip() or "<lspci unavailable>"))

    lines.append("\n## VRAM per card")
    rc, txt = run(["bash", "-lc",
                   "for f in /sys/class/drm/card*/device/mem_info_vram_total; do "
                   "echo \"$f = $(( $(cat $f) / 1024 / 1024 )) MiB\"; done"])
    lines.append(txt.strip())

    # Power caps are not cosmetic: the repo's own A/Bs were taken at fixed caps (85 W on the MI50 pair,
    # 272 W on the 7900 XTX), and a decode difference of this size can otherwise be a cap, not a change.
    # A cap that differs between arms would be reported as a code effect.
    lines.append("\n## GPU power caps (record them; they must not differ between arms)")
    rc, txt = run(["bash", "-lc",
                   "command -v rocm-smi >/dev/null && rocm-smi --showpower --showtemp 2>/dev/null; "
                   "for c in /sys/class/drm/card*/device/hwmon/hwmon*; do "
                   "  [ -r \"$c/power1_cap\" ] && echo \"$c power1_cap=$(cat $c/power1_cap) "
                   "power1_cap_max=$(cat $c/power1_cap_max 2>/dev/null)\"; done"])
    lines.append(txt.strip() or "<no rocm-smi and no power1_cap exposed>")

    lines.append("\n## System memory")
    rc, txt = run(["bash", "-lc", "grep -E 'MemTotal|MemAvailable|SwapTotal' /proc/meminfo"])
    lines.append(txt.strip())

    lines.append("\n## CPU")
    rc, txt = run(["bash", "-lc", "lscpu | grep -E 'Model name|^CPU\\(s\\)|NUMA|Thread|Core|Socket'"])
    lines.append(txt.strip())

    lines.append("\n## Storage (the file tier's floor on Q4/FP8)")
    rc, txt = run(["bash", "-lc", "lsblk -d -o NAME,MODEL,SIZE,ROTA 2>/dev/null | head -20"])
    lines.append(txt.strip())

    lines.append("\n## The engine's own view (arch, VRAM, wave32)")
    exe = cfg.get("exe", "")
    if exe:
        # A HIP engine needs the runtime its BUILD.json names; setup puts it in .venv.
        venv = ROOT / ".venv" / "lib"
        pre = f"export LD_LIBRARY_PATH=$(find {venv} -maxdepth 3 -name 'libamdhip64.so*' -printf '%h:' 2>/dev/null)$LD_LIBRARY_PATH; "
        rc, txt = run(["bash", "-lc", pre + f'"{exe}" --list-devices'], timeout=120)
        lines.append(f"$ {exe} --list-devices   (rc={rc})\n{txt.strip()}")

    lines.append("\n## Config in use")
    lines.append(json.dumps(cfg, indent=2))

    text = "\n".join(lines) + "\n"
    (out_dir / "hardware.txt").write_text(text)
    print(f"  hardware -> {out_dir / 'hardware.txt'}")
    return text


# ------------------------------------------------------------------------------------- engine log parse
def parse_engine_log(path: Path) -> dict:
    """Pull the startup numbers and the decode-timing split out of the engine's stderr."""
    try:
        text = path.read_text(errors="replace")
    except OSError as e:
        return {"error": f"cannot read {path}: {e}"}

    out = {"log": str(path), "bytes": len(text)}

    m = re.search(r"expert cache\s+([0-9.]+) GB\s+\((\d+) slots of ([\d,]+) B\)", text)
    if m:
        out["expert_cache_gb"] = float(m.group(1))
        out["expert_cache_slots"] = int(m.group(2))
        out["expert_blob_bytes"] = int(m.group(3).replace(",", ""))

    for pat, key in ((r"layer split auto: (K=\d+[^\n]*)", "layer_split_auto"),
                     (r"layer split: ([^\n]+)", "layer_split"),
                     (r"resident RAM[^\n]*", "resident_ram"),
                     (r"PCIe probe: ([^\n]+)", "pcie_probe"),
                     (r"layer split: CUDA(\d+) PCIe probe ([^\n]+)", "stage_pcie_probe"),
                     (r"native pack: ([^\n]+)", "native_pack"),
                     (r"tuning enabled \(([^\n]+)\)", "hipblaslt_tuning")):
        hits = re.findall(pat, text)
        if hits:
            out[key] = hits if len(hits) > 1 else hits[0]

    # "strata serve: expert tiers: ..." and the router predictor's hit rate
    out["expert_tiers_lines"] = re.findall(r"expert tiers[^\n]*", text)[:8]
    out["routing_prefetch_lines"] = re.findall(r"routing prefetch[^\n]*", text)[:4]

    # The decode-timing line is the single most important thing this script collects: it separates
    # the wait for the GPU from the CPU pool from the host-side stage cost.
    dt = re.findall(r"strata decode timing:[^\n]*", text)
    out["decode_timing"] = dt
    gp = re.findall(r"strata decode GPU stages[^\n]*", text)
    out["decode_gpu_stages"] = gp

    # Whether the ROMA/WMMA kernels announced themselves at all.
    out["mentions_wmma"] = len(re.findall(r"WMMA|wmma", text))
    return out


def summarise_bench(bench_path: Path) -> dict:
    """bench_prefill.py's json -> the fresh-prompt medians that comparisons are made on."""
    try:
        rows = json.loads(bench_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return {"error": str(e)}
    fresh = [r["metrics"] for r in rows if r.get("kind") == "fresh"]
    follow = [r["metrics"] for r in rows if r.get("kind") == "followup"]

    def med(vals):
        vals = sorted(vals)
        return vals[len(vals) // 2] if vals else None

    return {
        "fresh_n": len(fresh),
        "fresh_prefill_tps_median": med([f["prefill_tps"] for f in fresh]),
        "fresh_prefill_tps_all": [f["prefill_tps"] for f in fresh],
        "fresh_decode_tps_median": med([f["decode_tps"] for f in fresh]),
        "fresh_decode_tps_all": [f["decode_tps"] for f in fresh],
        "fresh_prompt_tokens": [int(f["prompt_tokens"]) for f in fresh],
        "followup_decode_tps_median": med([f["decode_tps"] for f in follow]),
    }


# ---------------------------------------------------------------------------------------------- server
def wait_health(port: int, proc: subprocess.Popen, timeout_s: int) -> bool:
    """Poll /health.  Model load takes minutes on this hardware, so the timeout is long by design."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            print(f"  !! server exited early (rc={proc.returncode})")
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            pass  # not up yet
        time.sleep(3)
    return False


def stop(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def run_arm(label: str, arm: dict, cfg: dict, out_dir: Path, port: int,
            model: str, ready_timeout: int, engine_stderr_to: Path, profile: bool = False):
    """One arm: a fresh config with the arm's env/args, a fresh server, then bench_prefill.py."""
    arm_dir = out_dir / label
    arm_dir.mkdir(parents=True, exist_ok=True)

    arm_cfg = json.loads(json.dumps(cfg))                 # deep copy; the user's config is not touched
    arm_cfg["port"] = port
    diag = {**DIAG_ENV, **VERBOSE_ENV, **(PROFILE_ENV if profile else {})}
    arm_cfg["env"] = {**(arm_cfg.get("env") or {}), **diag, **(arm.get("env") or {})}
    for extra in arm.get("args", []):
        if extra not in arm_cfg.get("args", []):
            arm_cfg["args"] = list(arm_cfg.get("args", [])) + [extra]

    engine_log = Path(cfg.get("log") or (ROOT / "strata.log"))
    engine_log.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg = arm_dir / "config.json"
    tmp_cfg.write_text(json.dumps(arm_cfg, indent=2))

    print(f"\n=== arm {label} : env={arm.get('env') or {}} args={arm.get('args') or []}")
    with engine_stderr_to.open("wb") as errf:
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata",
             "--config", str(tmp_cfg), "--port", str(port)],
            cwd=str(ROOT), stdout=errf, stderr=subprocess.STDOUT, env=dict(os.environ))
        try:
            if not wait_health(port, proc, ready_timeout):
                (arm_dir / "FAILED").write_text("server did not become healthy\n")
                return None
            print(f"  up; running bench_prefill.py")
            bench_json = arm_dir / "bench.json"
            rc, txt = run([sys.executable, str(ROOT / "tools" / "hip" / "bench_prefill.py"),
                           "--url", f"http://127.0.0.1:{port}", "--model", model,
                           "--engine-log", str(engine_log), "--output", str(bench_json),
                           "--label", label], timeout=3600)
            (arm_dir / "bench_prefill.out").write_text(txt)
            if rc != 0:
                print(f"  !! bench_prefill failed rc={rc}; see {arm_dir / 'bench_prefill.out'}")
        finally:
            stop(proc)
            time.sleep(5)                                # let the driver release the cards

    # the engine's log is the real evidence; keep a copy beside the arm
    if engine_log.is_file():
        shutil.copy2(engine_log, arm_dir / "engine.log")

    parsed = parse_engine_log(engine_log)
    parsed["bench"] = summarise_bench(arm_dir / "bench.json")
    parsed["arm_env"] = arm.get("env") or {}
    parsed["arm_args"] = arm.get("args") or []
    (arm_dir / "parsed.json").write_text(json.dumps(parsed, indent=2))
    b = parsed.get("bench", {})
    print(f"  prefill {b.get('fresh_prefill_tps_median')} tok/s | decode {b.get('fresh_decode_tps_median')} tok/s"
          f" | slots {parsed.get('expert_cache_slots')}")
    return parsed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="the strata-*.json the server runs")
    ap.add_argument("--arms", default="base",
                    help="comma list of arms, or 'all'. Known: " + ", ".join(ARMS))
    ap.add_argument("--rounds", type=int, default=1,
                    help="how many times to cycle the arm list; 2+ interleaves A/B/A/B (needed for a real comparison)")
    ap.add_argument("--label", help="override the arm name (single-arm runs only)")
    ap.add_argument("--model", default=None, help="model name for the API (default: the config's model_name)")
    ap.add_argument("--port", type=int, default=None, help="default: the config's port")
    ap.add_argument("--out", type=Path, default=None, help="default: bench/results/<date>-dual-r9700")
    ap.add_argument("--ready-timeout", type=int, default=2400, help="seconds to wait for /health (model load)")
    ap.add_argument("--profile", action="store_true",
                    help="add STRATA_VERIFY_PROFILE=1 (per-GPU-stage event timings). Slows decode: use the "
                         "result to see WHERE time goes, never as a speed measurement")
    ap.add_argument("--no-hardware", action="store_true", help="skip the hardware capture")
    args = ap.parse_args()

    cfg_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config)
    if not cfg_path.is_file():
        sys.exit(f"no such config: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())

    out_dir = args.out or (ROOT / "bench" / "results" / f"{date.today().isoformat()}-dual-r9700")
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(ARMS) if args.arms == "all" else [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        sys.exit(f"unknown arm(s): {unknown}. Known: {', '.join(ARMS)}")
    if args.label and len(names) != 1:
        sys.exit("--label is for a single-arm run; use --arms for comparisons")

    port = args.port or int(cfg.get("port") or 8080)
    model = args.model or cfg.get("model_name") or "strata"

    print(f"config : {cfg_path}")
    print(f"out    : {out_dir}")
    print(f"arms   : {names} x {args.rounds} round(s)   port {port}   model {model!r}")
    print(f"note   : each arm restarts the engine; a model load takes minutes, so this is slow by nature")

    if not args.no_hardware:
        hardware(out_dir, cfg)

    results = []
    for rnd in range(1, args.rounds + 1):
        for name in names:
            label = args.label or (name if args.rounds == 1 else f"{name}-r{rnd}")
            parsed = run_arm(label, ARMS[name], cfg, out_dir, port, model,
                             args.ready_timeout, out_dir / f"server-{label}.out", profile=args.profile)
            if parsed:
                parsed["round"] = rnd
                parsed["arm"] = name
                results.append(parsed)

    # ---------------------------------------------------------------- summary
    lines = [f"dual_gpu_bench summary  {date.today().isoformat()}",
             f"config: {cfg_path}", f"arms: {names} x {args.rounds}", ""]
    head = f"{'arm':<14}{'slots':>8}{'prefill tok/s':>15}{'decode tok/s':>14}  layer split"
    lines.append(head)
    lines.append("-" * len(head))
    for r in results:
        b = r.get("bench") or {}
        pf = b.get("fresh_prefill_tps_median")
        dc = b.get("fresh_decode_tps_median")
        ls = r.get("layer_split", r.get("layer_split_auto", ""))
        if isinstance(ls, list):
            ls = ls[0]
        lines.append(f"{r.get('arm','?'):<14}{str(r.get('expert_cache_slots','?')):>8}"
                     f"{(f'{pf:.1f}' if pf else '?'):>15}{(f'{dc:.1f}' if dc else '?'):>14}  {str(ls)[:60]}")

    lines += ["", "## decode timing (the line that decides whether the host gap is worth attacking)",
              "Look for: the wait for the GPU, the CPU expert pool, the stage, commit, draft."]
    for r in results:
        lines.append(f"\n[{r.get('arm')}]")
        for ln in (r.get("decode_timing") or ["<none - was STRATA_DECODE_TIMING honoured?>"]):
            lines.append("  " + ln.strip())
        for ln in (r.get("decode_gpu_stages") or []):
            lines.append("  " + ln.strip())

    lines += ["", "## expert tiers and router-prefetch hit rate (how much the file tier is still costing)",
              "(on IQ3_S with every expert in VRAM these should show 0 file blobs; on Q4 they are the story)"]
    for r in results:
        lines.append(f"\n[{r.get('arm')}]")
        for ln in (r.get("expert_tiers_lines") or ["<no 'expert tiers' lines>"]):
            lines.append("  " + ln.strip())
        for ln in (r.get("routing_prefetch_lines") or []):
            lines.append("  " + ln.strip())

    lines += ["", "## per-arm spread (interleaving is what makes the comparison valid)"]
    for r in results:
        b = r.get("bench") or {}
        lines.append(f"{r.get('arm'):<14} prefill {b.get('fresh_prefill_tps_all')}  decode {b.get('fresh_decode_tps_all')}")

    (out_dir / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {out_dir / 'SUMMARY.txt'}")


if __name__ == "__main__":
    main()
