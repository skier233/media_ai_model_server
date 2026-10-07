"""Measure where shot-boundary analysis time goes, one layer at a time.

Every level runs on the same stream-copied excerpt of a source, so the gap
between two adjacent levels is the cost of the layer between them::

    L0    read the source file from its own storage, cold cache
    L1    ffmpeg software decode only                     (-f null)
    L1t   L1 with -threads set to every available CPU     (-f null)
    L1c   ffmpeg NVDEC decode only                        (-f null)
    L2    L1 plus the production scale and RGB conversion (-f null)
    L2t   L2 with -threads set to every available CPU     (-f null)
    L2c   L1c plus the production GPU scale chain         (-f null)
    L3    the production decode piped into Python, frames discarded
    L3c   L3 on the NVDEC backend
    L3av  L3 on the PyAV backend
    L4    AIShotBoundaryModel._analyze, timed at its seams
    L4c   L4 on the NVDEC backend

L1t, L2t, L3av and L4c are not run by default. The L1/L2 commands come from
``ffmpeg_pipe.build_command`` and differ from production only in where the
frames go. Every level decodes at the model's clip geometry; each level fixes
its own decode backend except L4, which uses the model config's.

    python scripts/bench_shot_boundary.py pick --candidates C.tsv --out corpus.json
    python scripts/bench_shot_boundary.py excerpt --corpus corpus.json --dir excerpts
    python scripts/bench_shot_boundary.py run --corpus corpus.json --out results.jsonl
    python scripts/bench_shot_boundary.py report results.jsonl

The candidates file is a TSV with a header row and the columns ``path``,
``codec``, ``width``, ``height``, ``fps``, ``duration``, ``bitrate`` and,
optionally, ``is_vr``. Results identify a source only by an anonymous id and
its stratum, never by path, so a results file can be shared as it is. The
corpus file maps ids to paths and should stay private.

Linux only: it reads /proc and /sys and evicts the page cache with
posix_fadvise.
"""

import argparse
import asyncio
import bisect
import contextlib
import csv
import hashlib
import inspect
import json
import logging
import os
import platform
import random
import resource
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# numpy only; torch and the model are imported by `run` alone.
from lib.model.preprocessing_python import ffmpeg_pipe as fp  # noqa: E402

DEFAULT_MODEL_CONFIG = "config/models/omnishotcut_shot_boundaries.yaml"

DEFAULT_LEVELS = ("L0", "L1", "L1c", "L2", "L2c", "L3", "L3c", "L4")
# Display order; the extra levels are opt-in.
ALL_LEVELS = ("L0", "L1", "L1t", "L1c", "L2", "L2t", "L2c", "L3", "L3c", "L3av", "L4", "L4c")
NVDEC_LEVELS = frozenset({"L1c", "L2c", "L3c", "L4c"})
GPU_LEVELS = NVDEC_LEVELS | {"L4"}
# libavcodec caps its automatic thread count at 16, so on a bigger host the
# default decode never uses every CPU. These levels ask for all of them.
THREADED_LEVELS = {"L1t": "L1", "L2t": "L2"}
LEVEL_NOTES = {
    "L0": "read the source file, cold cache",
    "L1": "ffmpeg software decode only",
    "L1t": "L1 with -threads set to every available CPU",
    "L1c": "ffmpeg NVDEC decode only",
    "L2": "L1 + production scale and RGB conversion",
    "L2t": "L2 with -threads set to every available CPU",
    "L2c": "L1c + production GPU scale chain and download",
    "L3": "production decode piped into Python, frames discarded",
    "L3c": "L3 on the NVDEC backend",
    "L3av": "L3 on the PyAV backend",
    "L4": "production _analyze: decode + inference",
    "L4c": "L4 on the NVDEC backend",
}

clock = time.perf_counter


# ── corpus ───────────────────────────────────────────────────────────────

LEGACY_WMV = {"wmv3", "wmv2", "msmpeg4v3", "msmpeg4v2", "vc1"}
INTERLACED = {"tt", "bb", "tb", "bt"}


def _pixels(candidate) -> int:
    return candidate["width"] * candidate["height"]


def _sane_fps(fps) -> bool:
    return 1.0 <= float(fps or 0) <= 120.0


class Stratum:
    """A slice of the library: which candidates belong, how many to pick, and
    how many frames an excerpt needs for a stable throughput reading."""

    def __init__(self, name, quota, frames, match, probe_check=None, prefer=None):
        self.name = name
        self.quota = quota
        self.frames = frames
        self.match = match
        self.probe_check = probe_check
        self.prefer = prefer


# Quotas follow each class's share of *decoded pixels*, which is what decode
# cost tracks, rather than its share of files. Excerpt lengths aim at a few
# seconds of decode per level at that class's expected speed.
STRATA = (
    Stratum("1080p-h264-le30", 5, 10000,
            lambda c: c["codec"] == "h264" and 1_500_000 <= _pixels(c) < 3_000_000
            and _sane_fps(c["fps"]) and c["fps"] <= 31 and not c["is_vr"] and c["duration"] >= 600),
    Stratum("4k-h264-le30", 4, 4000,
            lambda c: c["codec"] == "h264" and 7_000_000 <= _pixels(c) <= 9_000_000
            and _sane_fps(c["fps"]) and c["fps"] <= 31 and not c["is_vr"] and c["duration"] >= 600),
    Stratum("4k-h264-50-60", 4, 4000,
            lambda c: c["codec"] == "h264" and 7_000_000 <= _pixels(c) <= 9_000_000
            and 47 <= c["fps"] <= 61 and not c["is_vr"] and c["duration"] >= 600),
    Stratum("above4k-hevc", 4, 2000,
            lambda c: c["codec"] == "hevc" and _pixels(c) > 9_000_000
            and _sane_fps(c["fps"]) and c["duration"] >= 600),
    Stratum("above4k-h264", 1, 2000,
            lambda c: c["codec"] == "h264" and _pixels(c) > 9_000_000
            and _sane_fps(c["fps"]) and c["duration"] >= 600),
    Stratum("hevc-10bit", 2, 4000,
            lambda c: c["codec"] == "hevc" and _pixels(c) <= 9_000_000
            and _sane_fps(c["fps"]) and c["duration"] >= 300,
            probe_check=lambda p: "10" in (p.get("pix_fmt") or "")),
    Stratum("av1", 2, 4000,
            lambda c: c["codec"] == "av1" and _pixels(c) <= 9_000_000
            and _sane_fps(c["fps"]) and c["duration"] >= 300),
    Stratum("legacy-wmv", 1, 10000,
            lambda c: c["codec"] in LEGACY_WMV and _sane_fps(c["fps"]) and c["duration"] >= 180),
    Stratum("legacy-mpeg4", 1, 10000,
            lambda c: c["codec"] == "mpeg4" and _sane_fps(c["fps"]) and c["duration"] >= 180),
    Stratum("le720-h264", 2, 10000,
            lambda c: c["codec"] == "h264" and _pixels(c) < 1_200_000
            and _sane_fps(c["fps"]) and c["duration"] >= 300),
    Stratum("vfr", 1, 6000,
            lambda c: c["codec"] == "h264" and _pixels(c) < 3_000_000
            and _sane_fps(c["fps"]) and c["duration"] >= 300,
            probe_check=lambda p: bool(p.get("vfr"))),
    Stratum("interlaced", 1, 6000,
            lambda c: c["codec"] in {"h264", "mpeg2video", "vc1", "mpeg4"} and _pixels(c) <= 3_000_000
            and _sane_fps(c["fps"]) and c["fps"] <= 31 and c["duration"] >= 180,
            probe_check=lambda p: p.get("field_order") in INTERLACED,
            prefer=lambda c: c["codec"] == "mpeg2video"),
    Stratum("bogus-fps", 1, 6000,
            lambda c: not _sane_fps(c["fps"]) and c["duration"] >= 180),
)


def file_id(path: str) -> str:
    """Anonymous, stable id for a source: results carry this, never the path."""
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:10]


def read_candidates(path) -> list:
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                rows.append({
                    "path": row["path"],
                    "codec": (row.get("codec") or "").strip().lower(),
                    "width": int(float(row.get("width") or 0)),
                    "height": int(float(row.get("height") or 0)),
                    "fps": float(row.get("fps") or 0),
                    "duration": float(row.get("duration") or 0),
                    "bitrate": int(float(row.get("bitrate") or 0)),
                    "is_vr": str(row.get("is_vr") or "").strip().lower() in ("t", "true", "1", "yes"),
                })
            except (KeyError, ValueError):
                continue
    return rows


def find_ffprobe():
    ffmpeg = fp.find_ffmpeg()
    if ffmpeg:
        sibling = Path(ffmpeg).with_name("ffprobe")
        if sibling.is_file():
            return str(sibling)
    return shutil.which("ffprobe")


def _rate(text) -> float:
    numerator, _, denominator = str(text or "").partition("/")
    try:
        return float(numerator) / float(denominator or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_source(ffprobe, path, start_fraction):
    """Stream facts the library database does not carry, plus a VFR check."""
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,avg_frame_rate,field_order"
             ":format=duration,size,bit_rate,format_name",
             "-of", "json", path],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
    except (ValueError, KeyError, IndexError):
        return None
    fmt = data.get("format") or {}
    duration = float(fmt.get("duration") or 0)
    return {
        "codec": stream.get("codec_name"),
        "profile": stream.get("profile"),
        "pix_fmt": stream.get("pix_fmt"),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "r_frame_rate": _rate(stream.get("r_frame_rate")),
        "avg_frame_rate": _rate(stream.get("avg_frame_rate")),
        "field_order": stream.get("field_order") or "unknown",
        "duration": duration,
        "size": int(fmt.get("size") or 0),
        "bit_rate": int(fmt.get("bit_rate") or 0),
        "format_name": fmt.get("format_name"),
        "vfr": _looks_vfr(ffprobe, path, duration * start_fraction),
    }


def _looks_vfr(ffprobe, path, start):
    """True when frame spacing around the excerpt start is irregular.

    Presentation order removes B-frame reordering; timestamp rounding in a
    coarse time base moves a delta by a tick, far below the 25% threshold.
    """
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-read_intervals", f"{start:.3f}%+#300",
             "-show_entries", "packet=pts_time", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    times = []
    for line in result.stdout.split():
        value = line.strip().strip(",")
        if value and value != "N/A":
            try:
                times.append(float(value))
            except ValueError:
                pass
    times.sort()
    deltas = [later - earlier for earlier, later in zip(times, times[1:]) if later > earlier]
    if len(deltas) < 20:
        return None
    median = statistics.median(deltas)
    irregular = sum(1 for delta in deltas if abs(delta - median) > 0.25 * median)
    return irregular / len(deltas) > 0.02


def cmd_pick(args) -> int:
    ffprobe = find_ffprobe()
    if not ffprobe:
        raise SystemExit("ffprobe not found")
    candidates = read_candidates(args.candidates)
    # Hash order rather than random.shuffle, so the pick does not depend on the
    # order the candidates were exported in.
    order = sorted(
        candidates,
        key=lambda c: hashlib.sha1(f"{args.seed}:{c['path']}".encode("utf-8")).hexdigest(),
    )

    used, files, summary = set(), [], []
    for stratum in STRATA:
        pool = [c for c in order if c["path"] not in used and stratum.match(c)]
        if stratum.prefer:
            pool.sort(key=lambda c: 0 if stratum.prefer(c) else 1)
        picked = probes = 0
        for candidate in pool:
            if picked >= stratum.quota or probes >= args.max_probes:
                break
            if not os.access(candidate["path"], os.R_OK):
                continue
            probes += 1
            info = probe_source(ffprobe, candidate["path"], args.start_fraction)
            if info is None or info["width"] <= 0:
                continue
            if stratum.probe_check and not stratum.probe_check(info):
                continue
            used.add(candidate["path"])
            picked += 1
            files.append({
                "id": file_id(candidate["path"]),
                "stratum": stratum.name,
                "path": candidate["path"],
                "excerpt_frames": stratum.frames,
                "db": {key: candidate[key] for key in
                       ("codec", "width", "height", "fps", "duration", "bitrate", "is_vr")},
                "probe": info,
            })
        summary.append({"stratum": stratum.name, "quota": stratum.quota, "picked": picked,
                        "probes": probes, "pool": len(pool)})
        print(f"{stratum.name:18s} {picked}/{stratum.quota} picked "
              f"({probes} probed, {len(pool)} candidates)")

    corpus = {
        "version": 1,
        "seed": args.seed,
        "created": _utc_now(),
        "start_fraction": args.start_fraction,
        "candidates": len(candidates),
        "strata": summary,
        "files": files,
    }
    Path(args.out).write_text(json.dumps(corpus, indent=2), encoding="utf-8")
    print(f"\n{len(files)} files -> {args.out}")
    return 0


def cmd_excerpt(args) -> int:
    """Stream-copy an excerpt of each source, so every level decodes identical
    bytes from fast local storage. L0 measures the source storage separately."""
    corpus_path = Path(args.corpus)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    ffmpeg = fp.find_ffmpeg()
    if not ffmpeg:
        raise SystemExit("ffmpeg not found")
    out_dir = Path(args.dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    start_fraction = float(corpus.get("start_fraction", 0.3))

    for entry in corpus["files"]:
        existing = entry.get("excerpt")
        if existing and Path(existing["file"]).is_file() and not args.force:
            continue
        info = entry["probe"]
        fps = next((value for value in (info["avg_frame_rate"], info["r_frame_rate"], entry["db"]["fps"])
                    if _sane_fps(value)), 30.0)
        frames = int(entry["excerpt_frames"])
        duration = float(info["duration"] or entry["db"]["duration"])
        start = duration * start_fraction
        needed = frames / fps
        if start + needed > duration - 5:
            start = max(0.0, duration - needed - 5)

        target = None
        suffix = Path(entry["path"]).suffix.lstrip(".").lower() or "mkv"
        for container in dict.fromkeys(("mkv", suffix)):
            candidate = out_dir / f"{entry['id']}.{container}"
            command = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                       "-ss", f"{start:.3f}", "-i", entry["path"],
                       "-map", "0:v:0", "-c", "copy", "-an", "-sn", "-dn",
                       "-frames:v", str(frames), str(candidate)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0 and candidate.is_file() and candidate.stat().st_size > 0:
                target = candidate
                break
            candidate.unlink(missing_ok=True)
        if target is None:
            # stderr names the source; keep it out of the console log.
            print(f"{entry['id']} {entry['stratum']:18s} excerpt failed")
            continue
        entry["excerpt"] = {
            "file": str(target),
            "start_seconds": round(start, 3),
            "frames_requested": frames,
            "bytes": target.stat().st_size,
        }
        corpus_path.write_text(json.dumps(corpus, indent=2), encoding="utf-8")
        print(f"{entry['id']} {entry['stratum']:18s} {target.stat().st_size / 1e6:8.1f} MB "
              f"from {start:7.1f}s")
    return 0


# ── host sampling ────────────────────────────────────────────────────────

def _read_proc_stat():
    """Per-CPU (busy, total, iowait) jiffies. Host-wide, not per process."""
    cores = []
    with open("/proc/stat", encoding="ascii") as handle:
        for line in handle:
            name, _, rest = line.partition(" ")
            if not name.startswith("cpu") or name == "cpu":
                continue
            values = [int(value) for value in rest.split()] + [0] * 8
            user, nice, system, idle, iowait, irq, softirq, steal = values[:8]
            busy = user + nice + system + irq + softirq + steal
            cores.append((busy, busy + idle + iowait, iowait))
    return cores


def _rss_kb(pid):
    try:
        with open(f"/proc/{pid}/status", encoding="ascii", errors="replace") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _tree_rss_mb(exclude):
    """RSS of this process plus its direct children (the ffmpeg decoders)."""
    me = os.getpid()
    total = _rss_kb(me)
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in exclude:
            continue
        try:
            with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as handle:
                stat = handle.read()
            parent = int(stat[stat.rfind(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        if parent == me:
            total += _rss_kb(pid)
    return total / 1024


def _cpu_mhz_mean():
    values = []
    try:
        with open("/proc/cpuinfo", encoding="ascii", errors="replace") as handle:
            for line in handle:
                if line.startswith("cpu MHz"):
                    values.append(float(line.split(":")[1]))
    except (OSError, ValueError):
        return None
    return sum(values) / len(values) if values else None


def _find_cpu_temp():
    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            name = (hwmon / "name").read_text().strip()
        except OSError:
            continue
        if name in ("k10temp", "coretemp", "zenpower") and (hwmon / "temp1_input").is_file():
            return hwmon / "temp1_input"
    return None


def _read_temp(path):
    if path is None:
        return None
    try:
        return int(path.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def _num(text):
    try:
        return float(text.strip())
    except (ValueError, AttributeError):
        return None


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def _max(values):
    values = [v for v in values if v is not None]
    return round(max(values), 3) if values else None


def _p95(values):
    values = sorted(v for v in values if v is not None)
    return round(values[min(len(values) - 1, int(0.95 * len(values)))], 3) if values else None


class HostSampler:
    """Background samples of host CPU, process-tree memory and GPU state.

    CPU figures are host-wide, so they include whatever else the machine is
    doing; that is the point of the background window taken before each run.
    """

    def __init__(self, interval=0.2):
        self.interval = interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = []
        self._smi = None
        self._temp_path = _find_cpu_temp()
        self.cpu_t, self.cpu = [], []      # busy cores, busiest core, iowait cores, tree RSS MB
        self.slow_t, self.slow = [], []    # CPU MHz mean, CPU temperature
        self.gpu_t, self.gpu = [], []      # util %, decoder %, VRAM MB, SM MHz, temp, power W

    def start(self):
        self._spawn(self._cpu_loop)
        smi = shutil.which("nvidia-smi")
        if smi:
            self._smi = subprocess.Popen(
                [smi, "--query-gpu=utilization.gpu,utilization.decoder,memory.used,"
                      "clocks.sm,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits", "-lms", str(int(self.interval * 1000)), "-i", "0"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            self._spawn(self._gpu_loop)

    def stop(self):
        self._stop.set()
        if self._smi is not None:
            self._smi.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._smi.wait(timeout=5)
        for thread in self._threads:
            thread.join(timeout=2)

    def _spawn(self, target):
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _cpu_loop(self):
        exclude = {self._smi.pid} if self._smi is not None else set()
        previous = _read_proc_stat()
        tick = 0
        while not self._stop.wait(self.interval):
            current = _read_proc_stat()
            now = clock()
            busy = busiest = iowait = 0.0
            for (busy0, total0, wait0), (busy1, total1, wait1) in zip(previous, current):
                span = total1 - total0
                if span <= 0:
                    continue
                share = (busy1 - busy0) / span
                busy += share
                busiest = max(busiest, share)
                iowait += (wait1 - wait0) / span
            previous = current
            sample = (busy, busiest, iowait, _tree_rss_mb(exclude))
            slow = (_cpu_mhz_mean(), _read_temp(self._temp_path)) if tick % 5 == 0 else None
            with self._lock:
                self.cpu_t.append(now)
                self.cpu.append(sample)
                if slow is not None:
                    self.slow_t.append(now)
                    self.slow.append(slow)
            tick += 1

    def _gpu_loop(self):
        for line in self._smi.stdout:
            values = [_num(value) for value in line.split(",")]
            if len(values) == 6:
                with self._lock:
                    self.gpu_t.append(clock())
                    self.gpu.append(tuple(values))
            if self._stop.is_set():
                break

    def window(self, start, end) -> dict:
        with self._lock:
            cpu = self.cpu[bisect.bisect_left(self.cpu_t, start):bisect.bisect_right(self.cpu_t, end)]
            slow = self.slow[bisect.bisect_left(self.slow_t, start):bisect.bisect_right(self.slow_t, end)]
            gpu = self.gpu[bisect.bisect_left(self.gpu_t, start):bisect.bisect_right(self.gpu_t, end)]
        column = lambda rows, index: [row[index] for row in rows]  # noqa: E731
        return {
            "samples": len(cpu),
            "cpu_busy_mean": _mean(column(cpu, 0)),
            "cpu_busy_p95": _p95(column(cpu, 0)),
            "core_busiest_mean": _mean(column(cpu, 1)),
            "iowait_mean": _mean(column(cpu, 2)),
            "rss_peak_mb": _max(column(cpu, 3)),
            "cpu_mhz_mean": _mean(column(slow, 0)),
            "cpu_temp_max": _max(column(slow, 1)),
            "gpu_util_mean": _mean(column(gpu, 0)),
            "gpu_util_max": _max(column(gpu, 0)),
            "nvdec_util_mean": _mean(column(gpu, 1)),
            "nvdec_util_max": _max(column(gpu, 1)),
            "vram_max_mb": _max(column(gpu, 2)),
            "sm_clock_mean": _mean(column(gpu, 3)),
            "gpu_temp_max": _max(column(gpu, 4)),
            "gpu_power_mean": _mean(column(gpu, 5)),
        }


# ── seam instrumentation ─────────────────────────────────────────────────

class SeamTimers:
    def __init__(self):
        self.probe_s = 0.0
        self.open_s = 0.0
        self.first_frame_at = None
        self.last_frame_at = None
        self.frames = 0
        self.source_next_s = 0.0
        self.to_tensor_s = 0.0
        self.run_window_s = 0.0
        self.normalize_s = 0.0
        self.model_call_s = 0.0
        self.windows = 0
        self.events = []


class _TimedSource:
    """Wraps a FrameSource and times each next(): the wait on the decoder."""

    def __init__(self, source, timers):
        self._source = source
        self._timers = timers

    def __getattr__(self, name):
        return getattr(self._source, name)

    def __iter__(self):
        timers = self._timers
        iterator = iter(self._source)
        while True:
            started = clock()
            try:
                item = next(iterator)
            except StopIteration:
                timers.source_next_s += clock() - started
                return
            now = clock()
            timers.source_next_s += now - started
            if timers.first_frame_at is None:
                timers.first_frame_at = now
            timers.last_frame_at = now
            timers.frames += 1
            yield item


@contextlib.contextmanager
def instrumented(model, sbm, torch, timers):
    """Time the production analysis at its seams without editing it.

    Patches the module globals and instance attributes ``_analyze`` reaches
    through, so the code between the seams runs unmodified. GPU forward time
    comes from CUDA events; the model call's host time additionally covers
    the H2D copy and the device-to-host copy of the outputs.
    """
    runner = model.model
    original_probe = sbm.probe_video
    original_make = sbm.make_video_frame_source
    original_apply = sbm.apply_spec_batch
    original_module = runner.model
    # Older model code converted each frame on arrival; time it when present.
    to_tensor = getattr(model, "_to_frame_tensor", None)
    run_window = model._run_window
    model_call = runner.run_raw_multi_output
    use_events = torch.cuda.is_available() and getattr(runner.device, "type", "") == "cuda"

    def probe_video(*args, **kwargs):
        started = clock()
        try:
            return original_probe(*args, **kwargs)
        finally:
            timers.probe_s += clock() - started

    def make_video_frame_source(*args, **kwargs):
        started = clock()
        try:
            return _TimedSource(original_make(*args, **kwargs), timers)
        finally:
            timers.open_s += clock() - started

    def apply_spec_batch(*args, **kwargs):
        started = clock()
        try:
            return original_apply(*args, **kwargs)
        finally:
            timers.normalize_s += clock() - started

    def timed_to_tensor(*args, **kwargs):
        started = clock()
        try:
            return to_tensor(*args, **kwargs)
        finally:
            timers.to_tensor_s += clock() - started

    def timed_run_window(*args, **kwargs):
        started = clock()
        try:
            return run_window(*args, **kwargs)
        finally:
            timers.run_window_s += clock() - started
            timers.windows += 1

    def timed_model_call(*args, **kwargs):
        started = clock()
        try:
            return model_call(*args, **kwargs)
        finally:
            timers.model_call_s += clock() - started

    def module(*args, **kwargs):
        if not use_events:
            return original_module(*args, **kwargs)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        outputs = original_module(*args, **kwargs)
        end.record()
        timers.events.append((begin, end))
        return outputs

    sbm.probe_video = probe_video
    sbm.make_video_frame_source = make_video_frame_source
    sbm.apply_spec_batch = apply_spec_batch
    if to_tensor is not None:
        model._to_frame_tensor = timed_to_tensor
    model._run_window = timed_run_window
    runner.run_raw_multi_output = timed_model_call
    runner.model = module
    try:
        yield timers
    finally:
        sbm.probe_video = original_probe
        sbm.make_video_frame_source = original_make
        sbm.apply_spec_batch = original_apply
        if to_tensor is not None:
            del model._to_frame_tensor
        del model._run_window
        del runner.run_raw_multi_output
        runner.model = original_module


def _gpu_seconds(torch, events) -> float:
    if not events:
        return 0.0
    torch.cuda.synchronize()
    return sum(begin.elapsed_time(end) for begin, end in events) / 1000.0


# ── levels ───────────────────────────────────────────────────────────────

class Context:
    """Everything the level runners share for one `run` invocation.

    ``model`` is None unless an L4 level was requested: the decode levels
    take their geometry from the model config (or its label sidecar, as the
    model does) and never touch the GPU.
    """

    def __init__(self, config, torch, sbm, model, features, start_fraction, scrub_paths):
        self.config = config
        self.torch = torch
        self.sbm = sbm
        self.model = model
        self.features = features
        self.decode_size = clip_size(config) if model is None else (model._spec.width, model._spec.height)
        self.default_backend = str((config.get("decode") or {}).get("backend", "auto"))
        self.start_fraction = start_fraction
        self.scrub_paths = sorted(scrub_paths, key=len, reverse=True)
        self.infos = {}


def clip_size(config) -> tuple:
    """The model's clip geometry: the config's preprocess_config, else its sidecar."""
    preprocess = config.get("preprocess_config") or {}
    if preprocess.get("width") and preprocess.get("height"):
        return int(preprocess["width"]), int(preprocess["height"])
    sidecar = Path("models") / f"{config.get('model_file_name')}.labels.json"
    labels = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
    if labels.get("frame_width") and labels.get("frame_height"):
        return int(labels["frame_width"]), int(labels["frame_height"])
    raise SystemExit(f"the model config gives no clip geometry and {sidecar} has none either")


def _scrub(text, paths) -> str:
    """Error text can quote a path; results must not."""
    text = str(text)
    for path in paths:
        if path:
            text = text.replace(path, "<file>")
    return text


def _progress_frames(stdout):
    frames = None
    for line in stdout.splitlines():
        if line.startswith("frame="):
            with contextlib.suppress(ValueError):
                frames = int(line[6:].strip())
    return frames


def ffmpeg_null_command(ctx, path, level, info) -> list:
    """The production command for this level, writing to the null muxer."""
    threaded = level in THREADED_LEVELS
    level = THREADED_LEVELS.get(level, level)
    hw = level in ("L1c", "L2c")
    chain = ""
    if level in ("L2", "L2c"):
        plan = fp.plan_scale(info.width, info.height, ctx.decode_size, "exact")
        chain = fp.build_filter_chain(plan, 1, hw)
    command = fp.build_command(ctx.features.path, path, chain, hw, ctx.features.has_fps_mode, None)
    tail = ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    if command[-len(tail):] != tail:
        raise RuntimeError("ffmpeg_pipe.build_command changed shape; update ffmpeg_null_command")
    # L2 keeps the output pixel format: production converts to RGB in the same
    # swscale pass that scales. L1 leaves frames in the decoder's own format.
    command = command[:-len(tail)] + (["-pix_fmt", "rgb24"] if chain else []) + ["-f", "null", "-"]
    command[1:1] = ["-nostats", "-progress", "pipe:1"]
    if threaded:
        command[command.index("-i"):command.index("-i")] = ["-threads", str(len(os.sched_getaffinity(0)))]
    if hw:
        # A failed hwaccel setup is only a warning, after which ffmpeg decodes
        # in software; surface it so the run is not misattributed to NVDEC.
        command[command.index("-loglevel") + 1] = "warning"
    return command


def measure_ffmpeg(ctx, entry, level) -> dict:
    path = entry["excerpt"]["file"]
    command = ffmpeg_null_command(ctx, path, level, ctx.infos[entry["id"]])
    started = clock()
    result = subprocess.run(command, capture_output=True, text=True)
    wall = clock() - started
    detail = {"returncode": result.returncode}
    if level in NVDEC_LEVELS and any(
        marker in result.stderr
        for marker in ("Failed setup for format cuda", "hwaccel initialisation returned error")
    ):
        # NVDEC refused the stream (e.g. H.264 wider than its limit). ffmpeg
        # then decodes in software, or fails at scale_cuda; neither is an
        # NVDEC measurement.
        return {"skipped": "NVDEC could not decode this source; ffmpeg fell back to software"}
    if result.returncode != 0:
        tail = _scrub(result.stderr.strip()[-300:], ctx.scrub_paths)
        return {"error": f"ffmpeg exited {result.returncode}: {tail}"}
    return {"wall_s": wall, "frames": _progress_frames(result.stdout), "detail": detail}


def measure_pipe(ctx, entry, level) -> dict:
    backend = {"L3": "ffmpeg_cpu", "L3c": "ffmpeg_cuda", "L3av": "av"}[level]
    path = entry["excerpt"]["file"]
    started = clock()
    info = fp.probe_video(path)
    probed = clock()
    source = fp.make_video_frame_source(
        path, decode_size=ctx.decode_size, frame_step=1, backend=backend, quality="exact", info=info,
    )
    opened = clock()
    if source.backend_name != backend:
        return {"skipped": f"fell back to {source.backend_name}"}
    frames = 0
    first = last = None
    for _index, _frame in source:
        now = clock()
        frames += 1
        if first is None:
            first = now
        last = now
    ended = clock()
    return {
        "wall_s": ended - started,
        "frames": frames,
        "detail": {
            "backend": source.backend_name,
            "probe_s": probed - started,
            "open_s": opened - probed,
            "first_frame_s": (first - started) if first else None,
            "steady_fps": (frames - 1) / (last - first) if frames > 1 and last > first else None,
        },
    }


def measure_model(ctx, entry, level) -> dict:
    backend = "ffmpeg_cuda" if level == "L4c" else ctx.default_backend
    model = ctx.model
    timers = SeamTimers()
    model.decode_backend = backend
    try:
        with instrumented(model, ctx.sbm, ctx.torch, timers):
            started = clock()
            result = model._analyze(entry["excerpt"]["file"], False)
            wall = clock() - started
        gpu_forward = _gpu_seconds(ctx.torch, timers.events)
    finally:
        model.decode_backend = ctx.default_backend
    if level == "L4c" and result.get("decode_backend") != "ffmpeg_cuda":
        return {"skipped": f"fell back to {result.get('decode_backend')}"}
    first, last = timers.first_frame_at, timers.last_frame_at
    frames = int(result["source_frame_count"])
    boundaries = result.get("boundaries") or []
    return {
        "wall_s": wall,
        "frames": frames,
        "detail": {
            "backend": result.get("decode_backend"),
            "probe_s": timers.probe_s,
            "open_s": timers.open_s,
            "first_frame_s": (first - started) if first else None,
            "steady_fps": (frames - 1) / (last - first) if frames > 1 and last and last > first else None,
            "source_next_s": timers.source_next_s,
            "to_tensor_s": timers.to_tensor_s,
            "run_window_s": timers.run_window_s,
            "normalize_s": timers.normalize_s,
            "model_call_s": timers.model_call_s,
            "gpu_forward_s": gpu_forward,
            "windows": timers.windows,
            # Newer model code overlaps decode with inference and reports how
            # long scoring waited on it; older code reported total - inference.
            "decode_wait_s": result.get("decode_wait_seconds"),
            "prod_decode_s": result.get("decode_seconds"),
            "prod_inference_s": result.get("inference_seconds"),
            "boundaries": len(boundaries),
            "shots": len(result.get("shots") or []),
            "boundaries_sha1": hashlib.sha1(
                json.dumps(boundaries, sort_keys=True).encode("utf-8")).hexdigest()[:12],
        },
    }


def measure_storage(ctx, entry, level, block_mb=256) -> dict:
    """Sequential read from the source's own storage, after evicting that range.

    Decoding reads the source at bitrate x (decode speed / source rate); this
    is the rate the storage can sustain for comparison.
    """
    path = entry["path"]
    size = os.path.getsize(path)
    length = min(block_mb << 20, size)
    offset = (int(size * ctx.start_fraction) >> 20) << 20
    if offset + length > size:
        offset = max(0, size - length)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, offset, length, os.POSIX_FADV_DONTNEED)
        os.lseek(descriptor, offset, os.SEEK_SET)
        started = clock()
        read = 0
        while read < length:
            chunk = os.read(descriptor, min(8 << 20, length - read))
            if not chunk:
                break
            read += len(chunk)
        wall = clock() - started
        # Leave nothing behind in the page cache for the next reader.
        os.posix_fadvise(descriptor, offset, length, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)
    return {
        "wall_s": wall,
        "frames": None,
        "detail": {"bytes": read, "MBps": round(read / wall / 1e6, 1) if wall > 0 else None},
    }


LEVEL_RUNNERS = {
    "L0": measure_storage,
    "L1": measure_ffmpeg,
    "L1t": measure_ffmpeg,
    "L1c": measure_ffmpeg,
    "L2": measure_ffmpeg,
    "L2t": measure_ffmpeg,
    "L2c": measure_ffmpeg,
    "L3": measure_pipe,
    "L3c": measure_pipe,
    "L3av": measure_pipe,
    "L4": measure_model,
    "L4c": measure_model,
}


def applicable(ctx, entry, level):
    """None when the level can run on this source, else why it cannot."""
    if level == "L0":
        return None if os.access(entry["path"], os.R_OK) else "source not readable"
    if level in NVDEC_LEVELS:
        if not ctx.features.has_cuda:
            return "ffmpeg has no CUDA"
        if level != "L1c" and not ctx.features.has_scale_cuda:
            return "ffmpeg has no scale_cuda"
        supported, reason = fp.nvdec_supports(ctx.infos[entry["id"]])
        if not supported:
            return reason
    return None


def inference_microbench(ctx, windows=40, warmup=5) -> dict:
    """Pure inference rate on synthetic frames: the ceiling decode never touches."""
    torch = ctx.torch
    model = ctx.model
    spec = model._spec
    generator = torch.Generator().manual_seed(0)
    frames = [torch.randint(0, 256, (spec.height, spec.width, 3), generator=generator,
                            dtype=torch.uint8).numpy()
              for _ in range(model.window_frames)]
    if hasattr(model, "_to_frame_tensor"):  # older model code took float CHW tensors
        frames = [torch.from_numpy(frame).permute(2, 0, 1).float() for frame in frames]
    # Newer model code places each window at a frame offset; older code did not.
    # Decided before instrumenting, which replaces the method with a wrapper.
    if "offset" in inspect.signature(model._run_window).parameters:
        window_args = (frames, 0, 0)
    else:
        window_args = (frames, 0)
    for _ in range(warmup):
        model._run_window(*window_args, [], [], [])
    timers = SeamTimers()
    with instrumented(model, ctx.sbm, torch, timers):
        started = clock()
        for _ in range(windows):
            model._run_window(*window_args, [], [], [])
        wall = clock() - started
    gpu_forward = _gpu_seconds(torch, timers.events)
    per_window = lambda seconds: round(seconds / windows * 1000, 2)  # noqa: E731
    return {
        "windows": windows,
        "window_frames": model.window_frames,
        "ms_per_window": per_window(wall),
        "fps_equivalent": round(windows * model.window_frames / wall, 1),
        "normalize_ms": per_window(timers.normalize_s),
        "model_call_ms": per_window(timers.model_call_s),
        "gpu_forward_ms": per_window(gpu_forward),
        "other_ms": per_window(timers.run_window_s - timers.normalize_s - timers.model_call_s),
    }


# ── run ──────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_records(path) -> list:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _command_output(command) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def environment(ctx, args, entries) -> dict:
    import torch
    meminfo = {}
    with contextlib.suppress(OSError):
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            meminfo[key] = value.strip()
    cpu_model = next((line.split(":", 1)[1].strip()
                      for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")), None)
    cpufreq = {}
    for name in ("scaling_driver", "scaling_governor", "energy_performance_preference"):
        with contextlib.suppress(OSError):
            cpufreq[name] = Path(f"/sys/devices/system/cpu/cpu0/cpufreq/{name}").read_text().strip()
    gpu = _command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                           "--format=csv,noheader", "-i", "0"])
    import av
    import numpy
    return {
        "type": "env",
        "started_utc": _utc_now(),
        "levels": list(args.levels_parsed),
        "reps": args.reps,
        "gap_s": args.gap,
        "seed": args.seed,
        "files": len(entries),
        "harness_sha1": hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12],
        "git": {
            "head": _command_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])[:12],
            "tracked_changes": bool(_command_output(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"])),
        },
        "ffmpeg": {
            "version": ctx.features.version,
            "has_cuda": ctx.features.has_cuda,
            "has_scale_cuda": ctx.features.has_scale_cuda,
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "pyav": av.__version__,
        "numpy": numpy.__version__,
        "gpu": gpu,
        "cpu": {"model": cpu_model, "logical": os.cpu_count(),
                "affinity": len(os.sched_getaffinity(0)), **cpufreq},
        "memory": {"total": meminfo.get("MemTotal"), "available": meminfo.get("MemAvailable"),
                   "swap_free": meminfo.get("SwapFree")},
        "kernel": platform.release(),
        "model": {
            "config": args.model_config,
            "loaded": ctx.model is not None,
            "window_frames": ctx.model.window_frames if ctx.model is not None else None,
            "clip": f"{ctx.decode_size[0]}x{ctx.decode_size[1]}",
            "context_frames": ctx.config.get("context_frames"),
            "mode": ctx.config.get("mode"),
            "decode_backend": ctx.default_backend,
        },
    }


def _wait_for_quiet(sampler, args):
    """Sleep through the gap and return the background it saw.

    With --quiet-cpu/--quiet-gpu, keep waiting (up to --quiet-timeout) until
    the rest of the machine is below both limits, so a shared host is
    measured in its quiet moments rather than averaged over its busy ones.
    """
    waited = 0.0
    while True:
        time.sleep(args.gap)
        now = clock()
        # The last second only: GPU utilisation is averaged by the driver over
        # up to a second, so the gap's first moments still show the last run.
        background = sampler.window(now - min(1.0, args.gap), now)
        cpu = background["cpu_busy_mean"] or 0.0
        gpu = background["gpu_util_mean"] or 0.0
        quiet = ((args.quiet_cpu is None or cpu <= args.quiet_cpu)
                 and (args.quiet_gpu is None or gpu <= args.quiet_gpu))
        if quiet or waited >= args.quiet_timeout:
            return background, waited, quiet
        waited += args.gap


def cmd_run(args) -> int:
    corpus_path = Path(args.corpus).resolve()
    out_path = Path(args.out).resolve()
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    levels = [level.strip() for level in args.levels.split(",") if level.strip()]
    unknown = [level for level in levels if level not in ALL_LEVELS]
    if unknown:
        raise SystemExit(f"unknown levels: {', '.join(unknown)} (known: {', '.join(ALL_LEVELS)})")
    args.levels_parsed = levels
    wanted_ids = set(args.files.split(",")) if args.files else None
    wanted_strata = set(args.strata.split(",")) if args.strata else None
    entries = [
        entry for entry in corpus["files"]
        if entry.get("excerpt") and Path(entry["excerpt"]["file"]).is_file()
        and (wanted_ids is None or entry["id"] in wanted_ids)
        and (wanted_strata is None or entry["stratum"] in wanted_strata)
    ]
    if not entries:
        raise SystemExit("no corpus entry with an excerpt matches the selection")

    previous = _load_records(out_path) if out_path.exists() else []
    if previous and not args.resume:
        raise SystemExit(f"{out_path} already has results; pass --resume to continue it")
    run_id = next((r["run"] for r in previous if r.get("type") == "env"), None) \
        or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    done = {(r["rep"], r["id"], r["level"]) for r in previous if r.get("type") == "measure"}
    have_files = {r["id"] for r in previous if r.get("type") == "file"}
    have_skips = {(r["id"], r["level"]) for r in previous if r.get("type") == "skip"}
    have_inference = {r["rep"] for r in previous if r.get("type") == "inference"}

    import yaml

    os.chdir(REPO_ROOT)  # the model resolves ./models/ relative to the repo
    logging.getLogger("logger").setLevel(logging.WARNING)
    # Every level takes its clip geometry from the model config; only L4/L4c
    # load the model itself.
    if not Path(args.model_config).is_file():
        raise SystemExit(
            f"model config not found: {args.model_config} (relative to {REPO_ROOT}); "
            f"pass --model-config with the shot-boundary model's yaml"
        )
    with open(args.model_config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    torch = sbm = model = None
    if any(level in ("L4", "L4c") for level in levels):
        import torch
        import lib.model.ai_shot_boundary_model as sbm
        model = sbm.AIShotBoundaryModel(config)
        asyncio.run(model.load())
    features = fp.ffmpeg_features()
    if features is None:
        raise SystemExit("ffmpeg not found")
    scrub = [entry["path"] for entry in entries] + [entry["excerpt"]["file"] for entry in entries]
    scrub.append(str(Path(entries[0]["excerpt"]["file"]).parent))
    ctx = Context(config, torch, sbm, model, features, float(corpus.get("start_fraction", 0.3)), scrub)

    sampler = HostSampler()
    sampler.start()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_path, "a", encoding="utf-8") as out:
            def emit(record):
                record.setdefault("run", run_id)
                out.write(json.dumps(record) + "\n")
                out.flush()

            if not previous:
                emit(environment(ctx, args, entries))

            # Warm-up pass: probe each excerpt, pull it into the page cache and
            # take the software decode's frame count as the reference every
            # other level must reproduce exactly.
            print(f"warm-up: {len(entries)} files", flush=True)
            references = {}
            for entry in entries:
                info = fp.probe_video(entry["excerpt"]["file"])
                ctx.infos[entry["id"]] = info
                warm = measure_ffmpeg(ctx, entry, "L1")
                references[entry["id"]] = warm.get("frames")
                if entry["id"] not in have_files:
                    probe = entry["probe"]
                    emit({
                        "type": "file",
                        "id": entry["id"],
                        "stratum": entry["stratum"],
                        "codec": info.codec,
                        "profile": probe.get("profile"),
                        "pix_fmt": info.pix_fmt,
                        "width": info.width,
                        "height": info.height,
                        "fps": round(info.fps, 3),
                        "field_order": info.field_order,
                        "vfr": probe.get("vfr"),
                        "is_vr": entry["db"].get("is_vr"),
                        "frames_ref": warm.get("frames"),
                        "excerpt_bytes": entry["excerpt"]["bytes"],
                        "source_bit_rate": probe.get("bit_rate"),
                        "source_duration": probe.get("duration"),
                        "source_frames_est": round(probe.get("duration", 0) * info.fps),
                        "warmup_error": warm.get("error"),
                    })
                for level in levels:
                    reason = applicable(ctx, entry, level)
                    if reason and (entry["id"], level) not in have_skips:
                        emit({"type": "skip", "id": entry["id"], "stratum": entry["stratum"],
                              "level": level, "reason": reason})

            # The first NVDEC use in a session pays one-off costs (GPU leaving
            # its idle clocks, CUDA kernel JIT) that no later run sees; take
            # them here rather than in whichever measurement happens to be first.
            if any(level in NVDEC_LEVELS for level in levels):
                for entry in entries:
                    if applicable(ctx, entry, "L2c") is None:
                        warm = measure_ffmpeg(ctx, entry, "L2c")
                        if not warm.get("error") and not warm.get("skipped"):
                            break

            schedule = [(entry, level) for entry in entries for level in levels
                        if applicable(ctx, entry, level) is None]
            total = len(schedule) * args.reps
            finished = sum(1 for rep in range(1, args.reps + 1)
                           for entry, level in schedule if (rep, entry["id"], level) in done)
            session_started, session_count = clock(), 0

            for rep in range(1, args.reps + 1):
                if model is not None and rep not in have_inference:
                    background, _, _ = _wait_for_quiet(sampler, args)
                    micro = inference_microbench(ctx)
                    emit({"type": "inference", "rep": rep, **micro,
                          "bg_gpu_util": background["gpu_util_mean"]})
                    print(f"rep {rep}: inference {micro['ms_per_window']} ms/window "
                          f"({micro['fps_equivalent']} fps equivalent)", flush=True)

                pairs = [(entry, level) for entry, level in schedule
                         if (rep, entry["id"], level) not in done]
                random.Random(f"{args.seed}:{rep}").shuffle(pairs)
                for entry, level in pairs:
                    background, waited, quiet = _wait_for_quiet(sampler, args)
                    reference = references.get(entry["id"])
                    self_before = resource.getrusage(resource.RUSAGE_SELF)
                    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
                    started_utc = _utc_now()
                    started = clock()
                    try:
                        outcome = LEVEL_RUNNERS[level](ctx, entry, level)
                    except Exception as exception:  # noqa: BLE001 - recorded, run continues
                        outcome = {"error": _scrub(f"{type(exception).__name__}: {exception}",
                                                   ctx.scrub_paths)}
                    ended = clock()
                    self_after = resource.getrusage(resource.RUSAGE_SELF)
                    children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
                    wall = outcome.get("wall_s")
                    frames = outcome.get("frames")
                    emit({
                        "type": "measure",
                        "rep": rep,
                        "id": entry["id"],
                        "stratum": entry["stratum"],
                        "level": level,
                        "started_utc": started_utc,
                        "wall_s": round(wall, 4) if wall is not None else None,
                        "frames": frames,
                        "frames_ref": reference,
                        "frames_ok": (frames == reference) if frames is not None and reference else None,
                        "fps": round(frames / wall, 2) if frames and wall else None,
                        "cpu_s_ffmpeg": round((children_after.ru_utime + children_after.ru_stime)
                                              - (children_before.ru_utime + children_before.ru_stime), 3),
                        "cpu_s_python": round((self_after.ru_utime + self_after.ru_stime)
                                              - (self_before.ru_utime + self_before.ru_stime), 3),
                        "host": sampler.window(started, ended),
                        "bg": {
                            "cpu_busy_mean": background["cpu_busy_mean"],
                            "gpu_util_mean": background["gpu_util_mean"],
                            "nvdec_util_mean": background["nvdec_util_mean"],
                            "vram_max_mb": background["vram_max_mb"],
                            "loadavg1": round(os.getloadavg()[0], 2),
                            "waited_s": waited,
                            "quiet": quiet,
                        },
                        "detail": outcome.get("detail"),
                        "skipped": outcome.get("skipped"),
                        "error": outcome.get("error"),
                    })
                    finished += 1
                    session_count += 1
                    rate = (clock() - session_started) / session_count
                    eta_min = rate * (total - finished) / 60
                    status = (outcome.get("error") and "ERROR") or (outcome.get("skipped") and "skipped") \
                        or (f"{frames / wall:9.1f} fps" if frames and wall else f"{wall or 0:8.2f} s")
                    flag = "  FRAME COUNT" if frames is not None and reference and frames != reference else ""
                    print(f"[{finished}/{total}] rep {rep} {entry['id']} {entry['stratum']:16s} "
                          f"{level:4s} {status}{flag}  (eta {eta_min:.0f} min)", flush=True)
    finally:
        sampler.stop()
    print(f"\nresults -> {out_path}")
    return 0


# ── report ───────────────────────────────────────────────────────────────

def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _fmt(value, digits=0):
    if value is None:
        return "–"
    if isinstance(value, float) and digits == 0 and abs(value) < 10:
        return f"{value:.1f}"
    return f"{value:,.{digits}f}"


def _pct(part, whole):
    return None if part is None or not whole else 100.0 * part / whole


def _ratio(numerator, denominator):
    return None if numerator is None or not denominator else numerator / denominator


def cmd_report(args) -> int:
    # Several results files merge: GPU levels may have been measured in a
    # separate run from the CPU ones.
    records = [record for path in args.results for record in _load_records(Path(path))]
    envs = [r for r in records if r.get("type") == "env"]
    env = envs[0] if envs else {}
    files = {r["id"]: r for r in records if r.get("type") == "file"}
    measures = [r for r in records if r.get("type") == "measure"]
    good = [r for r in measures if not r.get("error") and not r.get("skipped") and r.get("wall_s")]
    micro = [r for r in records if r.get("type") == "inference"]
    skips = [r for r in records if r.get("type") == "skip"]
    levels = [level for level in ALL_LEVELS if any(r["level"] == level for r in good)]

    by_pair = {}
    for record in good:
        by_pair.setdefault((record["id"], record["level"]), []).append(record)

    def med(file_id_, level, getter):
        return _median([getter(r) for r in by_pair.get((file_id_, level), [])])

    def fps_of(file_id_, level):
        return med(file_id_, level, lambda r: r.get("fps"))

    def detail(file_id_, level, key):
        return med(file_id_, level, lambda r: (r.get("detail") or {}).get(key))

    print(f"# Shot-boundary ladder, run{'s' if len(envs) > 1 else ''} "
          f"{', '.join(e.get('run', '?') for e in envs) or '?'}\n")
    cpu = env.get("cpu") or {}
    print(f"- Host: {cpu.get('model')}, {cpu.get('affinity')} logical CPUs, "
          f"{cpu.get('scaling_driver') or cpu.get('governor')} {cpu.get('scaling_governor', '')} "
          f"{cpu.get('energy_performance_preference', '')}; GPU {env.get('gpu')}")
    print(f"- ffmpeg: {(env.get('ffmpeg') or {}).get('version')}; torch {env.get('torch')} "
          f"(CUDA {env.get('cuda')}), PyAV {env.get('pyav')}")
    for run_env in envs:
        git = run_env.get("git") or {}
        print(f"- Run {run_env.get('run')}: server {git.get('head')}"
              f"{' with tracked changes' if git.get('tracked_changes') else ''}, "
              f"harness {run_env.get('harness_sha1')}, {run_env.get('reps')} reps, "
              f"{run_env.get('files')} files, levels {', '.join(run_env.get('levels') or [])}")
    model = next(((e.get("model") or {}) for e in envs if (e.get("model") or {}).get("loaded")),
                 env.get("model") or {})
    windows = f"{model.get('window_frames')}-frame windows" if model.get("loaded") else "not loaded"
    print(f"- Model: {windows}, clip {model.get('clip')}, decode backend {model.get('decode_backend')}\n")

    # GPU background only disturbs levels that use the GPU.
    def is_noisy(record):
        background = record.get("bg") or {}
        if (background.get("cpu_busy_mean") or 0) > args.noise_cpu:
            return True
        return record["level"] in GPU_LEVELS and (background.get("gpu_util_mean") or 0) > args.noise_gpu

    noisy = [r for r in good if is_noisy(r)]
    print(f"Background before each measurement: median "
          f"{_fmt(_median([(r.get('bg') or {}).get('cpu_busy_mean') for r in good]), 1)} busy CPUs, "
          f"{_fmt(_median([(r.get('bg') or {}).get('gpu_util_mean') for r in good]), 0)}% GPU. "
          f"{len(noisy)} of {len(good)} measurements started with more than {args.noise_cpu:g} busy CPUs, "
          f"or with more than {args.noise_gpu:g}% GPU on a level that uses it.\n")

    if micro:
        print("## Inference alone (synthetic frames)\n")
        print("| ms/window | fps equivalent | normalize ms | model call ms | GPU forward ms | other ms |")
        print("|---:|---:|---:|---:|---:|---:|")
        print(f"| {_fmt(_median([r['ms_per_window'] for r in micro]), 1)} "
              f"| {_fmt(_median([r['fps_equivalent'] for r in micro]))} "
              f"| {_fmt(_median([r['normalize_ms'] for r in micro]), 1)} "
              f"| {_fmt(_median([r['model_call_ms'] for r in micro]), 1)} "
              f"| {_fmt(_median([r['gpu_forward_ms'] for r in micro]), 1)} "
              f"| {_fmt(_median([r['other_ms'] for r in micro]), 1)} |\n")

    ordered = sorted(files.values(), key=lambda f: (f["stratum"], f["id"]))
    fps_levels = [level for level in levels if level != "L0"]

    print("## Throughput by level (median fps per file)\n")
    print("| id | stratum | source | frames | " + " | ".join(fps_levels) + " | L4/L1 | L4/L3 |")
    print("|---|---|---|---:|" + "---:|" * len(fps_levels) + "---:|---:|")
    for f in ordered:
        cells = [_fmt(fps_of(f["id"], level)) for level in fps_levels]
        l1, l3, l4 = fps_of(f["id"], "L1"), fps_of(f["id"], "L3"), fps_of(f["id"], "L4")
        source = f"{f['codec']} {f['width']}x{f['height']}@{_fmt(f['fps'], 2)}"
        print(f"| {f['id']} | {f['stratum']} | {source} | {f.get('frames_ref') or '–'} | "
              + " | ".join(cells)
              + f" | {_fmt(_ratio(l4, l1), 2)} | {_fmt(_ratio(l4, l3), 2)} |")

    print("\n## By stratum (median of file medians)\n")
    last = "L4" if "L4" in levels else "L3"
    ratio_pairs = [(num, den) for num, den in
                   (("L2", "L1"), ("L3", "L1"), ("L4", "L1"), ("L4", "L3"),
                    ("L1t", "L1"), ("L2t", "L2"), ("L1c", "L1"), ("L3c", "L3"), ("L4c", "L4"))
                   if num in levels and den in levels]
    print(f"| stratum | files | L1 fps | {last} fps | L1 Mpx/s | {last} Mpx/s | "
          + " | ".join(f"{num}/{den}" for num, den in ratio_pairs) + " |")
    print("|---|---:|---:|---:|---:|---:|" + "---:|" * len(ratio_pairs))
    for stratum in sorted({f["stratum"] for f in ordered}):
        members = [f for f in ordered if f["stratum"] == stratum]

        def across(getter):
            return _median([getter(f) for f in members])

        def mpx(f, level):
            fps = fps_of(f["id"], level)
            return fps * f["width"] * f["height"] / 1e6 if fps else None

        ratios = [_fmt(across(lambda f, n=num, d=den: _ratio(fps_of(f["id"], n), fps_of(f["id"], d))), 2)
                  for num, den in ratio_pairs]
        print(f"| {stratum} | {len(members)} "
              f"| {_fmt(across(lambda f: fps_of(f['id'], 'L1')))} "
              f"| {_fmt(across(lambda f: fps_of(f['id'], last)))} "
              f"| {_fmt(across(lambda f: mpx(f, 'L1')))} "
              f"| {_fmt(across(lambda f: mpx(f, last)))} | " + " | ".join(ratios) + " |")

    if "L4" in levels:
        print("\n## Where L4 spends its time (median per file, % of wall)\n")
        print("| id | stratum | wall s | setup s | decoder wait | to tensor | window | "
              "of which normalize | model call | GPU forward | ffmpeg cores | python cores | overlap bound |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for f in ordered:
            wall = med(f["id"], "L4", lambda r: r["wall_s"])
            if wall is None:
                continue
            window = detail(f["id"], "L4", "run_window_s")
            to_tensor = detail(f["id"], "L4", "to_tensor_s")
            l3_wall = med(f["id"], "L3", lambda r: r["wall_s"])
            # With decode on its own thread the wall would be bounded by the
            # slower side: the L3 pipeline, or the consumer's own work.
            bound = max(l3_wall or 0.0, (window or 0.0) + (to_tensor or 0.0))
            ffmpeg_cores = med(f["id"], "L4", lambda r: (r.get("cpu_s_ffmpeg") or 0) / r["wall_s"])
            python_cores = med(f["id"], "L4", lambda r: (r.get("cpu_s_python") or 0) / r["wall_s"])
            print(f"| {f['id']} | {f['stratum']} | {_fmt(wall, 1)} "
                  f"| {_fmt(detail(f['id'], 'L4', 'first_frame_s'), 2)} "
                  f"| {_fmt(_pct(detail(f['id'], 'L4', 'decode_wait_s') if detail(f['id'], 'L4', 'decode_wait_s') is not None else detail(f['id'], 'L4', 'source_next_s'), wall))}% "
                  f"| {_fmt(_pct(to_tensor, wall))}% "
                  f"| {_fmt(_pct(window, wall))}% "
                  f"| {_fmt(_pct(detail(f['id'], 'L4', 'normalize_s'), wall))}% "
                  f"| {_fmt(_pct(detail(f['id'], 'L4', 'model_call_s'), wall))}% "
                  f"| {_fmt(_pct(detail(f['id'], 'L4', 'gpu_forward_s'), wall))}% "
                  f"| {_fmt(ffmpeg_cores, 1)} | {_fmt(python_cores, 1)} "
                  f"| {_fmt(_ratio(wall, bound), 2)}x |")

    if "L0" in levels:
        print("\n## Storage (L0) against what decoding needs\n")
        print("| id | stratum | read MB/s | needed at L1 MB/s | needed at L4 MB/s |")
        print("|---|---|---:|---:|---:|")
        for f in ordered:
            read = detail(f["id"], "L0", "MBps")
            if read is None:
                continue
            byte_rate = (f.get("source_bit_rate") or 0) / 8 / 1e6
            need = lambda level: byte_rate * fps_of(f["id"], level) / f["fps"] \
                if fps_of(f["id"], level) and f.get("fps") else None  # noqa: E731
            print(f"| {f['id']} | {f['stratum']} | {_fmt(read)} "
                  f"| {_fmt(need('L1'), 1)} | {_fmt(need('L4'), 1)} |")

    print("\n## Repeatability: (max − min) / median of fps across reps, median over files\n")
    print("| level | spread | measurements |")
    print("|---|---:|---:|")
    for level in fps_levels:
        spreads = []
        for f in ordered:
            values = [r["fps"] for r in by_pair.get((f["id"], level), []) if r.get("fps")]
            if len(values) >= 2:
                spreads.append((max(values) - min(values)) / statistics.median(values))
        count = sum(1 for r in good if r["level"] == level)
        print(f"| {level} | {_fmt(_pct(_median(spreads), 1))}% | {count} |")

    mismatches = [r for r in good if r.get("frames_ok") is False]
    errors = [r for r in measures if r.get("error")]
    skipped = [r for r in measures if r.get("skipped")]
    print("\n## Checks\n")
    print(f"- Frame count differs from the software reference: {len(mismatches)} measurement(s)")
    for record in mismatches[:40]:
        print(f"  - {record['id']} ({record['stratum']}) {record['level']}: "
              f"{record['frames']} vs {record['frames_ref']}")
    print(f"- Errors: {len(errors)}")
    for record in errors[:20]:
        print(f"  - {record['id']} ({record['stratum']}) {record['level']}: {record['error'][:200]}")
    reasons = {}
    for record in skips + skipped:
        key = (record["level"], record.get("reason") or record.get("skipped"))
        reasons[key] = reasons.get(key, 0) + 1
    print(f"- Not applicable or fell back: {sum(reasons.values())}")
    for (level, reason), count in sorted(reasons.items()):
        print(f"  - {level}: {reason} ({count})")
    return 0


# ── cli ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    pick = sub.add_parser("pick", help="choose a stratified, seeded corpus")
    pick.add_argument("--candidates", required=True, help="TSV of candidate sources")
    pick.add_argument("--out", required=True)
    pick.add_argument("--seed", type=int, default=1)
    pick.add_argument("--start-fraction", type=float, default=0.3,
                      help="where excerpts start, as a fraction of the duration")
    pick.add_argument("--max-probes", type=int, default=300,
                      help="ffprobe calls allowed per stratum before giving up on its quota")

    excerpt = sub.add_parser("excerpt", help="stream-copy an excerpt of each corpus source")
    excerpt.add_argument("--corpus", required=True)
    excerpt.add_argument("--dir", required=True)
    excerpt.add_argument("--force", action="store_true")

    run = sub.add_parser("run", help="measure every (file, level) pair")
    run.add_argument("--corpus", required=True)
    run.add_argument("--out", required=True, help="JSONL results file")
    run.add_argument("--levels", default=",".join(DEFAULT_LEVELS),
                     help=f"comma-separated, from {', '.join(ALL_LEVELS)}")
    run.add_argument("--reps", type=int, default=3)
    run.add_argument("--seed", type=int, default=1, help="seeds the measurement order")
    run.add_argument("--files", help="comma-separated corpus ids")
    run.add_argument("--strata", help="comma-separated strata")
    run.add_argument("--gap", type=float, default=2.0,
                     help="idle seconds before each measurement; its last second is the background sample")
    run.add_argument("--quiet-cpu", type=float, help="wait until background busy CPUs are at most this")
    run.add_argument("--quiet-gpu", type=float, help="wait until background GPU utilisation is at most this %%")
    run.add_argument("--quiet-timeout", type=float, default=300.0,
                     help="longest wait for quiet before measuring anyway, flagged")
    run.add_argument("--resume", action="store_true", help="continue an existing results file")
    run.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG,
                     help="shot-boundary model yaml, relative to the repository (default: %(default)s)")

    report = sub.add_parser("report", help="aggregate results files into markdown")
    report.add_argument("results", nargs="+")
    report.add_argument("--noise-cpu", type=float, default=2.0)
    report.add_argument("--noise-gpu", type=float, default=15.0)

    args = parser.parse_args()
    return {"pick": cmd_pick, "excerpt": cmd_excerpt, "run": cmd_run, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
