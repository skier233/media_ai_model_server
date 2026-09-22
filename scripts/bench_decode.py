"""Compare decode backends on one file: frames, wall time, fps.

    python scripts/bench_decode.py VIDEO [--frames 2000] [--size 128x96]
"""

import argparse
import sys
import time

sys.path.insert(0, ".")

from lib.model.preprocessing_python import ffmpeg_pipe as fp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--frames", type=int, default=2000, help="0 = whole file")
    parser.add_argument("--size", default="128x96")
    parser.add_argument("--backends", default="ffmpeg_cuda,ffmpeg_cpu,av")
    options = parser.parse_args()

    width, height = (int(part) for part in options.size.split("x"))
    info = fp.probe_video(options.video)
    print(f"source: {info.width}x{info.height} {info.codec} {info.pix_fmt} "
          f"{info.fps:.3f}fps {info.duration:.1f}s frames={info.nb_frames}")
    print(f"target: {width}x{height}\n")

    for backend in options.backends.split(","):
        backend = backend.strip()
        try:
            source = fp.make_video_frame_source(
                options.video, decode_size=(width, height), backend=backend, info=info
            )
        except fp.DecodeSetupError as exception:
            print(f"{backend:14s} unavailable: {exception}")
            continue
        if source.backend_name != backend:
            print(f"{backend:14s} fell back to {source.backend_name}, skipping")
            continue

        started = time.perf_counter()
        count = 0
        try:
            for _index, _frame in source:
                count += 1
                if options.frames and count >= options.frames:
                    break
        except fp.DecodeStreamError as exception:
            print(f"{backend:14s} stream error after {count}: {exception}")
            continue
        elapsed = time.perf_counter() - started
        print(f"{backend:14s} {count:6d} frames  {elapsed:7.2f}s  {count / elapsed:8.1f} fps")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
