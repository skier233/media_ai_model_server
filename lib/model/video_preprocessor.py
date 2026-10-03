import asyncio
import functools
import logging
import os
import queue
import threading
import time

import torch
from lib.async_lib.async_processing import ItemFuture, model_skip_reason
from lib.model.dense_frames import dense_stream_for
from lib.model.model import Model
from lib.model.preprocessing_python.image_preprocessing import (
    preprocess_video_deffcode,
    preprocess_video_deffcode_gpu,
    preprocess_video_deffcode_auto,
    preprocess_video_av,
    preprocess_video_av_seek,
    preprocess_video_mp_pyav,
    probe_keyframe_interval_seconds,
    probe_video_dimensions,
)
from lib.model.preprocessing_python.ffmpeg_pipe import vr_crop_rect
from lib.pipeline.preprocess_spec import (
    PreprocessSpec,
    apply_spec,
    apply_spec_batch,
    decode_long_edge_for_specs,
)


def compute_auto_pending_frames(per_frame_mb, ram_fraction, assumed_concurrency,
                                _min=32, _max=4096, _fallback=256):
    """RAM-safe per-video cap on in-flight preprocessed frames.

    Preprocessed frames live in system RAM (device="cpu" specs), so the backlog
    is bounded by ``ram_fraction`` of total RAM, split across ``assumed_concurrency``
    concurrent videos, divided by the measured per-frame footprint. Clamped to a
    sane floor/ceiling; falls back to a fixed value if psutil/RAM is unavailable.
    """
    if per_frame_mb <= 0:
        return _fallback
    try:
        import psutil
        total_mb = psutil.virtual_memory().total / (1024 ** 2)
    except Exception:
        return _fallback
    budget_mb = total_mb * ram_fraction
    cap = int(budget_mb / max(assumed_concurrency, 1) / per_frame_mb)
    return max(_min, min(cap, _max))


class VideoPreprocessorModel(Model):
    """Spec-driven video preprocessor.

    The ``specs`` list (set at pipeline construction time by the dynamic AI
    manager) declares every preprocessed tensor the pipeline needs *per
    frame*.  A single video decode pass produces all of them.

    Fixed per-frame outputs (always present):
        output_names[0] → ``dynamic_children`` (parent-level: list of child futures)
        output_names[1] → ``frame_index``
        output_names[2] → ``dynamic_threshold``
        output_names[3] → ``dynamic_return_confidence``
        output_names[4] → ``dynamic_skipped_categories``
        output_names[5] → ``dynamic_requested_model_names``

    Spec-driven outputs (one per spec, starting at index ``FIXED_OUTPUT_COUNT``):
        output_names[FIXED_OUTPUT_COUNT + i] → tensor for ``specs[i]``
    """

    FIXED_OUTPUT_COUNT = 6  # children, frame_index, threshold, return_confidence, skipped_categories, requested_model_names

    def __init__(self, configValues):
        super().__init__(configValues)
        self.frame_interval = configValues.get("frame_interval", 0.5)
        self.logger = logging.getLogger("logger")

        # Populated by the dynamic_ai_manager at pipeline construction.
        self.specs: list[PreprocessSpec] = []
        # Config/file names of the models fed by this preprocessor, injected by
        # DynamicAIManager so the decode can be skipped when none of them will
        # run for a request. None means "not wired by the dynamic manager", and
        # is distinct from an empty set, which means "nothing downstream at all".
        self.downstream_model_names = None
        # Spec key -> config/file names of the models that read that spec,
        # injected by DynamicAIManager. A request that names its models gets
        # only the specs those models read. None means "not wired": every spec
        # is produced for every request.
        self.spec_consumers = None
        # Models that need every frame of the video (``requires_every_frame``),
        # injected by DynamicAIManager. When one runs for a request, the video
        # is decoded in full once and every frame is resized for that model
        # only; everything else still gets just the interval frames.
        self.dense_consumers = []
        # Let the decoder drop frames nothing else references when sampling at
        # an interval. Roughly twice as fast on sources with B-frames; a sample
        # can land on the next decodable frame instead of its own.
        self._skip_nonref = bool(configValues.get("skip_nonref_frames", True))

        # Cap how many preprocessed frames can be in-flight before the
        # preprocessor pauses to let inference catch up.  Preprocessed frames
        # live in system RAM (device="cpu" specs), so this bounds RAM use on
        # long videos with heavy pipelines.  "auto" (default) sizes the cap from
        # total system RAM so low-RAM machines don't OOM; an explicit positive
        # int pins it (overrides auto); 0/null also means auto.
        _max_pending = configValues.get("max_pending_frames", "auto")
        if isinstance(_max_pending, str) and _max_pending.strip().lower() == "auto":
            self._max_pending_frames = 0
            self._max_pending_auto = True
        else:
            self._max_pending_frames = int(_max_pending) if _max_pending else 0
            self._max_pending_auto = self._max_pending_frames <= 0
        # Auto-cap tuning: fraction of total RAM the backlog may use, and how
        # many concurrent videos to budget for.
        self._preprocess_ram_fraction = float(configValues.get("preprocess_ram_fraction", 0.6))
        self._preprocess_assumed_concurrency = max(1, int(configValues.get("preprocess_assumed_concurrency", 3)))

        # When using deffcode_auto, GPU decoding is chosen only when the
        # video's longest edge is >= this threshold.  0 = always GPU.
        self._gpu_min_long_edge = int(configValues.get("gpu_min_long_edge", 3600))

        # Number of parallel decode workers for true-interval (av_parallel)
        # sampling.  Each worker decodes a contiguous time-chunk in its own
        # PyAV container.  Default scales with CPU count (capped at 8, which
        # saturates memory bandwidth on typical multi-core boxes).
        _workers = configValues.get("decode_workers", 0)
        self._decode_workers = int(_workers) if _workers else min(8, os.cpu_count() or 4)

        # How many frames to resize/normalize per batched GPU call.  Larger =
        # fewer GIL-held dispatches (better decode/inference overlap) at the cost
        # of more transient VRAM for the in-flight batch.
        _pbs = configValues.get("preprocess_batch_size", 32)
        self._preprocess_batch_size = max(1, int(_pbs) if _pbs else 32)

        requested_backend = str(configValues.get("preprocess_backend", "deffcode_auto")).lower()

        if requested_backend == "av":
            self._preprocess_backend = "av"
            self._preprocess_callable = preprocess_video_av
            self.logger.info("Video preprocessor using PyAV threaded backend")
        elif requested_backend == "av_seek":
            self._preprocess_backend = "av_seek"
            self._preprocess_callable = preprocess_video_av_seek
            self.logger.info("Video preprocessor using PyAV seek backend")
        elif requested_backend == "av_auto":
            self._preprocess_backend = "av_auto"
            self._preprocess_callable = preprocess_video_av_seek  # default; overridden per-request
            self.logger.info("Video preprocessor using PyAV auto backend (seek when interval >= 1s, threaded otherwise)")
        elif requested_backend == "deffcode_gpu":
            if not torch.cuda.is_available():
                self.logger.warning(
                    "CUDA is not available; falling back to DeFFcode CPU backend for video preprocessing"
                )
                self._preprocess_backend = "deffcode"
                self._preprocess_callable = preprocess_video_deffcode
            else:
                self._preprocess_backend = "deffcode_gpu"
                self._preprocess_callable = preprocess_video_deffcode_gpu
                self.logger.info("Video preprocessor using DeFFcode GPU backend")
        elif requested_backend == "deffcode":
            self._preprocess_backend = "deffcode"
            self._preprocess_callable = preprocess_video_deffcode
            self.logger.info("Video preprocessor using DeFFcode CPU backend")
        else:
            # Default: deffcode_auto — picks GPU or CPU per-video based on resolution.
            if not torch.cuda.is_available():
                self.logger.info(
                    "DeFFcode Auto selected and CUDA is not available; using DeFFcode CPU backend"
                )
                self._preprocess_backend = "deffcode"
                self._preprocess_callable = preprocess_video_deffcode
            else:
                self._preprocess_backend = "deffcode_auto"
                self._preprocess_callable = preprocess_video_deffcode_auto
                self.logger.info(
                    "Video preprocessor using DeFFcode Auto backend (gpu_min_long_edge=%d)",
                    self._gpu_min_long_edge,
                )

    def _should_skip_decode(self, itemFuture, input_names) -> bool:
        """True when no model downstream of this preprocessor will run."""
        downstream = self.downstream_model_names
        if downstream is None:
            return False
        if not downstream:
            # No frame-scope model is active at all (e.g. a deployment with only
            # an asset-scope model installed): nothing can consume these frames.
            return True
        requested = None
        for name in ("dynamic_requested_model_names", "requested_model_names"):
            if name in input_names and name in itemFuture.data:
                requested = itemFuture[name]
                break
        if not requested:
            return False
        normalized = {str(value).strip() for value in requested if str(value).strip()}
        return bool(normalized) and normalized.isdisjoint(downstream)

    def _active_specs(self, requested_model_names) -> list:
        """The specs a request needs: all of them unless it names its models."""
        consumers = self.spec_consumers
        if consumers is None or not requested_model_names:
            return list(self.specs)
        requested = {str(value).strip() for value in requested_model_names if str(value).strip()}
        if not requested:
            return list(self.specs)
        selected = [spec for spec in self.specs if not requested.isdisjoint(consumers.get(spec.key, ()))]
        # A requested model this map does not know still has to get its frames.
        return selected or list(self.specs)

    def _dense_models_for(self, skipped_categories, requested_model_names) -> list:
        """The every-frame models that will run for a request. Uses the same
        rule as their own skip gate, so the decode and the model agree."""
        return [
            model for model in self.dense_consumers
            if model_skip_reason(model, skipped_categories, requested_model_names) is None
        ]

    async def worker_function(self, data):
        for item in data:
            dense_streams = []
            try:
                preprocess_time = 0.0
                itemFuture = item.item_future
                input_data = itemFuture[item.input_names[0]]
                use_timestamps = itemFuture[item.input_names[1]]
                frame_interval = itemFuture[item.input_names[2]] or self.frame_interval
                vr_video = itemFuture[item.input_names[5]]

                # Nothing downstream will run for this request, so decoding the
                # video would produce thousands of child futures that all resolve
                # to Skip(). The per-model skip gate cannot prevent this: it only
                # applies to AI models, and a preprocessor is not one. Return an
                # empty child list before the "no frames" guard below, which
                # would otherwise treat this as a decode failure.
                _skipped_categories = itemFuture[item.input_names[6]] if len(item.input_names) > 6 else None
                _requested_names = itemFuture[item.input_names[7]] if len(item.input_names) > 7 else None
                dense_models = self._dense_models_for(_skipped_categories, _requested_names)
                frames_wanted = not self._should_skip_decode(itemFuture, item.input_names)
                if not frames_wanted and not dense_models:
                    self.logger.info(
                        "Skipping video decode for '%s': no requested model uses this preprocessor.",
                        input_data,
                    )
                    await itemFuture.set_data(item.output_names[0], [])
                    continue

                active_specs = self._active_specs(_requested_names) if frames_wanted else []
                active_keys = {spec.key for spec in active_specs}

                # One stream per every-frame model, grouped by clip size so each
                # size is resized once however many models share it.
                dense_sinks = []
                if dense_models:
                    root_future = getattr(itemFuture, "root_future", itemFuture)
                    by_size = {}
                    for model in dense_models:
                        size = tuple(model.dense_frame_size())
                        stream = dense_stream_for(
                            root_future, getattr(model, "config_name", None) or model.model_file_name, *size)
                        by_size.setdefault(size, []).append(stream)
                        dense_streams.append(stream)
                    dense_sinks = list(by_size.items())

                children = []
                frame_count = 0
                preprocess_callable = self._preprocess_callable
                backend_used = self._preprocess_backend

                # av_auto: choose between true-interval parallel decode and the
                # seek backend based on how frame_interval compares to the GOP.
                #
                # When frame_interval < GOP, seeking would snap several adjacent
                # targets onto the *same* keyframe (duplicate frames), so we must
                # actually decode each GOP to produce a distinct frame per
                # target — done in parallel across CPU cores for speed.
                #
                # When frame_interval >= GOP, each target lands on its own
                # keyframe, so the cheap seek backend yields distinct frames
                # without full decode (ideal for sparse sampling of long files).
                _source_size = probe_video_dimensions(input_data)
                if dense_sinks:
                    # Every frame has to be decoded anyway, so the parallel
                    # decode serves the interval frames from the same pass.
                    dense_crop = None
                    if vr_video:
                        if _source_size is None:
                            raise RuntimeError(f"could not read the dimensions of '{input_data}'")
                        dense_crop = vr_crop_rect(*_source_size)
                    preprocess_callable = functools.partial(
                        preprocess_video_mp_pyav,
                        decode_workers=self._decode_workers,
                        dense_sinks=dense_sinks,
                        dense_crop=dense_crop,
                        emit_interval=frames_wanted,
                    )
                    backend_used = "mp_pyav"
                    self.logger.info(
                        "Decoding every frame of '%s' for %s%s",
                        input_data,
                        ", ".join(getattr(m, "config_name", None) or m.model_file_name for m in dense_models),
                        "" if frames_wanted else " (no interval frames requested)",
                    )
                elif backend_used == "av_auto":
                    gop_seconds = probe_keyframe_interval_seconds(input_data)
                    if gop_seconds is not None and gop_seconds > 0:
                        use_parallel = frame_interval < gop_seconds
                    else:
                        # GOP unknown: assume dense sampling needs true decode.
                        use_parallel = frame_interval < 4.0
                    if use_parallel:
                        # Parallel PyAV decode in worker *processes*: true
                        # distinct frames at the requested interval, decoded with
                        # separate GILs so the decoders overlap the async
                        # inference pipeline instead of contending with it.
                        preprocess_callable = functools.partial(
                            preprocess_video_mp_pyav,
                            decode_workers=self._decode_workers,
                            skip_nonref=self._skip_nonref,
                        )
                        backend_used = "mp_pyav"
                    else:
                        preprocess_callable = preprocess_video_av_seek
                        backend_used = "av_seek"
                    self.logger.info(
                        "av_auto: frame_interval=%.3fs gop=%s → %s backend",
                        frame_interval,
                        f"{gop_seconds:.3f}s" if gop_seconds else "unknown",
                        backend_used,
                    )

                # Determine the max decode resolution from the specs.
                # If every spec has a finite cap we can downscale at decode
                # time → dramatically less per-frame data. The cap depends on
                # the source's shape, so a source that cannot be probed is
                # decoded at native resolution rather than guessed at.
                _max_decode_long_edge = 0
                if _source_size is not None:
                    _max_decode_long_edge = decode_long_edge_for_specs(
                        active_specs, _source_size[0], _source_size[1], vr_video=bool(vr_video))

                # Decode at native (or capped) resolution — apply_spec handles
                # per-model resize/normalize/device.  With norm_config=-1 the
                # backends yield (frame_index, tensor) where tensor is a
                # [0,255] float32 CHW tensor — exactly what apply_spec expects.
                try:
                    frame_source = preprocess_callable(
                        input_data,
                        frame_interval,
                        0,             # image_size=0: no resize at decode level
                        False,         # use_half_precision=False: fp32 base
                        "cpu",         # device: keep on CPU — apply_spec handles resize/norm
                        use_timestamps,
                        vr_video=vr_video,
                        norm_config=-1,            # skip normalization
                        max_decode_long_edge=_max_decode_long_edge,
                        gpu_min_long_edge=self._gpu_min_long_edge,
                    )
                except Exception as exc:
                    if preprocess_callable is preprocess_video_deffcode_gpu:
                        self.logger.warning(
                            "DeFFcode GPU preprocessing failed for '%s'. Falling back to DeFFcode CPU. Error: %s",
                            input_data, exc,
                        )
                        preprocess_callable = preprocess_video_deffcode
                        backend_used = "deffcode"
                        frame_source = preprocess_callable(
                            input_data, frame_interval, 0, False, "cpu",
                            use_timestamps, vr_video=vr_video, norm_config=-1,
                            max_decode_long_edge=_max_decode_long_edge,
                        )
                    elif preprocess_callable is preprocess_video_deffcode_auto:
                        self.logger.warning(
                            "DeFFcode Auto preprocessing failed for '%s'. Falling back to DeFFcode CPU. Error: %s",
                            input_data, exc,
                        )
                        preprocess_callable = preprocess_video_deffcode
                        backend_used = "deffcode"
                        frame_source = preprocess_callable(
                            input_data, frame_interval, 0, False, "cpu",
                            use_timestamps, vr_video=vr_video, norm_config=-1,
                            max_decode_long_edge=_max_decode_long_edge,
                        )
                    else:
                        raise

                spec_start = self.FIXED_OUTPUT_COUNT
                # Explicit cap: build the semaphore up front.  Auto cap: defer
                # until the first frame so we can size it from the real
                # per-frame RAM footprint (created lazily in the consumer loop).
                frame_semaphore = None
                if not self._max_pending_auto and self._max_pending_frames > 0:
                    frame_semaphore = asyncio.Semaphore(self._max_pending_frames)
                frame_iterator = iter(frame_source)
                loop = asyncio.get_running_loop()

                # ---- Continuous producer-consumer pipeline ----
                #
                # Previously, decode→transform→apply_spec was done one
                # frame at a time via ``await run_in_executor()``, which
                # forced an event-loop round-trip between every frame
                # (~10 ms overhead × 1863 frames = ~19 s wasted).
                #
                # Now a background thread runs the full loop without
                # ever yielding to the event loop.  Results go into a
                # bounded queue; the async consumer just pulls them out
                # and creates ItemFutures.  Three-stage pipeline:
                #
                #   Thread A (av_seek prefetch): seek+decode (GIL-free)
                #   Thread B (producer below) : transform+apply_spec
                #   Async consumer            : ItemFuture → GPU
                #
                # All three stages overlap.  Total ≈ max(decode, GPU).

                _FRAME_DONE = object()
                _QUEUE_DEPTH = 64
                _result_q = queue.Queue(maxsize=_QUEUE_DEPTH)
                _producer_error = []
                _cumulative_cpu = [0.0]   # mutable float for thread
                _requested_model_names = _requested_names
                # A spec no requested model reads is not produced, but its key
                # is still set: the models wired to it are triggered by it,
                # and that is where they report themselves skipped.
                _unused_spec_names = [
                    item.output_names[spec_start + index]
                    for index, spec in enumerate(self.specs) if spec.key not in active_keys
                ]

                def _frame_producer():
                    # Apply each spec on its native device, in batches.  A batch
                    # is one H2D copy + a few resize/normalize kernels per N
                    # frames, instead of a CPU resize+normalize per frame.  The
                    # per-frame variant holds the GIL the whole time and
                    # ping-pongs with the async inference pipeline; batching
                    # pushes that work onto the (idle) GPU in a handful of
                    # dispatches, so decode and inference actually overlap.
                    _specs = list(active_specs)
                    _spec_names = [item.output_names[spec_start + self.specs.index(sp)] for sp in _specs]
                    _out_names = item.output_names
                    _ss = spec_start
                    _batch = []  # list of (frame_index, raw_frame CHW)

                    def _flush():
                        if not _batch:
                            return
                        t0 = time.perf_counter()
                        raws = torch.stack([rf for _, rf in _batch], dim=0)  # [N,C,H,W]
                        spec_batches = [apply_spec_batch(raws, sp) for sp in _specs]
                        del raws
                        for bi, (fidx, _) in enumerate(_batch):
                            st = {
                                _spec_names[si]: spec_batches[si][bi]
                                for si in range(len(_specs))
                            }
                            _result_q.put((fidx, st))
                        _cumulative_cpu[0] += time.perf_counter() - t0
                        _batch.clear()

                    try:
                        while True:
                            try:
                                frame_data = next(frame_iterator)
                            except StopIteration:
                                break
                            _batch.append((frame_data[0], frame_data[1]))
                            del frame_data
                            if len(_batch) >= self._preprocess_batch_size:
                                _flush()
                        _flush()
                    except Exception as exc:
                        _producer_error.append(exc)
                    finally:
                        _result_q.put(_FRAME_DONE)

                producer = threading.Thread(
                    target=_frame_producer, daemon=True,
                    name="video-preprocess-producer",
                )
                producer.start()

                try:
                    _threshold = itemFuture[item.input_names[3]]
                    _return_conf = itemFuture[item.input_names[4]]
                    _skipped_cats = itemFuture[item.input_names[6]]
                    _out_names = item.output_names

                    def _pull_batch():
                        """Block for the first item, then drain non-blocking."""
                        first = _result_q.get()
                        if first is _FRAME_DONE:
                            return [first]
                        items = [first]
                        while len(items) < 64:
                            try:
                                nxt = _result_q.get_nowait()
                                items.append(nxt)
                                if nxt is _FRAME_DONE:
                                    break
                            except queue.Empty:
                                break
                        return items

                    while True:
                        batch = await loop.run_in_executor(None, _pull_batch)

                        hit_done = False
                        for result in batch:
                            if result is _FRAME_DONE:
                                hit_done = True
                                break

                            frame_index, spec_tensors = result
                            frame_count += 1

                            # Auto cap: size the in-flight limit from the real
                            # per-frame RAM footprint on the first frame.
                            if frame_semaphore is None and self._max_pending_auto:
                                _pf_mb = sum(
                                    t.element_size() * t.nelement()
                                    for t in spec_tensors.values() if isinstance(t, torch.Tensor)
                                ) / (1024 ** 2)
                                _cap = compute_auto_pending_frames(
                                    _pf_mb, self._preprocess_ram_fraction,
                                    self._preprocess_assumed_concurrency)
                                self.logger.info(
                                    "Auto max_pending_frames=%d (per-frame %.1f MB, "
                                    "%.0f%% RAM budget, ~%d concurrent videos assumed)",
                                    _cap, _pf_mb, self._preprocess_ram_fraction * 100,
                                    self._preprocess_assumed_concurrency)
                                frame_semaphore = asyncio.Semaphore(_cap)

                            payload = {
                                _out_names[1]: frame_index,
                                _out_names[2]: _threshold,
                                _out_names[3]: _return_conf,
                                _out_names[4]: _skipped_cats,
                                _out_names[5]: _requested_model_names,
                            }
                            payload.update(spec_tensors)
                            for _unused in _unused_spec_names:
                                payload[_unused] = None
                            if frame_semaphore is not None:
                                await frame_semaphore.acquire()
                            child = await ItemFuture.create(item, payload, item.item_future.handler)
                            if frame_semaphore is not None:
                                child.future.add_done_callback(lambda _, s=frame_semaphore: s.release())
                            children.append((frame_index, child))

                        if hit_done:
                            if _producer_error:
                                raise _producer_error[0]
                            break

                    preprocess_time = _cumulative_cpu[0]
                    producer.join(timeout=10.0)
                finally:
                    # Drain queue so the producer isn't stuck on put().
                    while producer.is_alive():
                        try:
                            _result_q.get_nowait()
                        except queue.Empty:
                            break
                    producer.join(timeout=5.0)
                    close = getattr(frame_source, "close", None)
                    if callable(close):
                        close()

                if frame_count > 0:
                    avg_time = preprocess_time / frame_count
                    root_future = getattr(itemFuture, "root_future", itemFuture)
                    metrics = getattr(root_future, "_pipeline_metrics", None)
                    if metrics is None:
                        metrics = {}
                        setattr(root_future, "_pipeline_metrics", metrics)
                    metrics["preprocess_seconds"] = metrics.get("preprocess_seconds", 0.0) + preprocess_time
                    metrics["frames_preprocessed"] = metrics.get("frames_preprocessed", 0) + frame_count
                    metrics["preprocess_backend"] = backend_used
                    metrics["average_frame_preprocess_seconds"] = avg_time
                    self.logger.info(
                        "Preprocessed %s frames in %.4f seconds (avg %.4f s/frame) using %s backend.",
                        frame_count, preprocess_time, avg_time, backend_used,
                    )
                elif frames_wanted:
                    error_msg = f"No frames were produced during preprocessing of '{input_data}' using {backend_used} backend."
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)

                # Frames may have been produced out of timestamp order (the
                # parallel backend decodes independent chunks concurrently).
                # Downstream timespan assembly assumes monotonic frame_index,
                # so emit the children sorted by their frame time.
                children.sort(key=lambda fc: (fc[0] is None, fc[0]))
                ordered_children = [child for _, child in children]
                await itemFuture.set_data(item.output_names[0], ordered_children)
            except FileNotFoundError as fnf_error:
                self.logger.error(f"File not found error: {fnf_error}")
                self.logger.debug("Stack trace:", exc_info=True)
                _fail_streams(dense_streams, fnf_error)
                itemFuture.set_exception(fnf_error)
            except IOError as io_error:
                self.logger.error(f"IO error (video might be corrupted): {io_error}")
                self.logger.debug("Stack trace:", exc_info=True)
                _fail_streams(dense_streams, io_error)
                itemFuture.set_exception(io_error)
            except Exception as e:
                self.logger.error(f"An unexpected error occurred: {e}")
                self.logger.debug("Stack trace:", exc_info=True)
                _fail_streams(dense_streams, e)
                itemFuture.set_exception(e)


def _fail_streams(streams, exception):
    """Tell every-frame models still waiting on this request's decode that it
    failed. A stream that already finished ignores it."""
    for stream in streams:
        if stream.segment_counts is None:
            threading.Thread(target=stream.fail, args=(exception,), daemon=True).start()