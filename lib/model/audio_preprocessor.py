"""Audio preprocessor — one-to-many model following the video pipeline pattern.

Extracts audio, optionally separates vocals (Demucs), runs energy VAD to
find vocal regions, then spawns one child ItemFuture per overlapping window.
Each child carries preprocessed tensors (filterbank for the embedding model,
mel-spectrogram for the classifier) and flows through the per-window AI model DAG independently.

Pipeline:
  1. PyAV          → decode audio in-process (16 kHz mono)
  2. Demucs        → vocal / accompaniment separation (optional, resamples to 44.1 kHz internally)
  3. Energy VAD    → detect vocal regions in the vocal stem
  4. Windowing     → 4 s windows, 2 s hop, energy-floor filtering
  5. Per window    → compute spec tensors (fbank, mel-spectrogram)
    6. Spawn child   → one ItemFuture per window, picked up by the audio model batchers

After all children complete, batch_awaiter collects results and the
audio_result_postprocessor performs semantic filtering, type-binning,
and per-type centroid embedding computation.
"""

import asyncio
import logging
import os
import time

import numpy as np
import torch
import torchaudio

from lib.async_lib.async_processing import ItemFuture
from lib.model.model import Model
from lib.pipeline.audio_preprocess_spec import AudioPreprocessSpec

# ── Constants ────────────────────────────────────────────────────────
TARGET_SR = 16000


class AudioPreprocessorModel(Model):
    """One-to-many audio preprocessor (analogous to VideoPreprocessorModel).

    The ``specs`` list is injected at pipeline construction time by the
    dynamic AI manager.  It declares what preprocessed tensor each
    downstream model needs.
    """

    # First N output slots are fixed metadata; spec tensors follow.
    FIXED_OUTPUT_COUNT = 5  # dynamic_children, window_index, window_start, window_end, requested_model_names

    def __init__(self, configValues):
        super().__init__(configValues)
        self.logger = logging.getLogger("logger")
        self.specs: list[AudioPreprocessSpec] = []

        # Windowing
        self.window_sec = float(configValues.get("window_sec", 4.0))
        self.hop_sec = float(configValues.get("hop_sec", 2.0))
        self.min_window_sec = float(configValues.get("min_window_sec", 1.0))

        # Energy VAD
        self.energy_floor_db = float(configValues.get("energy_floor_db", -60.0))
        self.adaptive_percentile = float(configValues.get("adaptive_percentile", 25.0))
        self.min_speech_ms = int(configValues.get("min_speech_ms", 100))
        self.min_silence_ms = int(configValues.get("min_silence_ms", 150))
        self.pad_ms = int(configValues.get("pad_ms", 80))

        # Vocal separation (Demucs)
        self.enable_separation = bool(configValues.get("enable_separation", True))
        self.separator_model = str(configValues.get("separator_model", "htdemucs"))
        self._demucs_model = None

        _max_pending = configValues.get("max_pending_items", 0)
        self._max_pending_items = int(_max_pending) if _max_pending else 0

    # ── Demucs lazy loader ───────────────────────────────────────────

    def _get_demucs(self, device):
        """Lazy-load Demucs vocal separation model."""
        if self._demucs_model is not None:
            return self._demucs_model
        try:
            from demucs.pretrained import get_model
        except ImportError:
            raise ImportError(
                "Demucs is required for vocal separation but is not installed. "
                "Install it with: pip install demucs   — or set enable_separation: false "
                "in audio_preprocessor_dynamic.yaml to skip separation."
            )
        self.logger.info(f"[AudioPreprocessor] Loading Demucs ({self.separator_model})...")
        model = get_model(self.separator_model)
        model.to(device)
        model.eval()
        self._demucs_model = model
        self.logger.info(
            f"[AudioPreprocessor] Demucs ready — sources: {model.sources}, sr: {model.samplerate}"
        )
        return model

    # ── Main worker ──────────────────────────────────────────────────

    async def worker_function(self, data):
        loop = asyncio.get_running_loop()

        for item in data:
            try:
                itemFuture = item.item_future
                audio_path = itemFuture[item.input_names[0]]

                start_time = time.perf_counter()
                children = []

                # ── Phase 1: Extract audio ──────────────────────────
                waveform_16k = await loop.run_in_executor(
                    None, _extract_wav, audio_path, TARGET_SR,
                )
                duration = waveform_16k.shape[-1] / TARGET_SR
                self.logger.info(
                    f"[AudioPreprocessor] Audio: {duration:.1f}s from {os.path.basename(audio_path)}"
                )

                # ── Phase 2: Vocal separation ───────────────────────
                if self.enable_separation:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    vocals_16k = await loop.run_in_executor(
                        None, self._separate_vocals, waveform_16k, device,
                    )
                else:
                    vocals_16k = waveform_16k

                # ── Phase 3: Energy VAD ─────────────────────────────
                speech_timestamps = _energy_vad(
                    vocals_16k, TARGET_SR,
                    adaptive_percentile=self.adaptive_percentile,
                    min_speech_ms=self.min_speech_ms,
                    min_silence_ms=self.min_silence_ms,
                    pad_ms=self.pad_ms,
                )
                total_speech = sum(
                    (ts["end"] - ts["start"]) / TARGET_SR for ts in speech_timestamps
                )
                self.logger.info(
                    f"[AudioPreprocessor] VAD: {len(speech_timestamps)} regions, "
                    f"{total_speech:.1f}s speech"
                )

                if not speech_timestamps:
                    self.logger.warning(
                        "[AudioPreprocessor] No speech detected — producing empty result"
                    )
                    preprocess_time = time.perf_counter() - start_time
                    self._record_metrics(itemFuture, preprocess_time, 0,
                                         len(speech_timestamps), total_speech, duration)
                    await itemFuture.set_data(item.output_names[0], children)
                    continue

                # ── Phase 4: Windowed extraction ────────────────────
                window_samples = int(self.window_sec * TARGET_SR)
                hop_samples = int(self.hop_sec * TARGET_SR)
                min_samples = int(self.min_window_sec * TARGET_SR)

                semaphore = None
                if self._max_pending_items > 0:
                    semaphore = asyncio.Semaphore(self._max_pending_items)

                spec_start = self.FIXED_OUTPUT_COUNT
                requested_model_names = itemFuture[item.input_names[2]] if len(item.input_names) > 2 else None
                window_idx = 0
                skipped_quiet = 0

                for ts in speech_timestamps:
                    seg_start = ts["start"]
                    seg_end = ts["end"]
                    if seg_end - seg_start < min_samples:
                        continue

                    pos = seg_start
                    while pos + min_samples <= seg_end:
                        win_end = min(pos + window_samples, seg_end)
                        chunk = vocals_16k[:, pos:win_end]
                        if chunk.shape[1] < min_samples:
                            break

                        # Skip very quiet windows (residual music bleed)
                        rms = float((chunk ** 2).mean().sqrt())
                        db = 20 * np.log10(max(rms, 1e-10))
                        if db < self.energy_floor_db:
                            skipped_quiet += 1
                            pos += hop_samples
                            continue

                        # Compute spec tensors for this window
                        spec_tensors = await loop.run_in_executor(
                            None, _apply_all_specs,
                            chunk, TARGET_SR, self.specs,
                            item.output_names, spec_start,
                        )

                        # Build child payload
                        payload = {
                            item.output_names[1]: window_idx,
                            item.output_names[2]: round(pos / TARGET_SR, 3),
                            item.output_names[3]: round(win_end / TARGET_SR, 3),
                            item.output_names[4]: requested_model_names,
                        }
                        payload.update(spec_tensors)

                        if semaphore is not None:
                            await semaphore.acquire()

                        child = await ItemFuture.create(
                            item, payload, item.item_future.handler
                        )

                        if semaphore is not None:
                            child.future.add_done_callback(
                                lambda _, s=semaphore: s.release()
                            )

                        children.append(child)
                        window_idx += 1
                        pos += hop_samples

                if skipped_quiet:
                    self.logger.debug(
                        f"[AudioPreprocessor] Skipped {skipped_quiet} quiet windows "
                        f"(< {self.energy_floor_db:.0f} dB)"
                    )

                preprocess_time = time.perf_counter() - start_time
                self._record_metrics(
                    itemFuture, preprocess_time, window_idx,
                    len(speech_timestamps), total_speech, duration,
                )
                self.logger.info(
                    f"[AudioPreprocessor] {window_idx} windows in {preprocess_time:.2f}s "
                    f"(separation={'on' if self.enable_separation else 'off'})"
                )

                await itemFuture.set_data(item.output_names[0], children)

            except FileNotFoundError as fnf_error:
                self.logger.error(f"File not found: {fnf_error}")
                itemFuture.set_exception(fnf_error)
            except Exception as e:
                self.logger.error(f"Audio preprocessing error: {e}")
                self.logger.debug("Stack trace:", exc_info=True)
                itemFuture.set_exception(e)

    # ── Demucs vocal separation ──────────────────────────────────────

    def _separate_vocals(self, waveform_16k, device):
        """Run Demucs vocal separation.  Returns vocals at 16 kHz mono [1, T]."""
        from demucs.apply import apply_model

        model = self._get_demucs(device)
        model_sr = model.samplerate  # 44100

        # Upsample to Demucs native SR and fake stereo
        wav = torchaudio.functional.resample(waveform_16k, TARGET_SR, model_sr)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        wav = wav.unsqueeze(0).to(device)  # [1, 2, T]

        self.logger.debug("[AudioPreprocessor] Running Demucs separation...")
        with torch.no_grad():
            sources = apply_model(model, wav, split=True, overlap=0.25, progress=False)

        vocals_idx = model.sources.index("vocals")
        vocals = sources[0, vocals_idx]  # [2, T]

        # Mono → resample back to 16 kHz
        vocals_mono = vocals.mean(dim=0, keepdim=True).cpu()  # [1, T]
        vocals_16k = torchaudio.functional.resample(vocals_mono, model_sr, TARGET_SR)

        self.logger.debug(
            f"[AudioPreprocessor] Vocals: {vocals_16k.shape[1] / TARGET_SR:.1f}s"
        )
        return vocals_16k

    # ── Metrics ──────────────────────────────────────────────────────

    @staticmethod
    def _record_metrics(itemFuture, preprocess_time, window_count,
                        speech_regions, total_speech, duration):
        root_future = getattr(itemFuture, "root_future", itemFuture)
        metrics = getattr(root_future, "_pipeline_metrics", None)
        if metrics is None:
            metrics = {}
            setattr(root_future, "_pipeline_metrics", metrics)
        metrics["audio_preprocess_seconds"] = preprocess_time
        metrics["windows_extracted"] = window_count
        metrics["speech_regions"] = speech_regions
        metrics["total_speech_seconds"] = round(total_speech, 1)
        metrics["audio_duration_seconds"] = round(duration, 1)
        metrics["preprocess_backend"] = "audio_preprocessor"


# ── Thread-pool helpers (no asyncio primitives) ──────────────────────

def _extract_wav(input_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """Extract audio from any media file → mono float32 tensor [1, T] at ``sample_rate``.

    Decodes in-process with PyAV, which links its own libav* libraries, so no
    ``ffmpeg`` executable needs to be on PATH and no temporary WAV is written.

    ``rematrix_maxval=1.0`` reproduces what the previous ``ffmpeg -ac 1 -acodec
    pcm_s16le`` call did implicitly: the CLI sets that swresample option when the
    output format is integer, which normalises the downmix matrix (stereo becomes
    0.5*L + 0.5*R rather than the energy-preserving 0.707 coefficients). Without
    it the waveform comes out ~3 dB hotter, which would shift the VAD energy
    floor and every downstream spectrogram. Letting libswresample do the downmix
    also keeps the layout-aware weighting for multichannel sources, where the
    centre channel carries most of the dialogue.
    """
    import av
    from av.filter import Graph

    chunks: list[np.ndarray] = []

    with av.open(input_path) as container:
        if not container.streams.audio:
            raise RuntimeError(f"No audio stream found in {input_path}")

        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"

        graph = Graph()
        source = graph.add_abuffer(template=stream)
        resample = graph.add(
            "aresample",
            f"rematrix_maxval=1.0:out_sample_rate={sample_rate}:out_chlayout=mono",
        )
        fmt = graph.add(
            "aformat",
            f"sample_fmts=fltp:sample_rates={sample_rate}:channel_layouts=mono",
        )
        sink = graph.add("abuffersink")
        source.link_to(resample)
        resample.link_to(fmt)
        fmt.link_to(sink)
        graph.configure()

        def _drain() -> None:
            # PyAV raises BlockingIOError when the graph needs more input and
            # EOFError once it has been flushed; both subclass the builtins.
            while True:
                try:
                    chunks.append(sink.pull().to_ndarray())
                except (BlockingIOError, EOFError):
                    return

        for frame in container.decode(stream):
            graph.push(frame)
            _drain()
        graph.push(None)  # flush frames buffered inside the filter graph
        _drain()

    if not chunks:
        raise RuntimeError(f"Decoded no audio samples from {input_path}")

    # Each chunk is planar float32 shaped [1, samples] (mono layout).
    waveform = np.concatenate(chunks, axis=1)
    return torch.from_numpy(waveform).to(torch.float32)  # [1, T]


def _waveform_to_fbank(
    waveform: torch.Tensor,
    sample_rate: int,
    n_mels: int = 80,
) -> torch.Tensor:
    """Convert waveform [1, T] to Fbank features [1, num_frames, n_mels].

    Reproduces the reference filterbank pipeline used to train the audio embedding model:
    400-sample window, 160-sample hop, 80 mel filterbank channels.

    SpeechBrain's pipeline:
      STFT → power spectrum (|STFT|²) → mel filterbank → amplitude_to_DB
    where amplitude_to_DB = 10 * log10(clamp(x, min=1e-10)), then top_db=80 clamping.
    """
    fbank_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=n_mels,
        power=2.0,
        window_fn=torch.hamming_window,  # SpeechBrain default
    )
    mel = fbank_transform(waveform)       # [1, n_mels, T']

    # SpeechBrain _amplitude_to_DB: 10 * log10(clamp(x, min=1e-10))
    mel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))

    # SpeechBrain top_db clamping: clamp to within 80 dB of the max
    top_db = 80.0
    max_db = mel.amax(dim=(-2, -1), keepdim=True)
    mel = torch.max(mel, max_db - top_db)

    mel = mel.permute(0, 2, 1)            # [1, T', n_mels]

    # Per-utterance mean normalization (SpeechBrain InputNormalization
    # with norm_type="sentence", std_norm=False — mean only, no std division)
    mean = mel.mean(dim=1, keepdim=True)
    mel = mel - mean
    return mel


def _waveform_to_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    n_mels: int = 128,
    target_length: int = 1024,
) -> torch.Tensor:
    """Convert waveform [1, T] to mel-spectrogram [1, n_mels, target_length]."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,
        hop_length=160,
        n_mels=n_mels,
        power=2.0,
    )
    mel = mel_transform(waveform)           # [1, n_mels, T']
    mel = torch.log(mel + 1e-6)

    t = mel.shape[-1]
    if t < target_length:
        pad = torch.zeros(1, n_mels, target_length - t)
        mel = torch.cat([mel, pad], dim=-1)
    elif t > target_length:
        mel = mel[:, :, :target_length]
    return mel


def _apply_audio_spec(
    waveform: torch.Tensor,
    sample_rate: int,
    spec: AudioPreprocessSpec,
) -> torch.Tensor:
    """Apply an AudioPreprocessSpec to a (possibly windowed) waveform."""
    if sample_rate != spec.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, spec.sample_rate)
        sample_rate = spec.sample_rate

    if spec.use_fbank:
        tensor = _waveform_to_fbank(waveform, sample_rate, n_mels=spec.n_fbank)
    elif spec.use_mel_spectrogram:
        tensor = _waveform_to_mel_spectrogram(
            waveform, sample_rate,
            n_mels=spec.n_mels,
            target_length=spec.target_length,
        )
    else:
        tensor = waveform

    if spec.device == "gpu" and torch.cuda.is_available():
        tensor = tensor.to(torch.device("cuda"))
    if spec.half_precision:
        tensor = tensor.half()
    return tensor


def _apply_all_specs(chunk, sample_rate, specs, output_names, spec_start):
    """Compute all spec tensors for a single window.  Runs in thread pool."""
    result = {}
    for i, spec in enumerate(specs):
        result[output_names[spec_start + i]] = _apply_audio_spec(chunk, sample_rate, spec)
    return result


# ── Energy VAD ───────────────────────────────────────────────────────

def _energy_vad(
    waveform: torch.Tensor,
    sr: int,
    frame_ms: float = 30.0,
    energy_threshold_db: float = -40.0,
    min_speech_ms: int = 200,
    min_silence_ms: int = 150,
    pad_ms: int = 80,
    adaptive: bool = True,
    adaptive_percentile: float = 25.0,
) -> list:
    """Energy-based VAD.  Returns list of ``{start, end}`` in **samples**."""
    wav = waveform.squeeze(0)
    frame_samples = int(sr * frame_ms / 1000)
    num_frames = len(wav) // frame_samples

    energies_db = []
    for i in range(num_frames):
        frame = wav[i * frame_samples:(i + 1) * frame_samples]
        rms = (frame ** 2).mean().sqrt().item()
        db = 20 * np.log10(max(rms, 1e-10))
        energies_db.append(db)

    if adaptive:
        non_dead = [e for e in energies_db if e > -70.0]
        if non_dead:
            energy_threshold_db = float(np.percentile(non_dead, adaptive_percentile))

    is_active = [db > energy_threshold_db for db in energies_db]

    min_speech_frames = max(1, int(min_speech_ms / frame_ms))
    min_silence_frames = max(1, int(min_silence_ms / frame_ms))
    pad_frames = max(0, int(pad_ms / frame_ms))

    regions = []
    in_speech = False
    start = 0
    silence_count = 0

    for i, active in enumerate(is_active):
        if active:
            if not in_speech:
                in_speech = True
                start = i
                silence_count = 0
            else:
                silence_count = 0
        else:
            if in_speech:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    end = i - silence_count + 1
                    if end - start >= min_speech_frames:
                        regions.append((start, end))
                    in_speech = False
                    silence_count = 0

    if in_speech:
        end = num_frames
        if end - start >= min_speech_frames:
            regions.append((start, end))

    total_samples = len(wav)
    timestamps = []
    for rs, re in regions:
        s = max(0, (rs - pad_frames) * frame_samples)
        e = min(total_samples, (re + pad_frames) * frame_samples)
        timestamps.append({"start": s, "end": e})

    # Merge overlapping
    if timestamps:
        merged = [timestamps[0]]
        for ts in timestamps[1:]:
            if ts["start"] <= merged[-1]["end"]:
                merged[-1]["end"] = max(merged[-1]["end"], ts["end"])
            else:
                merged.append(ts)
        timestamps = merged

    return timestamps
