"""Pure-torch audio ops that replace the two torchaudio features the server used.

torchaudio is in maintenance mode: 2.11 is its final release and it ships no
wheels for newer CUDA variants (cu132+), so importing it next to a current
torch build fails with a CUDA-version mismatch. Everything the server needs
from it is pure torch, so it lives here instead:

  * ``resample``        -- bandlimited sinc interpolation (torchaudio.functional.resample)
  * ``melscale_fbanks`` -- triangular mel filterbank matrix
  * ``spectrogram``     -- STFT -> magnitude/power spectrogram
  * ``MelSpectrogram``  -- Spectrogram + mel filterbank module

The implementations are ports of torchaudio 2.11 (BSD-2-Clause,
Copyright (c) 2017 Facebook Inc. (Soumith Chintala)) and are numerically
identical to it for the argument combinations used in this repo; see
tests/test_audio_ops.py. Keep them that way: the audio models were trained on
features produced by the torchaudio versions of these functions.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Optional, Union

import torch
from torch import Tensor

__all__ = ["resample", "melscale_fbanks", "spectrogram", "MelSpectrogram"]


# ── Resampling ───────────────────────────────────────────────────────


def _get_sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    gcd: int,
    lowpass_filter_width: int,
    rolloff: float,
    resampling_method: str,
    beta: Optional[float],
    device: torch.device,
    dtype: Optional[torch.dtype],
):
    if not (int(orig_freq) == orig_freq and int(new_freq) == new_freq):
        raise ValueError(
            "Frequencies must be integers. Reduce them to an integer ratio first, e.g. "
            "orig_freq=8, new_freq=1 instead of 44100 -> 5512.5."
        )
    if resampling_method not in ("sinc_interp_hann", "sinc_interp_kaiser"):
        raise ValueError(f"Invalid resampling method: {resampling_method}")
    if lowpass_filter_width <= 0:
        raise ValueError("Low pass filter width should be positive.")

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    # Anti-aliasing: remove the highest frequencies relative to the smaller rate.
    base_freq = min(orig_freq, new_freq) * rolloff

    # y[j] = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - j / new_freq)).
    # y[j + new_freq] reuses the filter of y[j] on x shifted by orig_freq, so the
    # resampler is a conv1d with stride orig_freq and new_freq output channels.
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx_dtype = dtype if dtype is not None else torch.float64

    idx = torch.arange(-width, width + orig_freq, dtype=idx_dtype, device=device)[None, None] / orig_freq
    t = torch.arange(0, -new_freq, -1, dtype=dtype, device=device)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)

    if resampling_method == "sinc_interp_hann":
        window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    else:
        if beta is None:
            beta = 14.769656459379492
        beta_tensor = torch.tensor(float(beta))
        window = torch.i0(beta_tensor * torch.sqrt(1 - (t / lowpass_filter_width) ** 2)) / torch.i0(beta_tensor)

    t *= math.pi
    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * scale

    if dtype is None:
        kernels = kernels.to(dtype=torch.float32)
    return kernels, width


def _apply_sinc_resample_kernel(
    waveform: Tensor, orig_freq: int, new_freq: int, gcd: int, kernel: Tensor, width: int
) -> Tensor:
    if not waveform.is_floating_point():
        raise TypeError(f"Expected floating point type for waveform tensor, but received {waveform.dtype}.")

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])
    num_wavs, length = waveform.shape

    waveform = torch.nn.functional.pad(waveform, (width, width + orig_freq))
    resampled = torch.nn.functional.conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    target_length = torch.ceil(torch.as_tensor(new_freq * length / orig_freq)).long()
    resampled = resampled[..., :target_length]

    return resampled.view(shape[:-1] + resampled.shape[-1:])


def resample(
    waveform: Tensor,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: Optional[float] = None,
) -> Tensor:
    """Resample ``waveform`` of shape ``(..., time)`` from ``orig_freq`` to ``new_freq`` Hz.

    Bandlimited sinc interpolation with a Hann (default) or Kaiser window;
    runs on whatever device the waveform is on.
    """
    if orig_freq <= 0 or new_freq <= 0:
        raise ValueError("Original frequency and desired frequency should be positive")
    if orig_freq == new_freq:
        return waveform

    gcd = math.gcd(int(orig_freq), int(new_freq))
    kernel, width = _get_sinc_resample_kernel(
        orig_freq, new_freq, gcd, lowpass_filter_width, rolloff, resampling_method, beta,
        waveform.device, waveform.dtype,
    )
    return _apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd, kernel, width)


# ── Mel filterbank ───────────────────────────────────────────────────


def _hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    if mel_scale not in ("slaney", "htk"):
        raise ValueError('mel_scale should be one of "htk" or "slaney".')
    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq / 700.0))

    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (freq - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels: Tensor, mel_scale: str = "htk") -> Tensor:
    if mel_scale not in ("slaney", "htk"):
        raise ValueError('mel_scale should be one of "htk" or "slaney".')
    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs


def _create_triangular_filterbank(all_freqs: Tensor, f_pts: Tensor) -> Tensor:
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    return torch.max(zero, torch.min(down_slopes, up_slopes))


def melscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
) -> Tensor:
    """Triangular mel filterbank matrix of shape ``(n_freqs, n_mels)``."""
    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')

    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_min = _hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale=mel_scale)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)

    fb = _create_triangular_filterbank(all_freqs, f_pts)

    if norm == "slaney":
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)

    if (fb.max(dim=0).values == 0.0).any():
        warnings.warn(
            "At least one mel filterbank has all zero values. "
            f"The value for `n_mels` ({n_mels}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )
    return fb


# ── Spectrogram ──────────────────────────────────────────────────────


def _get_spec_norms(normalized: Union[str, bool]):
    frame_length_norm, window_norm = False, False
    if isinstance(normalized, str):
        if normalized not in ("frame_length", "window"):
            raise ValueError("Invalid normalized parameter: {}".format(normalized))
        if normalized == "frame_length":
            frame_length_norm = True
        else:
            window_norm = True
    elif isinstance(normalized, bool):
        window_norm = bool(normalized)
    else:
        raise TypeError("Input type not supported")
    return frame_length_norm, window_norm


def spectrogram(
    waveform: Tensor,
    pad: int,
    window: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    power: Optional[float],
    normalized: Union[bool, str],
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
) -> Tensor:
    """STFT of ``waveform`` ``(..., time)`` -> ``(..., freq, time)``.

    ``power=None`` returns the complex spectrum, ``1`` the magnitude, ``2`` the power.
    """
    if pad > 0:
        waveform = torch.nn.functional.pad(waveform, (pad, pad), "constant")

    frame_length_norm, window_norm = _get_spec_norms(normalized)

    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    spec_f = torch.stft(
        input=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=frame_length_norm,
        onesided=onesided,
        return_complex=True,
    )
    spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-2:])

    if window_norm:
        spec_f /= window.pow(2.0).sum().sqrt()
    if power is not None:
        if power == 1.0:
            return spec_f.abs()
        return spec_f.abs().pow(power)
    return spec_f


class MelSpectrogram(torch.nn.Module):
    """Mel spectrogram of a raw waveform: STFT -> |.|^power -> mel filterbank.

    Drop-in for ``torchaudio.transforms.MelSpectrogram`` with the same
    defaults (Hann window, power 2, HTK mel scale, no normalisation).
    Input ``(..., time)``, output ``(..., n_mels, time)``.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 128,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: Union[bool, str] = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        if f_min > self.f_max:
            raise ValueError("Require f_min: {} <= f_max: {}".format(f_min, self.f_max))

        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer("window", window)
        fb = melscale_fbanks(self.n_fft // 2 + 1, self.f_min, self.f_max, n_mels, sample_rate, norm, mel_scale)
        self.register_buffer("fb", fb)

    def forward(self, waveform: Tensor) -> Tensor:
        specgram = spectrogram(
            waveform, self.pad, self.window, self.n_fft, self.hop_length, self.win_length,
            self.power, self.normalized, self.center, self.pad_mode,
        )
        # (..., time, freq) @ (freq, n_mels) -> (..., n_mels, time)
        return torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)
