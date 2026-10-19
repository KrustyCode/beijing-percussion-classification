from functools import lru_cache
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Callable, Dict, List, Tuple
from preprocessing import Dataset, load_split, prepare_data, summarize_feature
from dataclasses import dataclass, field
from tqdm import tqdm
from config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    WINDOW_LENGTH,
)

# Number of MFCC coefficients kept per frame (the `mfcc` default).
N_MFCC: int = 13


@dataclass
class Features:
    """Feature matrix of a set of clips, with their labels, kept index-aligned.

    Ready to pass to a model: ``model.fit(features.X, features.labels)``.

    Attributes:
        X: Feature matrix of shape ``(n_clips, n_features)``, one row per clip
            (see `feature_names` for what each column is).
        labels: Class label of each clip, one of ``LABEL``.
        names: File stem of each clip, e.g. ``"222361__ajaysm__xiaoluo-65_noise-20"``.
        sources: Original clip each row comes from. Use it as the group in
            group-aware cross-validation, so a clip and its augmented copies
            stay on the same side of a fold.
        feature_names: Name of each column of `X`, e.g. ``"mfcc_mean_0"``.
    """
    X: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    labels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=str))
    names: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=str))
    sources: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=str))
    feature_names: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.X)

def complex_to_magnitude(x: np.ndarray) -> np.ndarray:
    """Return the element-wise magnitude |x| of a complex array.

    Args:
        x: Complex (or real) array of any shape.

    Returns:
        Real-valued array with the same shape as `x`.
    """
    return np.abs(x)


def complex_to_power(x: np.ndarray) -> np.ndarray:
    """Return the element-wise power |x|**2 of a complex array.

    Args:
        x: Complex (or real) array of any shape.

    Returns:
        Real-valued array with the same shape as `x`.
    """
    return complex_to_magnitude(x) ** 2

def complex_to_db(x: np.ndarray, ref: float | None = None) -> np.ndarray:
    """Convert a complex array to a decibel-scaled power array.

    Values are floored at 1e-10 before the logarithm so that silent bins
    produce a finite floor instead of -inf.

    Args:
        x: Complex (or real) array of any shape.
        ref: Reference power that maps to 0 dB. Defaults to the peak power
            of `x`, which makes the loudest bin 0 dB.

    Returns:
        Real-valued dB array with the same shape as `x`. An empty input is
        returned unchanged, since a peak reference is undefined for it.
    """
    power = complex_to_power(x)
    if power.size == 0:
        return power
    if ref is None:
        ref = float(np.max(power))
    return 10 * np.log10(np.maximum(power, 1e-10) / ref)

def magnitude_to_db(x: np.ndarray, ref: float | None = None) -> np.ndarray:
    power = x**2
    if power.size == 0:
        return power
    if ref is None:
        ref = float(np.max(power))
    return 10 * np.log10(np.maximum(power, 1e-10) / ref)


def power_to_db(x: np.ndarray, ref: float | None = None) -> np.ndarray:
    """Convert an already power-scaled array to decibels.

    Args:
        x: Non-negative power values of any shape.
        ref: Reference power that maps to 0 dB. Defaults to ``max(x)``.

    Returns:
        Real-valued dB array with the same shape as `x`. An empty input is
        returned unchanged, since a peak reference is undefined for it.
    """
    if x.size == 0:
        return x
    if ref is None:
        ref = float(np.max(x))
    return 10 * np.log10(np.maximum(x, 1e-10) / ref)

# ---------------------------------------------------------------------------
# Window and transform primitives
# ---------------------------------------------------------------------------

def hann_window(win_length: int) -> np.ndarray:
    """Build a periodic-endpoint Hann window.

    The window follows ``w[n] = 0.5 * (1 - cos(2*pi*n / (L - 1)))``, i.e. the
    symmetric form that touches zero at both ends.

    Args:
        win_length: Number of samples in the window. Must be at least 2,
            because the formula divides by ``win_length - 1``.

    Returns:
        Real array of shape (win_length,).

    Raises:
        ValueError: If `win_length` is less than 2.
    """
    if win_length < 2:
        raise ValueError(f"'win_length'({win_length}) must be at least 2")

    n = np.arange(win_length)
    return 0.5 * (1 - np.cos((2 * np.pi * n) / (win_length - 1)))


def _is_power_of_two(n: int) -> bool:
    """Return True if `n` is a positive power of two."""
    return n > 0 and n & (n - 1) == 0
    

# Below this length `fft` evaluates the DFT sum directly instead of splitting
# further. It gives the same result, but stops the recursion from making
# thousands of tiny Python calls on 1- and 2-sample slices.
_FFT_DIRECT_SIZE: int = 32


@lru_cache(maxsize=None)
def _dft_matrix(N: int) -> np.ndarray:
    """Return the N x N DFT matrix ``W[n, k] = exp(-2j*pi*n*k / N)``, read-only."""
    n = np.arange(N)
    W = np.exp(-2j * np.pi * np.outer(n, n) / N)
    W.flags.writeable = False
    return W


@lru_cache(maxsize=None)
def _twiddles(N: int) -> np.ndarray:
    """Return the radix-2 twiddle factors ``exp(-2j*pi*k / N)`` for k < N/2, read-only."""
    t = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    t.flags.writeable = False
    return t


def fft(x: np.ndarray) -> np.ndarray:
    """Compute the discrete Fourier transform via recursive radix-2 FFT.

    This is the textbook Cooley-Tukey decimation-in-time split: the even and
    odd samples are transformed separately and recombined with twiddle
    factors. The radix-2 split only closes if every recursion level has an
    even length, so the input length must be a power of two.

    The transform runs along the last axis, so a ``(n_frames, N)`` stack of
    frames is transformed in one call. Each recursion level combines every
    frequency index and every frame with array arithmetic rather than a
    Python loop, and lengths up to `_FFT_DIRECT_SIZE` use the DFT definition
    directly.

    Args:
        x: Array whose last axis has length ``2**k``.

    Returns:
        Complex array of the same shape holding all frequency bins along the
        last axis.

    Raises:
        ValueError: If the last-axis length is not a power of two.
    """
    N: int = x.shape[-1]

    if not _is_power_of_two(N):
        raise ValueError(f"segment size must be a power of 2, the value given is: {N}")

    if N <= _FFT_DIRECT_SIZE:
        return x @ _dft_matrix(N)   # W is symmetric, so x @ W == W @ x per frame

    x_even: np.ndarray = fft(x[..., ::2])
    x_odd: np.ndarray = fft(x[..., 1::2])

    t = _twiddles(N) * x_odd
    return np.concatenate([x_even + t, x_even - t], axis=-1)


def stft(
    signal: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int | None = None,
    center: bool = True,
) -> np.ndarray:
    """Compute the short-time Fourier transform of a signal.

    Args:
        signal: 1-D real-valued audio samples.
        n_fft: FFT size. Must be a power of two, as required by `fft`.
        hop_length: Samples advanced between consecutive frames. Must be > 0.
        win_length: Length of the Hann window. Defaults to `n_fft`. When it is
            shorter, each windowed frame is zero-padded up to `n_fft`, which
            interpolates the spectrum without adding real resolution. Must not
            exceed `n_fft`.
        center: If True, reflect-pad the signal by ``n_fft // 2`` on both sides
            so frame `t` is centred on sample ``t * hop_length``.

    Returns:
        Complex array of shape ``(n_fft // 2 + 1, n_frames)``. Only the
        non-negative frequencies are kept; for a real signal the rest are
        conjugate duplicates.

    Raises:
        ValueError: If `n_fft` is not a power of two, if `hop_length` is not
            positive, if `win_length` exceeds `n_fft`, or if the signal is too
            short to fill a single frame.

    Steps:
        1. Validate the parameters and default `win_length` to `n_fft`.
        2. Reflect-pad the signal when `center` is set.
        3. Work out how many frames fit, and build the Hann window once.
        4. Stack every frame into one matrix, window it, zero-pad to `n_fft`,
           and FFT all frames in a single call.
        5. Keep the non-negative-frequency half of each spectrum.
    """
    # STEP 1) Validate the parameters and default win_length to n_fft
    if win_length is None:
        win_length = n_fft

    if not _is_power_of_two(n_fft):
        raise ValueError(f"'n_fft'({n_fft}) must be a power of 2 for this fft implementation")

    if hop_length <= 0:
        raise ValueError(f"'hop_length'({hop_length}) must be greater than 0")

    if win_length > n_fft:
        raise ValueError(f"'win_length'({win_length}) must be less than or equal to 'n_fft'({n_fft})")

    # STEP 2) Reflect-pad the signal when center is set
    if center:
        pad: int = n_fft // 2
        signal = np.pad(signal, (pad, pad), mode="reflect")

    # STEP 3) Work out how many frames fit, and build the Hann window once
    s_len: int = len(signal)
    num_frames: int = 1 + (s_len - win_length) // hop_length

    if num_frames < 1:
        raise ValueError(
            f"signal of {s_len} samples is too short for 'win_length'({win_length}); "
            "shorten the window or pass a longer signal"
        )

    window_vals: np.ndarray = hann_window(win_length)

    # STEP 4) Frame matrix (n_frames, win_length): a strided view, no copy
    # until the window multiplies it. Then zero-pad and FFT every frame at once.
    frames = sliding_window_view(signal, win_length)[::hop_length][:num_frames]
    windowed = frames * window_vals

    if win_length < n_fft:
        windowed = np.pad(windowed, ((0, 0), (0, n_fft - win_length)))

    spectrum = fft(windowed)

    # STEP 5) Keep the non-negative-frequency half, as (n_bins, n_frames)
    return np.ascontiguousarray(spectrum[:, :n_fft // 2 + 1].T)


# ---------------------------------------------------------------------------
# Mel scale and mel-based features
# ---------------------------------------------------------------------------

def hz_to_mel(f_hz: float) -> float:
    """Convert a frequency in Hz to the mel scale (O'Shaughnessy formula).

    Args:
        f_hz: Frequency in Hz. Accepts arrays element-wise.

    Returns:
        The corresponding mel value.
    """
    return 2595 * np.log10(1 + f_hz / 700)


def mel_to_hz(m_mel: float) -> float:
    """Convert a mel value back to a frequency in Hz.

    Args:
        m_mel: Mel value. Accepts arrays element-wise.

    Returns:
        The corresponding frequency in Hz.
    """
    return 700 * (10 ** (m_mel / 2595) - 1)


def fmaxmin_check(sample_rate: int, f_max: float | None, f_min: float) -> tuple[float, float]:
    """Resolve and validate the frequency range of a filterbank.

    Args:
        sample_rate: Sampling rate of the signal in Hz. Must be > 0.
        f_max: Upper edge of the filterbank in Hz. `None` defaults to the
            Nyquist frequency, ``sample_rate / 2``, which is the highest
            frequency the STFT can represent.
        f_min: Lower edge of the filterbank in Hz.

    Returns:
        The resolved ``(f_max, f_min)`` pair.

    Raises:
        ValueError: If `sample_rate` is not positive, if `f_min` is negative,
            if `f_max` exceeds Nyquist, or if `f_max` is not above `f_min`.
    """
    if sample_rate <= 0:
        raise ValueError(f"'sample_rate'({sample_rate}) must be greater than 0")

    nyquist = sample_rate / 2

    if f_max is None:
        f_max = nyquist
    elif f_max > nyquist:
        raise ValueError(
            f"'f_max'({f_max}) must not exceed the nyquist frequency ({nyquist})"
        )

    if f_min < 0:
        raise ValueError(f"'f_min'({f_min}) must be greater than or equal to 0")

    if f_max <= f_min:
        raise ValueError(f"'f_max'({f_max}) must be greater than 'f_min'({f_min})")

    return f_max, f_min


def _cached_readonly(fn: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    """Memoise an array-returning function of hashable settings.

    Filterbanks and DCT bases depend only on the settings, not on the signal,
    so they are built once per process instead of once per clip. The cached
    array is shared between callers, so it is made read-only to stop one
    caller's in-place edit from corrupting every later result.
    """
    @lru_cache(maxsize=None)
    def cached(*args, **kwargs):
        arr = fn(*args, **kwargs)
        arr.flags.writeable = False
        return arr

    cached.__doc__ = fn.__doc__
    cached.__name__ = fn.__name__
    return cached


def _triangular_filterbank(bin_points: np.ndarray, n_filter: int, n_fft: int) -> np.ndarray:
    """Build triangular filters from a set of FFT bin edges.

    Filter `m` rises linearly from ``bin_points[m]`` to its peak of 1.0 at
    ``bin_points[m + 1]``, then falls back to 0 at ``bin_points[m + 2]``, so
    adjacent filters overlap by half their width.

    Args:
        bin_points: FFT bin indices of the filter edges, length ``n_filter + 2``.
        n_filter: Number of triangular filters to build.
        n_fft: FFT size, which fixes the number of columns.

    Returns:
        Array of shape ``(n_filter, n_fft // 2 + 1)``. Filters whose edges
        collapse onto the same FFT bin, which happens at low frequencies when
        `n_filter` is large, come out as all-zero rows.
    """
    filterbank = np.zeros((n_filter, n_fft // 2 + 1))

    for m in range(1, n_filter + 1):
        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]

        for k in range(left, center):
            filterbank[m - 1, k] = (k - left) / (center - left)

        for k in range(center, right):
            filterbank[m - 1, k] = (right - k) / (right - center)

    return filterbank


@_cached_readonly
def mel_filterbank(
    n_fft: int,
    sample_rate: int,
    n_mels: int = 128,
    f_min: float = 32.7,
    f_max: float | None = None,
) -> np.ndarray:
    """Build a mel-spaced triangular filterbank.

    Args:
        n_fft: FFT size the filterbank will be applied to.
        sample_rate: Sampling rate of the signal in Hz.
        n_mels: Number of mel bands. Must be > 0.
        f_min: Lower edge in Hz. Defaults to 32.7 Hz (C1).
        f_max: Upper edge in Hz. Defaults to the Nyquist frequency.

    Returns:
        Array of shape ``(n_mels, n_fft // 2 + 1)``, ready to be matrix
        multiplied with an STFT magnitude array.

    Raises:
        ValueError: If `n_mels` is not positive, or from `fmaxmin_check` if
            the frequency range is invalid.

    Steps:
        1. Validate `n_mels` and resolve the frequency range.
        2. Convert the range to mel, where equal spacing matches perception.
        3. Place ``n_mels + 2`` equally spaced points on the mel scale.
        4. Convert those points back to Hz and snap them to FFT bin indices.
        5. Build one triangular filter per consecutive triple of points.
    """
    # STEP 1) Validate n_mels and resolve the frequency range
    if n_mels <= 0:
        raise ValueError(f"'n_mels'({n_mels}) must be greater than 0")

    f_max, f_min = fmaxmin_check(sample_rate, f_max, f_min)

    # STEP 2-3) Space the band edges evenly on the mel scale
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)

    # STEP 4) Back to Hz, then to FFT bin indices
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # STEP 5) Build the triangles
    return _triangular_filterbank(bin_points, n_mels, n_fft)


def mel_spectrogram(
    signal: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int | None = None,
    n_mels: int = 128,
    f_min: float = 32.7,
    f_max: float | None = None,
) -> np.ndarray:
    """Compute a mel spectrogram of the given signal.

    Args:
        signal: 1-D real-valued audio samples.
        sample_rate: Sampling rate of the signal in Hz.
        n_fft: FFT size. Must be a power of two.
        hop_length: Samples advanced between consecutive frames.
        win_length: Hann window length. Defaults to `n_fft`.
        n_mels: Number of mel bands.
        f_min: Lower edge of the filterbank in Hz.
        f_max: Upper edge of the filterbank in Hz. Defaults to Nyquist.

    Returns:
        Real array of shape ``(n_mels, n_frames)``.

    Raises:
        ValueError: Propagated from `stft` and `mel_filterbank`.

    Steps:
        1. Compute the STFT and take its magnitude.
        2. Build the mel filterbank for this FFT size and sample rate.
        3. Project the spectrum onto the mel bands by matrix multiplication.
    """
    # STEP 1) STFT magnitude
    stft_complex = stft(signal, n_fft, hop_length, win_length)
    stft_mag = complex_to_magnitude(stft_complex)

    # STEP 2) Mel filterbank
    filterbank = mel_filterbank(n_fft, sample_rate, n_mels, f_min, f_max)

    # STEP 3) Project onto the mel bands
    return filterbank @ stft_mag


# ---------------------------------------------------------------------------
# DCT and cepstral features
# ---------------------------------------------------------------------------

@_cached_readonly
def _dct_basis(N: int, n_out: int) -> np.ndarray:
    """Orthonormal DCT-II basis of shape ``(n_out, N)``.

    Row `k` holds ``2 * scale_k * cos(pi / N * (n + 0.5) * k)`` over the input
    index `n`, so one matrix product evaluates every cosine-weighted sum.
    """
    n = np.arange(N)
    k = np.arange(n_out)[:, None]
    basis = np.cos(np.pi / N * (n + 0.5) * k)

    # Orthonormal scale, which differs for k == 0
    scale = np.full(n_out, np.sqrt(1 / (2 * N)))
    scale[0] = np.sqrt(1 / (4 * N))

    return 2 * scale[:, None] * basis


def dct(x: np.ndarray, n_out: int) -> np.ndarray:
    """Compute the first `n_out` orthonormal DCT-II coefficients of `x`.

    Applied to a log-spectrum, the DCT decorrelates neighbouring bands and
    packs the spectral envelope into the first few coefficients, which is what
    makes MFCC/LFCC/CQCC compact.

    Args:
        x: 1-D input vector, typically one log-scaled spectral frame, or a
            2-D ``(n_bins, n_frames)`` spectrogram, in which case every column
            is transformed at once.
        n_out: Number of coefficients to keep. Must be > 0. Keeping more than
            ``len(x)`` is allowed but adds no information.

    Returns:
        Real array of shape ``(n_out,)`` for 1-D input or
        ``(n_out, n_frames)`` for 2-D input, scaled with the orthonormal
        ("ortho") normalisation so that the transform preserves energy.

    Raises:
        ValueError: If `n_out` is not positive, or if `x` is empty.

    Steps:
        1. Validate `n_out` and the input length.
        2. Build (or reuse) the scaled cosine basis for this size.
        3. Project the input onto it with one matrix product.
    """
    # STEP 1) Validate n_out and the input length
    if n_out <= 0:
        raise ValueError(f"'n_out'({n_out}) must be greater than 0")

    N = len(x)
    if N == 0:
        raise ValueError("'x' is an empty array, compute failed")

    # STEP 2-3) Cosine-weighted sums with orthonormal scaling, as a projection
    return _dct_basis(N, n_out) @ x


def mfcc(
    signal: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int | None = None,
    n_mels: int = 128,
    n_mfcc: int = 13,
    f_min: float = 32.7,
    f_max: float | None = None,
) -> np.ndarray:
    """Compute mel-frequency cepstral coefficients.

    Args:
        signal: 1-D real-valued audio samples.
        sample_rate: Sampling rate of the signal in Hz.
        n_fft: FFT size. Must be a power of two.
        hop_length: Samples advanced between consecutive frames.
        win_length: Hann window length. Defaults to `n_fft`.
        n_mels: Number of mel bands feeding the DCT.
        n_mfcc: Number of cepstral coefficients to keep. Must be in
            ``1..n_mels``.
        f_min: Lower edge of the filterbank in Hz.
        f_max: Upper edge of the filterbank in Hz. Defaults to Nyquist.

    Returns:
        Real array of shape ``(n_mfcc, n_frames)``.

    Raises:
        ValueError: If `n_mfcc` is outside ``1..n_mels``, or as propagated
            from `mel_spectrogram` and `dct`.

    Steps:
        1. Compute the mel spectrogram.
        2. Convert it to a dB scale, so the DCT sees a log-spectrum.
        3. DCT each time frame down to `n_mfcc` coefficients.
    """
    if not 1 <= n_mfcc <= n_mels:
        raise ValueError(f"'n_mfcc'({n_mfcc}) must be between 1 and 'n_mels'({n_mels})")

    # STEP 1) Mel spectrogram
    mel_spec = mel_spectrogram(signal, sample_rate, n_fft, hop_length, win_length, n_mels, f_min, f_max)

    # STEP 2) Log (dB) scale
    db_mel_spec = power_to_db(mel_spec)

    # STEP 3) DCT per frame (all frames in one projection)
    return dct(db_mel_spec, n_mfcc)


def fft_frequencies(sample_rate: int, n_fft:int) -> np.ndarray:
    nyquist = sample_rate/2
    n_bins = n_fft//2 + 1 

    frequencies = np.linspace(0, nyquist, n_bins)

    return frequencies.astype(np.float32)

def octave_bands(sample_rate: int) -> np.ndarray:
    nyquist = sample_rate / 2
    
    edge = 200.0
    bands = [0.0, edge]
    
    while True:
        edge *= 2
        if edge >= nyquist:
            bands.append(nyquist)
            break
        bands.append(edge)
        
    return np.array(bands, dtype=np.float32)



def spectral_flatness(
        signal: np.ndarray, sample_rate: int, 
        edges: List[int|None]=(0, None),
        n_fft: int = 2048, hop_length: int = 512, min_share=0.01
) -> np.ndarray:
    P = complex_to_power(stft(signal, n_fft, hop_length))
    freqs = fft_frequencies(sample_rate, n_fft)
    edges = [e if e is not None else sample_rate/2 + 1 for e in edges]
    total = P.sum(axis=0) + 1e-20
    flat, share = [], []

    for low, high in zip(edges[:-1], edges[1:]):
        band = P[(freqs >= low) & (freqs<high)]
        s = band.sum(axis=0)/total
        b = np.maximum(band, 1e-10)
        f = np.exp(np.mean(np.log(b), axis=0))/ np.mean(b, axis=0)
        f[s < min_share] = np.nan
        flat.append(f)
        share.append(s)
    return np.array(flat).astype(np.float32), np.array(share).astype(np.float32)


def spectral_contrast(signal: np.ndarray, sample_rate: int, n_fft: int, hop_length:int, quantile=0.02) -> np.ndarray:
    P = complex_to_magnitude(stft(signal, n_fft, hop_length))
    frequencies = fft_frequencies(sample_rate, n_fft)
    edges = octave_bands(sample_rate)

    contrast = []

    for low, high in zip(edges[:-1], edges[1:]):
        band = P[(frequencies >= low) & (frequencies < high)]
        q = max(int(round((band.shape[0] * quantile))), 1)
        sorted_bins = np.sort(band, axis=0)
        peak = np.mean((sorted_bins[-q:]), axis=0)
        valley = np.mean(sorted_bins[:q], axis=0)

        contrast_value = magnitude_to_db(peak) - magnitude_to_db(valley)

        contrast.append(contrast_value)

    return np.array(contrast).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature vectors for the classifiers
# ---------------------------------------------------------------------------

AUGMENTATION_NAMES: Tuple[str, ...] = ("noise", "mp3", "downsample")


def feature_names() -> List[str]:
    """Names of the columns produced by `extract_features`, in order."""
    n_flatness = 1      # spectral_flatness default edges (0, None): one band
    n_contrast = len(octave_bands(SAMPLE_RATE)) - 1
    names = []
    for feature, n in (("mfcc", N_MFCC), ("flatness", n_flatness), ("contrast", n_contrast)):
        for stat in ("mean", "std"):
            names += [f"{feature}_{stat}_{i}" for i in range(n)]
    return names


def extract_features(signal: np.ndarray) -> np.ndarray:
    """Turn one clip into a fixed-length feature vector.

    Each frame-wise feature is summarized over time by its mean and standard
    deviation, so clips of any length give vectors of the same size.

    Args:
        signal: Mono signal at ``SAMPLE_RATE``.

    Returns:
        float32 array, laid out as `feature_names` describes:
        ``[mfcc mean (13), mfcc std (13), flatness mean (1), flatness std (1),
        contrast mean (8), contrast std (8)]``.
    """
    mfcc_mean, mfcc_std = summarize_feature(
        mfcc(signal, sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
             win_length=WINDOW_LENGTH, n_mfcc=N_MFCC))

    # spectral_flatness returns (flatness, energy share); only flatness is a feature
    flatness, _ = spectral_flatness(signal, sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
    flat_mean, flat_std = summarize_feature(flatness)

    contrast_mean, contrast_std = summarize_feature(
        spectral_contrast(signal, sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH))

    return np.concatenate([mfcc_mean, mfcc_std, flat_mean, flat_std,
                           contrast_mean, contrast_std]).astype(np.float32)


def extract_all_features(dataset: Dataset, desc: str = "Extracting") -> Features:
    """Extract the feature vector of every clip in `dataset`.

    Args:
        dataset: Clips to process, at ``SAMPLE_RATE``.
        desc: Label for the progress bar.

    Returns:
        `Features` with one row per clip, plus the clips' labels, names and
        sources.

    Raises:
        ValueError: If a clip produces NaN or infinite features (e.g. a
            completely silent clip, where flatness is undefined).
    """
    names = feature_names()
    X = np.empty((len(dataset), len(names)), dtype=np.float32)
    for i, signal in enumerate(tqdm(dataset.signals, desc=desc)):
        X[i] = extract_features(signal)

    bad = ~np.isfinite(X).all(axis=1)
    if bad.any():
        first = int(np.argmax(bad))
        columns = [n for n, ok in zip(names, np.isfinite(X[first])) if not ok]
        shown = ", ".join(columns[:3]) + (f" and {len(columns) - 3} more" if len(columns) > 3 else "")
        raise ValueError(f"{int(bad.sum())} clip(s) have NaN/inf features, e.g. "
                         f"'{dataset.names[first]}' in {shown}")

    return Features(X, np.asarray(dataset.labels), np.asarray(dataset.names),
                    np.asarray(dataset.sources), names)


def run_extraction(prepare: bool = True) -> Tuple[Dict[str, Features], ...]:
    """Extract features for the original clips and every augmentation.

    Args:
        prepare: If True, run `prepare_data` first (reuses the saved split,
            rebuilds the augmented files, about 50 s). If False, read the
            files a previous `prepare_data` run left on disk with `load_split`.

    Returns:
        ``(original, noise, mp3, downsample)``. Each is a dict with a
        ``"train"`` and a ``"test"`` `Features`, e.g.::

            original, noise, mp3, downsample = run_extraction()
            model.fit(original["train"].X, original["train"].labels)
            y_pred = model.predict(noise["test"].X)     # predict noisy test clips
    """
    if prepare:
        train, test = prepare_data()
        data = {"train": train, "test": test}
    else:
        data = {split: {"original": load_split(split),
                        **{name: load_split(split, [name], include_original=False)
                           for name in AUGMENTATION_NAMES}}
                for split in ("train", "test")}

    return tuple({split: extract_all_features(data[split][name], desc=f"{name} ({split})")
                  for split in ("train", "test")}
                 for name in ("original", *AUGMENTATION_NAMES))