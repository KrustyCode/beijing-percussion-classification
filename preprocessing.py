"""Data loading, splitting, augmentation and scaling for the percussion dataset.

Pipeline (run with ``python preprocessing.py`` or call `prepare_data`):
    1. Load every ``.wav`` clip in ``FOLDER_PATH`` as mono float32 at
       ``SAMPLE_RATE`` and read its label from the file name.
    2. Split the clips into train / test. If ``SPLIT_PATH`` exists, the saved
       split is reused; otherwise a new stratified split is made and saved
       there. Pass ``resplit=True`` to make a new split on purpose.
    3. Write augmented copies of each split to its own folder. The original
       dataset is never modified.
    4. Optionally (``max_duration`` given) trim the silence from every clip
       and cut it to at most ``max_duration`` seconds, see `clip_datasets`.
       This only changes the returned signals; the files on disk and the
       manifest keep the full-length audio.
    5. Write a manifest CSV that lists every original and augmented file.
    6. Return train and test, each separated per augmentation. Test copies
       are also separated per augmentation value, because every test clip
       gets the same fixed values (e.g. one Dataset per SNR in TEST_SNR)::

        train, test = prepare_data()
        train["original"], train["noise"], train["mp3"], train["downsample"]
        train_all = merge_datasets(*train.values())

        test["original"]                  # clean test clips
        test["noise"][30], test["noise"][0]        # one Dataset per TEST_SNR
        test["mp3"][2], test["downsample"][8000]   # per TEST_COMPRESSION / DOWN_SR

    Train copies stay one Dataset per augmentation: their values are drawn
    at random per clip, so they do not form fixed levels.

Loading the prepared data later (no re-splitting, no re-augmenting)::

    train = load_split("train", augmentations=["noise", "mp3", "downsample"])
    test_noisy = load_split("test", augmentations=["noise"], include_original=False)
    test_noisy.by_param()                 # {30: Dataset, 20: Dataset, ...}

`load_split` returns the full-length clips from disk. To trim and clip
them like ``prepare_data(max_duration=...)`` does, use `clip_all` on a
Dataset's signals, or `clip_datasets` on the whole ``{"train": ..., "test": ...}``
dict.

Augmentations (settings live in config.py):
    noise       Additive white Gaussian noise at a given SNR (dB).
                train: N_AUG_COPIES distinct SNRs drawn from TRAIN_SNR.
                test:  one copy per SNR in TEST_SNR.
    mp3         MP3 encoding at a LAME VBR quality (0 best, 9 worst).
                train: N_AUG_COPIES distinct qualities drawn from TRAIN_COMPRESSION.
                test:  one copy per quality in TEST_COMPRESSION.
    downsample  Resampled to a lower sample rate and saved at that rate.
                train and test: one copy per rate in DOWN_SR.

Output layout::

    augmented_dataset/                 AUGMENTED_FOLDER_PATH
    ├── noise/                         NOISE_FOLDER_PATH
    │   ├── train/<clip>_noise-<snr>.wav
    │   └── test/<clip>_noise-<snr>.wav
    ├── mp3/                           MP3_FOLDER_PATH
    │   ├── train/<clip>_mp3-<quality>.mp3
    │   └── test/<clip>_mp3-<quality>.mp3
    ├── downsample/                    DOWNSAMPLE_FOLDER_PATH
    │   ├── train/<clip>_downsample-<sr>.wav
    │   └── test/<clip>_downsample-<sr>.wav
    └── manifest.csv                   MANIFEST_PATH

Augmented copies are made separately for each split, so a test clip never
has a copy in the training data. Train copies are for training; test copies
are for measuring how robust a model is to each degradation. Every run
rebuilds the ``train/`` and ``test/`` folders from scratch, so copies left
over from an older split can never leak into the new one.
"""
import csv
import io
import json
import shutil
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Union

import librosa as lb
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (
    SAMPLE_RATE,
    TRAIN_SIZE, TEST_SIZE,
    SEED,
    LABEL,
    N_AUG_COPIES,
    DOWN_SR,
    TRAIN_COMPRESSION, TEST_COMPRESSION,
    TRAIN_SNR, TEST_SNR,
    PROJECT_ROOT,
    FOLDER_PATH,
    SPLIT_PATH,
    NOISE_FOLDER_PATH, MP3_FOLDER_PATH, DOWNSAMPLE_FOLDER_PATH,
    MANIFEST_PATH,
    SCALER_PATH,
    AUGMENTATION_NAMES,
    MAX_DURATION
)

SPLITS: Tuple[str, str] = ("train", "test")

@dataclass
class Dataset:
    """A set of audio clips with their labels and names, kept index-aligned.

    Attributes:
        signals: Mono float32 signals at ``SAMPLE_RATE``.
        labels: Class label of each clip, one of ``LABEL``.
        names: File stem of each clip, e.g. ``"222361__ajaysm__xiaoluo-65"``
            or ``"222361__ajaysm__xiaoluo-65_noise-20"`` for a copy.
        sources: Name of the original clip each clip comes from (equal to
            `names` for originals). Use it as the group in group-aware
            cross-validation, so a clip and its copies stay together.
        params: Augmentation value of each clip: the SNR in dB for noise,
            the VBR quality for mp3, the sample rate for downsample, and
            ``None`` for originals. Filled with ``None`` if not given.
    """
    signals: List[np.ndarray] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    params: List[int | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.params:
            self.params = [None] * len(self.signals)

    def __len__(self) -> int:
        return len(self.signals)

    def subset(self, indices: Sequence[int]) -> "Dataset":
        """Return a new Dataset with only the clips at `indices`, in that order."""
        return Dataset([self.signals[i] for i in indices], [self.labels[i] for i in indices],
                       [self.names[i] for i in indices], [self.sources[i] for i in indices],
                       [self.params[i] for i in indices])

    def by_param(self) -> Dict[int | None, "Dataset"]:
        """Split the clips by augmentation value, e.g. one Dataset per test SNR.

        Returns:
            Dict from value to the clips with that value, in the order the
            values first appear (for test copies: the order in config.py).
        """
        indices: Dict[int | None, List[int]] = {}
        for i, param in enumerate(self.params):
            indices.setdefault(param, []).append(i)
        return {param: self.subset(idx) for param, idx in indices.items()}


def merge_datasets(*datasets: Dataset) -> Dataset:
    """Concatenate several Datasets into one, e.g. originals plus noise copies.

    Example::

        train, test = prepare_data()
        train_all = merge_datasets(train["original"], train["noise"], train["mp3"])
    """
    merged = Dataset()
    for data in datasets:
        merged.signals.extend(data.signals)
        merged.labels.extend(data.labels)
        merged.names.extend(data.names)
        merged.sources.extend(data.sources)
        merged.params.extend(data.params)
    return merged


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(f_path: Union[Path, str], duration: float | None = None) -> np.ndarray:
    """Load an audio file as a mono float32 signal at ``SAMPLE_RATE``.

    Args:
        f_path: Path to the audio file.
        duration: Only load this many seconds. Defaults to the whole file.

    Returns:
        Real array of shape (n_samples,).
    """
    y = lb.load(Path(f_path), mono=True, sr=SAMPLE_RATE, duration=duration)[0]
    return y.astype(np.float32)


def parse_label(f_path: Union[Path, str]) -> str:
    """Read the class label from a dataset file name.

    File names follow ``<id>__<user>__<label>-<number>.wav``, e.g.
    ``222361__ajaysm__xiaoluo-65.wav`` -> ``"xiaoluo"``.

    Args:
        f_path: Path or file name of the clip.

    Returns:
        The class label.

    Raises:
        ValueError: If the parsed label is not one of ``LABEL``.
    """
    # "222361__ajaysm__xiaoluo-65" -> ['222361', 'ajaysm', 'xiaoluo-65']
    parts = [item for item in Path(f_path).stem.split('_') if item]
    # "xiaoluo-65" -> "xiaoluo"
    label = parts[-1].split('-')[0]
    if label not in LABEL:
        raise ValueError(f"Unknown label '{label}' in file name '{Path(f_path).name}'")
    return label


def list_audio_files(folder: Path = FOLDER_PATH) -> List[Path]:
    """Return the ``.wav`` files in `folder`, sorted by name.

    Sorting keeps the order (and therefore the train / test split) the same
    on every operating system. Non-audio files such as the dataset's
    readme are skipped.

    Raises:
        FileNotFoundError: If `folder` contains no ``.wav`` files.
    """
    files = sorted(Path(folder).glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files found in '{folder}'")
    return files


def load_dataset(folder: Path = FOLDER_PATH) -> Dataset:
    """Load every clip in `folder` together with its label and name."""
    data = Dataset()
    for f in tqdm(list_audio_files(folder), desc="Loading"):
        data.signals.append(load_data(f))
        data.labels.append(parse_label(f))
        data.names.append(f.stem)
        data.sources.append(f.stem)
        data.params.append(None)
    return data

# ---------------------------------------------------------------------------
# Signal cleaning
# ---------------------------------------------------------------------------

def trim_silence(signal: np.ndarray, top_db: float = 60) -> np.ndarray:
    """Remove leading and trailing silence.

    Args:
        signal: Mono signal.
        top_db: Samples more than `top_db` below the peak count as silence.

    Returns:
        The trimmed signal.
    """
    return lb.effects.trim(signal, top_db=top_db)[0]


def duration_standardization(
        signal: np.ndarray, sample_rate: int = SAMPLE_RATE, duration: float | None = None
) -> np.ndarray:
    """Cut or zero-pad a signal to exactly `duration` seconds.

    Longer signals lose their end; shorter ones get zeros appended at the
    end. Not used by the pipeline, which uses `clip_duration` instead, so
    short clips are never padded with silence.

    Args:
        signal: Mono signal.
        sample_rate: Sample rate of `signal` in Hz.
        duration: Target length in seconds.

    Returns:
        Array of shape ``(round(sample_rate * duration),)``.

    Raises:
        ValueError: If `duration` is ``None``.
    """
    if duration is None:
        raise ValueError("Forget to give target duration")
    n_samples = int(round(sample_rate * duration))
    return lb.util.fix_length(signal, size=n_samples)


def clip_duration(signal: np.ndarray, sample_rate: int = SAMPLE_RATE, max_duration: float | None = None
                  )-> np.ndarray:
    """Trim the silence from a signal, then cut it to at most `max_duration` seconds.

    Trimming first means the cut counts from the start of the sound, not
    from the start of the file. Only the end is cut, and shorter signals
    are returned as they are (never padded), so the frame-wise features
    are not filled with silent frames.

    Args:
        signal: Mono signal.
        sample_rate: Sample rate of `signal` in Hz.
        max_duration: Longest length to keep, in seconds.

    Returns:
        The trimmed signal, at most ``round(sample_rate * max_duration)``
        samples long.

    Raises:
        ValueError: If `max_duration` is ``None``.
    """
    if max_duration is None:
        raise ValueError("Forget to give max duration")
    signal = trim_silence(signal)

    n_target_samples = int(round(sample_rate * max_duration))
    if len(signal) > n_target_samples:
        return signal[:n_target_samples]
    
    return signal


def clip_all(data_sequence: List, sample_rate: int = SAMPLE_RATE, max_duration: float = MAX_DURATION) -> List:
    """Apply `clip_duration` to every signal in a list.

    The input list is not changed. The result stays a list, not one array,
    because the clips have different lengths after trimming.

    Example::

        clipped = Dataset(clip_all(data.signals, max_duration=1.0),
                          data.labels, data.names, data.sources, data.params)

    Args:
        data_sequence: Mono signals, e.g. ``Dataset.signals``.
        sample_rate: Sample rate of the signals in Hz.
        max_duration: Longest length to keep, in seconds. Defaults to
            ``MAX_DURATION`` from config.py.

    Returns:
        The trimmed and clipped signals, in the same order.

    Raises:
        ValueError: If `max_duration` is ``None``.
    """
    return [clip_duration(s, sample_rate, max_duration=max_duration) for s in data_sequence]


def clip_datasets(datasets: dict, max_duration: float) -> dict:
    """Trim and clip every clip of both splits, in place (see `clip_all`).

    Changes the Datasets inside `datasets` directly. Calling it twice on
    the same data clips the already-clipped signals again, so to compare
    several durations, load the data again for each one (or build new
    Datasets with `clip_all`, as `evaluation.search_best_duration` does).

    Args:
        datasets: ``{"train": {...}, "test": {...}}`` as `prepare_data`
            builds it: each split maps ``"original"`` and every name in
            ``AUGMENTATION_NAMES`` to a Dataset, or, for test copies, to a
            dict ``{level: Dataset}``.
        max_duration: Longest length to keep, in seconds.

    Returns:
        The same `datasets` dict, with clipped signals.

    Raises:
        KeyError: If a split is missing ``"original"`` or one of
            ``AUGMENTATION_NAMES``.
    """
    for split in SPLITS:
        datasets[split]["original"].signals = clip_all(data_sequence=datasets[split]["original"].signals, max_duration=max_duration)
        for name in AUGMENTATION_NAMES:
            sets = datasets[split][name]
            # train: one Dataset; test: {level: Dataset}
            for dataset in (sets.values() if isinstance(sets, dict) else [sets]):
                dataset.signals = clip_all(dataset.signals, max_duration=max_duration)

    return datasets


# ---------------------------------------------------------------------------
# Degradations
# ---------------------------------------------------------------------------

def white_noise(signal: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise at a given signal-to-noise ratio.

    Args:
        signal: Mono signal.
        snr: Signal-to-noise ratio in dB, relative to the signal's mean power.
        rng: Random generator, so the noise is reproducible.

    Returns:
        Noisy float32 signal with the same shape as `signal`.
    """
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / 10 ** (snr / 10)
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    return (signal + noise).astype(np.float32)


def mp3_encoder(signal: np.ndarray, quality: int, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode a signal to MP3 using LAME variable bitrate (through libsndfile).

    libsndfile writes the VBR (Xing / LAME) header, so decoders read back
    the exact original length. Encoders that stream without this header,
    like ``lameenc``, make decoders guess the length from the first frame's
    bitrate, and the decoded clip comes out cut short.

    Args:
        signal: Mono float signal in ``[-1, 1]``; values outside are clipped.
        quality: LAME VBR quality, 0 (best, highest bitrate) to 9 (worst,
            lowest bitrate).
        sample_rate: Sample rate of `signal` in Hz.

    Returns:
        The encoded MP3 file contents.

    Raises:
        ValueError: If `quality` is outside ``0..9``.
    """
    if not 0 <= quality <= 9:
        raise ValueError(f"'quality'({quality}) must be between 0 and 9")

    buffer = io.BytesIO()
    # libsndfile maps compression_level in [0, 1) to LAME VBR quality * 10.
    sf.write(buffer, np.clip(signal, -1.0, 1.0), sample_rate, format="MP3",
             bitrate_mode="VARIABLE", compression_level=quality / 10)
    return buffer.getvalue()


def down_sample(signal: np.ndarray, target_sr: int, orig_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resample a signal to a lower sample rate.

    Args:
        signal: Mono signal.
        target_sr: New sample rate in Hz.
        orig_sr: Sample rate of `signal` in Hz.

    Returns:
        Resampled float32 signal of about ``len(signal) * target_sr / orig_sr``
        samples.
    """
    return lb.resample(signal, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

# (signal, parameter, rng) -> (audio array or encoded bytes, sample rate)
AugmentFn = Callable[[np.ndarray, int, np.random.Generator], Tuple[Union[np.ndarray, bytes], int]]


@dataclass(frozen=True)
class Augmentation:
    """Describes one augmentation and where its output is written.

    Attributes:
        name: Short name, used as the file name tag, e.g. ``"noise"``.
        folder: Output folder; files go to ``folder / "train"`` and
            ``folder / "test"``.
        extension: File extension of the output, without the dot.
        apply: Creates one augmented copy for a given parameter.
        train_params: Draws the parameters for one training clip.
        test_params: Parameters applied to every test clip.
    """
    name: str
    folder: Path
    extension: str
    apply: AugmentFn
    train_params: Callable[[np.random.Generator], Sequence[int]]
    test_params: Sequence[int]


def draw_distinct(rng: np.random.Generator, low: int, high: int, n: int = N_AUG_COPIES) -> List[int]:
    """Draw `n` distinct integers from the inclusive range ``[low, high]``.

    Distinct values avoid two copies of a clip that are identical in
    settings and would overwrite each other on disk.

    Raises:
        ValueError: If the range holds fewer than `n` values.
    """
    if high - low + 1 < n:
        raise ValueError(f"Cannot draw {n} distinct values from [{low}, {high}]")
    return sorted(int(v) for v in rng.choice(np.arange(low, high + 1), size=n, replace=False))


AUGMENTATIONS: List[Augmentation] = [
    Augmentation(
        name="noise",
        folder=NOISE_FOLDER_PATH,
        extension="wav",
        apply=lambda signal, snr, rng: (white_noise(signal, snr, rng), SAMPLE_RATE),
        train_params=lambda rng: draw_distinct(rng, *TRAIN_SNR),
        test_params=TEST_SNR,
    ),
    Augmentation(
        name="mp3",
        folder=MP3_FOLDER_PATH,
        extension="mp3",
        apply=lambda signal, quality, rng: (mp3_encoder(signal, quality), SAMPLE_RATE),
        train_params=lambda rng: draw_distinct(rng, *TRAIN_COMPRESSION),
        test_params=TEST_COMPRESSION,
    ),
    Augmentation(
        name="downsample",
        folder=DOWNSAMPLE_FOLDER_PATH,
        extension="wav",
        apply=lambda signal, sr, rng: (down_sample(signal, sr), sr),
        train_params=lambda rng: DOWN_SR,
        test_params=DOWN_SR,
    ),
]


def save_audio(path: Path, audio: Union[np.ndarray, bytes], sample_rate: int) -> None:
    """Write a signal (or already-encoded bytes) to `path`, creating folders.

    Args:
        path: Output file. The format follows its extension.
        audio: Float signal, or encoded file contents such as MP3 bytes.
        sample_rate: Sample rate of `audio` in Hz. Ignored for bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(audio, bytes):
        path.write_bytes(audio)
    else:
        sf.write(path, audio, sample_rate)


def _manifest_path(path: Path) -> str:
    """Return `path` relative to ``PROJECT_ROOT`` (portable), or absolute if outside it."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def augment_split(
        data: Dataset, split: str, augmentation: Augmentation, rng: np.random.Generator
) -> Tuple[Dataset, List[Dict[str, str]]]:
    """Write the augmented copies of every clip in one split.

    The output folder is emptied first, so it only ever holds copies of the
    clips in `data`. Each copy is read back from its file with `load_data`,
    so the returned signals are exactly what `load_split` loads later:
    MP3 copies decoded, down-sampled copies resampled back to
    ``SAMPLE_RATE``.

    Args:
        data: Clips of the split.
        split: ``"train"`` or ``"test"``; picks the parameters and the
            output subfolder.
        augmentation: The augmentation to apply.
        rng: Random generator for drawing parameters and noise.

    Returns:
        ``(copies, rows)``: the augmented clips, and one manifest row per
        written file (see `write_manifest`).
    """
    if split not in SPLITS:
        raise ValueError(f"'split'({split}) must be one of {SPLITS}")

    out_dir = augmentation.folder / split
    shutil.rmtree(out_dir, ignore_errors=True)
    copies, rows = Dataset(), []
    for signal, label, name in tqdm(zip(data.signals, data.labels, data.names),
                                    total=len(data), desc=f"{augmentation.name} ({split})"):
        params = augmentation.train_params(rng) if split == "train" else augmentation.test_params
        for param in params:
            audio, sample_rate = augmentation.apply(signal, param, rng)
            path = out_dir / f"{name}_{augmentation.name}-{param}.{augmentation.extension}"
            save_audio(path, audio, sample_rate)

            copies.signals.append(load_data(path))
            copies.labels.append(label)
            copies.names.append(path.stem)
            copies.sources.append(name)
            copies.params.append(param)
            rows.append({"path": _manifest_path(path), "label": label, "source": name,
                         "split": split, "augmentation": augmentation.name, "param": str(param)})
    return copies, rows


def write_manifest(rows: List[Dict[str, str]], path: Path = MANIFEST_PATH) -> None:
    """Write manifest rows to a CSV file.

    Columns: ``path, label, source, split, augmentation, param``.
    ``path`` is relative to ``PROJECT_ROOT``; ``augmentation`` is
    ``"original"`` for the untouched dataset clips.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "split",
                                               "augmentation", "param"])
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def print_distribution(data: Dataset, title: str) -> None:
    """Print the number of clips per class in `data`."""
    print(f"{title} data: {len(data)}")
    counts = Counter(data.labels)
    for label in LABEL:
        print(f"\t{label} distribution: {counts.get(label, 0)}")


def save_split(train: Dataset, test: Dataset, path: Path = SPLIT_PATH) -> None:
    """Save which clip belongs to which split as a CSV.

    Columns: ``name, label, split, seed, test_size``. Only clip names are
    stored (no folders), so the file stays valid when the project is moved.
    ``seed`` and ``test_size`` record the config the split was made with.
    """
    rows = [{"name": name, "label": label, "split": split, "seed": SEED, "test_size": TEST_SIZE}
            for split, data in zip(SPLITS, (train, test))
            for label, name in zip(data.labels, data.names)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "label", "split", "seed", "test_size"])
        writer.writeheader()
        writer.writerows(rows)


def read_split(path: Path = SPLIT_PATH) -> List[Dict[str, str]]:
    """Read the rows saved by `save_split`.

    Raises:
        ValueError: If the file has no rows.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Split file '{path}' is empty")
    return rows


def apply_saved_split(data: Dataset, rows: List[Dict[str, str]]) -> Tuple[Dataset, Dataset]:
    """Divide `data` into train / test following saved split rows.

    Args:
        data: All loaded clips.
        rows: Rows from `read_split`.

    Returns:
        ``(train, test)``, in the order the clips appear in `rows`.

    Raises:
        FileNotFoundError: If a listed clip is not in `data`.
        ValueError: If a row has an unknown split, or a label that differs
            from the one in the clip's file name.

    Warns:
        UserWarning: If some clips in `data` are not listed (they are left
            out of both sets), or the split was made with a different
            ``SEED`` / ``TEST_SIZE`` than the current config.
    """
    index = {name: i for i, name in enumerate(data.names)}

    missing = [row["name"] for row in rows if row["name"] not in index]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} clip(s) in the split file are missing from the dataset, "
            f"e.g. '{missing[0]}'. Restore them, or make a new split with resplit=True.")

    listed = {row["name"] for row in rows}
    unlisted = [name for name in data.names if name not in listed]
    if unlisted:
        warnings.warn(
            f"{len(unlisted)} clip(s) are not in the split file and are left out, "
            f"e.g. '{unlisted[0]}'. Make a new split with resplit=True to include them.")

    if any(row.get("seed") not in (None, str(SEED))
           or (row.get("test_size") is not None and float(row["test_size"]) != TEST_SIZE)
           for row in rows):
        warnings.warn(
            "The saved split was made with a different SEED or TEST_SIZE than config.py. "
            "It is still used as saved; make a new split with resplit=True to apply the config.")

    indices: Dict[str, List[int]] = {split: [] for split in SPLITS}
    for row in rows:
        if row["split"] not in indices:
            raise ValueError(f"Unknown split '{row['split']}' for '{row['name']}' in the split file")
        i = index[row["name"]]
        if data.labels[i] != row["label"]:
            raise ValueError(f"'{row['name']}' is labelled '{row['label']}' in the split file "
                             f"but '{data.labels[i]}' by its file name")
        indices[row["split"]].append(i)

    return data.subset(indices["train"]), data.subset(indices["test"])


def split_data(
        data: Dataset, path: Path = SPLIT_PATH, resplit: bool = False
) -> Tuple[Dataset, Dataset]:
    """Split clips into train and test, reusing the saved split if there is one.

    If `path` exists (and `resplit` is False) the saved split is applied
    with `apply_saved_split`. Otherwise a new stratified split is made with
    ``TRAIN_SIZE``, ``TEST_SIZE`` and ``SEED`` from config, which keeps the
    class proportions equal in both sets, and saved to `path`.

    Args:
        data: All loaded clips.
        path: Split file to read from or write to.
        resplit: Ignore and overwrite an existing split file.

    Returns:
        ``(train, test)``.
    """
    if Path(path).exists() and not resplit:
        train, test = apply_saved_split(data, read_split(path))
        print(f"Using saved split from {path}")
    else:
        idx_train, idx_test = train_test_split(
            np.arange(len(data)), test_size=TEST_SIZE, train_size=TRAIN_SIZE,
            random_state=SEED, stratify=data.labels
        )
        train, test = data.subset(idx_train), data.subset(idx_test)
        save_split(train, test, path)
        print(f"Saved new split to {path}")

    print_distribution(train, "Train")
    print_distribution(test, "Test")

    return train, test


def load_split(
        split: str,
        augmentations: Sequence[str] = (),
        include_original: bool = True,
        manifest_path: Path = MANIFEST_PATH,
        split_path: Path = SPLIT_PATH,
) -> Dataset:
    """Load one split from the files written by `prepare_data`.

    Reads the manifest, so nothing is re-split or re-augmented. Every file
    is loaded at ``SAMPLE_RATE``: down-sampled copies are resampled back up,
    so they match the other clips in rate but keep no content above their
    lower Nyquist frequency, like a low-rate recording would.

    Args:
        split: ``"train"`` or ``"test"``.
        augmentations: Names of the augmented copies to include, e.g.
            ``["noise", "mp3"]``.
        include_original: Include the un-augmented clips.
        manifest_path: Manifest written by `prepare_data`.
        split_path: Split file, used to check the manifest is up to date.

    Returns:
        The clips; `Dataset.sources` names the original of each copy and
        `Dataset.params` its augmentation value. Call `Dataset.by_param` to
        separate test copies per level.

    Raises:
        FileNotFoundError: If the manifest does not exist yet.
        ValueError: If `split` or an augmentation name is unknown, or the
            manifest was built from a different split than `split_path`.
    """
    if split not in SPLITS:
        raise ValueError(f"'split'({split}) must be one of {SPLITS}")
    if isinstance(augmentations, str):
        augmentations = [augmentations]
    if not Path(manifest_path).exists():
        raise FileNotFoundError(f"No manifest at '{manifest_path}'. Run prepare_data() first.")

    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    available = {row["augmentation"] for row in rows} - {"original"}
    unknown = set(augmentations) - available
    if unknown:
        raise ValueError(f"Unknown augmentation(s) {sorted(unknown)}; "
                         f"the manifest has {sorted(available)}")

    if Path(split_path).exists():
        saved = {row["name"]: row["split"] for row in read_split(split_path)}
        built = {row["source"]: row["split"] for row in rows if row["augmentation"] == "original"}
        if saved != built:
            raise ValueError(f"The manifest was built from a different split than "
                             f"'{split_path}'. Run prepare_data() again.")

    wanted = set(augmentations) | ({"original"} if include_original else set())
    data = Dataset()
    for row in tqdm([r for r in rows if r["split"] == split and r["augmentation"] in wanted],
                    desc=f"Loading {split}"):
        path = PROJECT_ROOT / row["path"]
        data.signals.append(load_data(path))
        data.labels.append(row["label"])
        data.names.append(path.stem)
        data.sources.append(row["source"])
        data.params.append(int(row["param"]) if row["param"] else None)
    return data


# ---------------------------------------------------------------------------
# Feature summary and scaling
# ---------------------------------------------------------------------------

def summarize_feature(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Summarize a frame-wise feature over time.

    Frames set to NaN are skipped, e.g. the frames `spectral_flatness` marks
    as having too little energy in a band. A row that is NaN in every frame
    stays NaN.

    Args:
        x: Feature of shape ``(n_features, n_frames)``.

    Returns:
        ``(mean, std)`` over frames, each of shape ``(n_features,)``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows: handled by the caller
        mean = np.nanmean(x, axis=1, dtype=np.float32)
        std = np.nanstd(x, axis=1, dtype=np.float32)
    return mean, std


def fit_scaler(data: np.ndarray, path: Path | None = SCALER_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std, and optionally save them as JSON.

    Fit on the training features only, then reuse the result for the test
    features, so no test information leaks into training.

    Args:
        data: Training features of shape ``(n_samples, n_features)``.
        path: JSON file to save to. ``None`` skips saving.

    Returns:
        ``(mean, std)``, each of shape ``(n_features,)``. A std of zero
        (constant feature) is replaced by 1 to avoid dividing by zero.
    """
    mean = np.mean(data, axis=0, dtype=np.float32)
    std = np.std(data, axis=0, dtype=np.float32)
    std[std == 0] = 1.0

    if path is not None:
        with open(path, "w") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
    return mean, std


def load_scaler(path: Path = SCALER_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """Load the ``(mean, std)`` saved by `fit_scaler`."""
    with open(path, "r") as f:
        saved = json.load(f)
    return np.asarray(saved["mean"], dtype=np.float32), np.asarray(saved["std"], dtype=np.float32)


def z_score_standardization(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Scale features to zero mean and unit variance: ``(data - mean) / std``.

    Args:
        data: Features of shape ``(n_samples, n_features)``.
        mean: Per-feature mean from `fit_scaler` or `load_scaler`.
        std: Per-feature std from `fit_scaler` or `load_scaler`.

    Returns:
        float32 array with the same shape as `data`.
    """
    return ((data - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def prepare_data(
        folder: Path = FOLDER_PATH,
        augmentations: Sequence[Augmentation] = AUGMENTATIONS,
        manifest_path: Path = MANIFEST_PATH,
        split_path: Path = SPLIT_PATH,
        resplit: bool = False,
        max_duration: float | None = None
) -> Tuple[Dict[str, Dataset], Dict[str, Union[Dataset, Dict[int, Dataset]]]]:
    """Load, split and augment the dataset, then write the manifest.

    Args:
        folder: Folder with the original ``.wav`` clips. It is only read.
        augmentations: Augmentations to write; each goes to its own folder.
        manifest_path: Where to write the manifest CSV.
        split_path: Saved split to reuse, or to create if it does not exist.
        resplit: Make and save a new split even if `split_path` exists.
        max_duration: If given, trim and clip every returned clip to at
            most this many seconds with `clip_datasets`, after the copies
            are written. ``None`` keeps the full clips. The files on disk
            are never clipped. Needs `augmentations` to cover every name in
            ``AUGMENTATION_NAMES``.

    Returns:
        ``(train, test)``, each a dict from augmentation name to its clips::

            train["original"]    # un-augmented clips
            train["noise"]       # one entry per augmentation in `augmentations`
            train["mp3"]
            train["downsample"]

            test["original"]     # un-augmented clips
            test["noise"]        # dict per level: {30: Dataset, 20: Dataset, ...}
            test["mp3"]          # {2: Dataset, 5: Dataset, 9: Dataset}
            test["downsample"]   # {22050: Dataset, 16000: Dataset, 8000: Dataset}

        Test copies are split per level so a model can be scored on each one
        separately. Use `merge_datasets` to combine several training sets
        into one. All signals are at ``SAMPLE_RATE``.

    Note:
        With `max_duration`, the copies are made from the full-length
        originals and trimmed afterwards. Noise raises the silence floor,
        so noisy copies lose less to `trim_silence` than their originals.
    """
    # 1. LOAD AND SPLIT
    original = dict(zip(SPLITS, split_data(load_dataset(folder), split_path, resplit)))
    result: Dict[str, Dict] = {split: {"original": original[split]} for split in SPLITS}

    rows = [{"path": _manifest_path(Path(folder) / f"{name}.wav"), "label": label,
             "source": name, "split": split, "augmentation": "original", "param": ""}
            for split in SPLITS
            for label, name in zip(original[split].labels, original[split].names)]

    # 2. AUGMENT EACH SPLIT
    rng = np.random.default_rng(seed=SEED)
    for augmentation in augmentations:
        for split in SPLITS:
            copies, aug_rows = augment_split(original[split], split, augmentation, rng)
            # Test levels are fixed (e.g. every TEST_SNR), so keep them apart
            result[split][augmentation.name] = copies.by_param() if split == "test" else copies
            rows.extend(aug_rows)

    # 3. TRIM AND CLIP EACH SPLIT (in memory only)
    if not max_duration is None:
        result = clip_datasets(result, max_duration)

    # 4. MANIFEST
    write_manifest(rows, manifest_path)
    print(f"Wrote {len(rows)} entries to {manifest_path}")

    return result["train"], result["test"]


if __name__ == "__main__":
    prepare_data()
