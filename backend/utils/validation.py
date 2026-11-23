"""Checks an uploaded audio file and decodes it into a mono signal."""
from pathlib import Path

import librosa as lb
import numpy as np
import soundfile as sf
from fastapi import UploadFile

from backend.utils.errors import ErrorCode, api_error
from config import SAMPLE_RATE

MAX_SIZE = 10 * 1024 * 1024                 # bytes
ALLOWED_EXTENSIONS = [".mp3", ".wav"]
# Some training strokes (bangu) are only ~0.07 s long; 0.05 s still holds one
# full N_FFT frame (2048 samples at 44.1 kHz).
# Keep in sync with LIMITS in frontend/src/constants/errors.ts.
MIN_UPLOAD_DURATION = 0.05                  # seconds
MAX_UPLOAD_DURATION = 10                    # seconds
SILENCE_THRESHOLD = 1e-3                    # peak amplitude


def check_format(file: UploadFile) -> bool:
    ext = Path(file.filename or "").suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def check_size(file: UploadFile) -> bool:
    file.file.seek(0, 2)          # jump to the end
    size = file.file.tell()       # position = size in bytes
    file.file.seek(0)             # rewind
    return size <= MAX_SIZE


def is_silent(data: np.ndarray) -> bool:
    if data.size == 0:
        return True
    peak = np.max(np.abs(data - data.mean()))
    return peak < SILENCE_THRESHOLD


def validate_file(file: UploadFile) -> np.ndarray:
    """Check an upload and decode it.

    Returns:
        Mono float32 signal at ``SAMPLE_RATE``, like `preprocessing.load_data`.

    Raises:
        HTTPException: with ``detail={"code", "message"}`` (see `errors`):
            415 UNSUPPORTED_FORMAT, 413 FILE_TOO_LARGE, and 422 for
            CORRUPT_FILE, DURATION_TOO_SHORT, DURATION_TOO_LONG, SILENT_AUDIO.
    """
    if not check_format(file):
        raise api_error(415, ErrorCode.UNSUPPORTED_FORMAT,
                        f"Only {', '.join(ALLOWED_EXTENSIONS)} files are accepted")

    if not check_size(file):
        raise api_error(413, ErrorCode.FILE_TOO_LARGE, f"File larger than {MAX_SIZE // (1024 * 1024)} MB")

    try:
        sf.info(file.file)                    # header only: is it audio at all?
    except RuntimeError:
        raise api_error(422, ErrorCode.CORRUPT_FILE, "Not a valid audio file")

    # soundfile cannot resample; librosa decodes, mixes to mono and
    # resamples to SAMPLE_RATE, exactly as the training clips were loaded
    file.file.seek(0)
    try:
        data, _ = lb.load(file.file, sr=SAMPLE_RATE, mono=True)
    except Exception:
        raise api_error(422, ErrorCode.CORRUPT_FILE, "Could not decode audio")

    # Measured on the decoded signal: a header's duration can be wrong for
    # MP3 files without a VBR header
    duration = len(data) / SAMPLE_RATE
    if duration < MIN_UPLOAD_DURATION:
        raise api_error(422, ErrorCode.DURATION_TOO_SHORT,
                        f"Duration must be at least {MIN_UPLOAD_DURATION}s, got {duration:.2f}s")
    if duration > MAX_UPLOAD_DURATION:
        raise api_error(422, ErrorCode.DURATION_TOO_LONG,
                        f"Duration must be at most {MAX_UPLOAD_DURATION}s, got {duration:.2f}s")

    if is_silent(data):
        raise api_error(422, ErrorCode.SILENT_AUDIO, "Audio is silent")

    return data.astype(np.float32)
