"""Holds the trained models in memory and turns audio into predictions.

The models are loaded once, when the server starts (see ``server.lifespan``),
and the same instance answers every request. Prediction only reads the
models, so concurrent requests can share them safely.

Every request goes through the same steps as the training data, otherwise
the model sees features it was never trained on:

    mono 44.1 kHz signal -> align_to_onset -> clip_duration(max_duration)
    -> extract_features -> model.predict   (Standardized scales inside predict)
"""
import time

import numpy as np

from config import MAX_DURATION, SAMPLE_RATE
from feature_extraction import extract_features
from model import Model, load_model
from preprocessing import align_to_onset, clip_duration


class ClassifierService:
    """Trained models plus the preprocessing they were trained with.

    Args:
        models: Loaded models by short name, e.g. ``{"svm": ..., "knn": ...}``.
        max_duration: Seconds kept after onset alignment; must be the
            duration the models were trained with.
    """

    def __init__(self, model: Model, file: str, max_duration: float):
        self.model = model
        self.name = file
        self.max_duration = max_duration

    @classmethod
    def from_file(cls, file: str, max_duration: float | None = MAX_DURATION) -> "ClassifierService":
        """Load models saved with `model.save_model`, e.g. ``{"svm": "all_svm"}``.

        The clip duration is taken from `max_duration`, else from a
        ``max_duration`` attribute saved on the models, else from
        ``config.MAX_DURATION``.
        """
        model = load_model(file)
        return cls(model, file, max_duration)

    def preprocess(self, signal: np.ndarray) -> np.ndarray:
        """Apply the training-time preprocessing to one mono signal at ``SAMPLE_RATE``."""
        signal = align_to_onset(signal, SAMPLE_RATE)
        return clip_duration(signal, SAMPLE_RATE, max_duration=self.max_duration) if self.max_duration is not None else signal

    def predict(self, signal: np.ndarray) -> dict:
        """Classify one mono signal at ``SAMPLE_RATE`` with the model `model_name`.

        Raises:
            KeyError: If `model_name` was not loaded.
            ValueError: If the signal gives NaN/inf features (e.g. silence).
        """
        model = self.model
        start = time.perf_counter()

        clip = self.preprocess(signal)
        features = extract_features(clip)
        if not np.isfinite(features).all():
            raise ValueError("Could not compute features for this audio")
        label = model.predict(features[np.newaxis, :])[0]

        return {
            "label": str(label),
            "analysed_seconds": round(len(clip) / SAMPLE_RATE, 3),
            "processing_ms": round((time.perf_counter() - start) * 1000, 1),
        }
